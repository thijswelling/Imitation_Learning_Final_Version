#!/usr/bin/env python3
import os
import glob
import argparse
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def ur3e_forward_kinematics_z(q: np.ndarray, tcp_offset_z: float = 0.18) -> float:
    """Berekent de Z-hoogte (in meters) van de naaldpunt (TCP) uit de 6 gewrichtshoeken."""
    d = [0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921 + tcp_offset_z]
    a = [0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0]
    alpha = [np.pi / 2.0, 0.0, 0.0, np.pi / 2.0, -np.pi / 2.0, 0.0]

    T = np.eye(4, dtype=np.float64)
    for i in range(6):
        ct, st = np.cos(q[i]), np.sin(q[i])
        ca, sa = np.cos(alpha[i]), np.sin(alpha[i])
        Ti = np.array([
            [ct, -st * ca,  st * sa, a[i] * ct],
            [st,  ct * ca, -ct * sa, a[i] * st],
            [0.0,      sa,       ca,      d[i]],
            [0.0,     0.0,      0.0,       1.0]
        ], dtype=np.float64)
        T = T @ Ti

    return float(T[2, 3])


def detect_joint_cols(df: pd.DataFrame):
    possible_joint_cols = [
        ['q0', 'q1', 'q2', 'q3', 'q4', 'q5'],
        ['q1', 'q2', 'q3', 'q4', 'q5', 'q6'],
        ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    ]
    for cols in possible_joint_cols:
        if all(c in df.columns for c in cols):
            return cols
    return [c for c in df.columns if c not in ['timestamp', 'image_path', 'gripper', 'fx', 'fy', 'fz']][:6]


def build_dataset(raw_dir: str, repo_id: str, fps: int, use_forces: bool,
                  trim_idle: bool = True, hold_frames: int = 150, tcp_offset_z: float = 0.18):
    episode_dirs = sorted(glob.glob(os.path.join(raw_dir, "episode_*")))
    if not episode_dirs:
        raise ValueError(f"Geen episodes gevonden in {raw_dir}")

    force_cols = ['fx', 'fy', 'fz']

    # =========================================================================
    # PASS 1: Bereken de gemiddelde Eindkracht (|Fz|) en Einddiepte (Z) over alle demos
    # =========================================================================
    ep_end_forces = []
    ep_min_z_m = []
    valid_episodes = []

    for ep_dir in episode_dirs:
        csv_path = os.path.join(ep_dir, "data.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if len(df) < 15:
            continue

        joint_cols = detect_joint_cols(df)
        q_all = df[joint_cols].to_numpy(dtype=np.float32)
        f_df = df[force_cols].rolling(window=5, min_periods=1).mean()
        f_all = f_df.to_numpy(dtype=np.float32)

        # Kijk naar het laatste kwart van de demonstratie (waar de injectie op diepte is)
        tail_start = int(len(q_all) * 0.70)
        tail_fz = np.abs(f_all[tail_start:, 2])
        peak_tail_fz = float(np.percentile(tail_fz, 90))  # Robuust tegen 1-frame uitschieters

        z_tail = [ur3e_forward_kinematics_z(q, tcp_offset_z) for q in q_all[tail_start:]]
        min_z = float(np.min(z_tail))

        ep_end_forces.append(peak_tail_fz)
        ep_min_z_m.append(min_z)
        valid_episodes.append((ep_dir, df, joint_cols, q_all, f_all))

    mean_target_fz = float(np.mean(ep_end_forces))
    std_target_fz = float(np.std(ep_end_forces))
    mean_target_z_m = float(np.mean(ep_min_z_m))
    std_target_z_m = float(np.std(ep_min_z_m))

    print("\n======================================================================")
    print(f"📊 PASS 1 ANALYSE ({len(valid_episodes)} episodes in {os.path.basename(raw_dir)}):")
    print(f"  • Gemiddelde Demonstratie Eindkracht |Fz| : {mean_target_fz:.2f} ± {std_target_fz:.2f} N")
    print(f"  • Gemiddelde Demonstratie Einddiepte Z    : {mean_target_z_m*1000:.1f} ± {std_target_z_m*1000:.1f} mm")
    print("======================================================================\n")

    # =========================================================================
    # PASS 2: Bouw dataset en markeer alles vanaf het doelwit als KLAAR (dq = 0)
    # =========================================================================
    state_dim = 9 if use_forces else 6
    features = {
        "observation.images.top_view": {
            "dtype": "video",
            "shape": (240, 320, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["q1", "q2", "q3", "q4", "q5", "q6"] + (["fx", "fy", "fz"] if use_forces else []),
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["dq1", "dq2", "dq3", "dq4", "dq5", "dq6"],
        },
    }

    cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
    if cache_dir.exists():
        print(f"Oude dataset gevonden. Wordt opgeruimd: {cache_dir}")
        shutil.rmtree(cache_dir)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        robot_type="ur3e",
    )

    for ep_dir, df, joint_cols, q_all, f_all in valid_episodes:
        img_paths = df['image_path'].tolist()
        dq_all = q_all[1:] - q_all[:-1]

        start_idx = 0
        if trim_idle:
            moving_indices = np.where(np.linalg.norm(dq_all, axis=1) > 0.0008)[0]
            if len(moving_indices) > 0:
                start_idx = max(0, int(moving_indices[0]) - 2)

        # Zoek in de tweede helft van de episode het moment waarop het doel (kracht of diepte) is bereikt
        half_idx = max(start_idx + 10, int(len(df) * 0.55))
        goal_idx = len(df) - 1

        for t in range(half_idx, len(df) - 1):
            z_t = ur3e_forward_kinematics_z(q_all[t], tcp_offset_z)
            fz_t = abs(float(f_all[t, 2]))

            # Doelwit bereikt als de kracht >= 92% van de gemiddelde demonstratie-eindkracht is,
            # OF als zowel de diepte binnen 1.5 mm van de gemiddelde einddiepte zit én er duidelijk contact is
            if fz_t >= 0.92 * mean_target_fz or (z_t <= (mean_target_z_m + 0.0015) and fz_t >= 0.65 * mean_target_fz):
                goal_idx = t
                break

        frames_added = 0
        last_img_rgb = None
        goal_q = q_all[goal_idx].copy()
        goal_f = f_all[goal_idx].copy()

        # Als de kracht op goal_idx net iets lager was dan het gemiddelde, trekken we het anker naar de echte eindkracht
        if abs(goal_f[2]) < 0.85 * mean_target_fz:
            sign_fz = -1.0 if f_all[-1, 2] <= 0 else 1.0
            goal_f[2] = sign_fz * mean_target_fz

        # Bewaar de laatste neerwaartse bewegingsrichting om bij te hoge kracht subtiel terug te kunnen veren
        recent_dq = np.mean(dq_all[max(start_idx, goal_idx - 10):max(start_idx + 1, goal_idx)], axis=0)

        # Voeg de actieve bewegingsframes toe tot aan het doelwit (goal_idx)
        for t in range(start_idx, goal_idx):
            q_t = q_all[t]
            dq_t = dq_all[t]
            f_t = f_all[t]

            state_t = np.concatenate([q_t, f_t], axis=0) if use_forces else q_t

            img_full_path = os.path.join(ep_dir, img_paths[t])
            img_pil = Image.open(img_full_path).convert("RGB").resize((320, 240), Image.BILINEAR)
            img_rgb = np.array(img_pil)

            dataset.add_frame({
                "observation.images.top_view": img_rgb,
                "observation.state": state_t.astype(np.float32),
                "action": dq_t.astype(np.float32),
                "task": "ur3e_injection_task",
            })
            frames_added += 1
            last_img_rgb = img_rgb

        # Laad het camerabeeld op het exacte doelwit-moment
        goal_img_path = os.path.join(ep_dir, img_paths[goal_idx])
        goal_img_rgb = np.array(Image.open(goal_img_path).convert("RGB").resize((320, 240), Image.BILINEAR))

        # Voeg 150 'DOEL BEREIKT / KLAAR' frames toe (5.0 seconden op 30Hz):
        # - Bij 95% tot 115% van de doelkracht -> dq = [0, 0, 0, 0, 0, 0] (volledig stilhouden)
        # - Bij 115% tot 200% van de doelkracht (alleen voor use_forces) -> lichte terugveer-actie (-0.15 * recent_dq)
        #   zodat het model actief druk vermindert als het ooit te ver doorduwt!
        if frames_added > 0 and hold_frames > 0:
            zero_action = np.zeros(6, dtype=np.float32)
            for h_idx in range(hold_frames):
                ratio = h_idx / max(1, hold_frames - 1)
                if use_forces:
                    scale = 0.95 + 1.05 * ratio  # Schaalt van 0.95x tot 2.00x van de doelkracht (bijv. -9.8N tot -20.5N)
                    aug_f = goal_f * scale
                    hold_state = np.concatenate([goal_q, aug_f], axis=0).astype(np.float32)

                    if scale > 1.20:
                        # Subtiele terugveer-actie als de kracht boven de 120% van het doel komt
                        rebound_factor = min(0.25, (scale - 1.20) * 0.30)
                        hold_action = (-rebound_factor * recent_dq).astype(np.float32)
                    else:
                        hold_action = zero_action
                else:
                    hold_state = goal_q.astype(np.float32)
                    hold_action = zero_action

                dataset.add_frame({
                    "observation.images.top_view": goal_img_rgb,
                    "observation.state": hold_state,
                    "action": hold_action,
                    "task": "ur3e_injection_task",
                })
                frames_added += 1

        if frames_added > 0:
            dataset.save_episode()
            z_goal_mm = ur3e_forward_kinematics_z(goal_q, tcp_offset_z) * 1000.0
            print(
                f"Verwerkt: {os.path.basename(ep_dir)} | Doel op stap {goal_idx}/{len(df)-1} "
                f"(Z={z_goal_mm:.1f}mm, Fz={goal_f[2]:.2f}N) + {hold_frames} stop-frames"
            )

    print(f"\n✅ Dataset succesvol aangemaakt: {repo_id} (state_dim={state_dim})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-case', choices=['needle', 'pick_place', 'trajectory'], required=True)
    parser.add_argument('--demo-mode', choices=['haptic', 'no_haptic'], required=True)
    parser.add_argument('--use-forces', action='store_true', help="Voeg krachtmetingen (Fx, Fy, Fz) toe")
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--hold-frames', type=int, default=150, help="Aantal extra stop-frames op het doelwit (150 = 5s)")
    parser.add_argument('--tcp-offset', type=float, default=0.18, help="TCP offset in meters")
    parser.add_argument('--no-trim', action='store_true')
    args = parser.parse_args()

    raw_dir = os.path.expanduser(f"~/ros2_ws/raw_demonstrations/{args.use_case}/{args.demo_mode}")
    force_tag = "with_force" if args.use_forces else "no_force"
    repo_id = f"local/ur3e_{args.use_case}_{args.demo_mode}_{force_tag}"

    build_dataset(
        raw_dir=raw_dir,
        repo_id=repo_id,
        fps=args.fps,
        use_forces=args.use_forces,
        trim_idle=not args.no_trim,
        hold_frames=args.hold_frames,
        tcp_offset_z=args.tcp_offset
    )