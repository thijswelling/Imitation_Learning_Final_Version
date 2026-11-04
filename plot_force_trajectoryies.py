#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_lerobot_episodes(base_folder, target_len=300):
    """Zoekt recursief naar alle data-bestanden binnen LeRobot episode mappen (bijv. episode_0000)."""
    if not base_folder or not os.path.exists(base_folder):
        print(f"⚠️ Map niet gevonden: {base_folder}")
        return []

    # Zoek naar alle mogelijke data bestanden in episode submappen
    files = glob.glob(os.path.join(base_folder, "**", "*data.csv"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(base_folder, "**", "*.csv"), recursive=True)
    if not files:
        # Misschien parquet bestanden? Check of pandas parquet aankan
        files = glob.glob(os.path.join(base_folder, "**", "*.parquet"), recursive=True)

    if not files:
        print(f"⚠️ Geen data bestanden gevonden in: {base_folder}")
        return []

    normalized_runs = []
    for f in files:
        try:
            if f.endswith('.parquet'):
                df = pd.read_parquet(f)
            else:
                df = pd.read_csv(f)
        except Exception as e:
            continue

        # Zoek naar krachtkolom
        col = None
        for candidate in ['fz', 'f_norm', 'observation.state.fz', 'action.fz']:
            if candidate in df.columns:
                col = candidate
                break
        if not col:
            # Neem de kolom die op 'fz' lijkt
            matching_cols = [c for c in df.columns if 'fz' in c.lower() or 'force' in c.lower()]
            if matching_cols:
                col = matching_cols[0]
            else:
                continue

        fz = np.abs(df[col].to_numpy(dtype=float))
        if len(fz) < 5:
            continue

        # Normaliseer naar 0 tot 100% taakvoortgang
        original_t = np.linspace(0, 1, len(fz))
        common_t = np.linspace(0, 1, target_len)
        resampled_fz = np.interp(common_t, original_t, fz)
        normalized_runs.append((common_t * 100.0, resampled_fz))

    print(f"  -> Succesvol {len(normalized_runs)} episodes geladen uit {base_folder}")
    return normalized_runs

def plot_group(ax, runs, color, label_prefix):
    if not runs:
        return
    for i, (t, fz) in enumerate(runs):
        lbl = f"{label_prefix} (runs, n={len(runs)})" if i == 0 else None
        ax.plot(t, fz, color=color, alpha=0.22, linewidth=1.1, label=lbl)

    all_fz = np.array([r[1] for r in runs])
    common_t = runs[0][0]
    mean_fz = np.mean(all_fz, axis=0)
    std_fz = np.std(all_fz, axis=0) if len(runs) > 1 else np.zeros_like(mean_fz)

    ax.plot(common_t, mean_fz, color=color, linewidth=2.4, label=f"{label_prefix} Mean")
    if len(runs) > 1:
        ax.fill_between(common_t, np.maximum(0, mean_fz - std_fz), mean_fz + std_fz, color=color, alpha=0.15)

def generate_trajectory_time_grid():
    base_eval = os.path.expanduser("~/ros2_ws/evaluation_results")
    
    # Correcte paden op basis van jouw directory-structuur
    conditions = [
        {
            "title": "1. Haptic Demo + No Force Input",
            "demo_path": os.path.expanduser("~/ros2_ws/raw_demonstrations/trajectory/haptic_wave"),
            "eval_path": os.path.join(base_eval, "trajectory_golf_haptic_no_force")
        },
        {
            "title": "2. No-Haptic Demo + No Force Input",
            "demo_path": os.path.expanduser("~/ros2_ws/raw_demonstrations/trajectory/no_haptic_wave"),
            "eval_path": os.path.join(base_eval, "trajectory_golf_no_haptic_no_force")
        },
        {
            "title": "3. Haptic Demo + WITH Force Input",
            "demo_path": os.path.expanduser("~/ros2_ws/raw_demonstrations/trajectory/haptic_wave"),
            "eval_path": os.path.join(base_eval, "trajectory_golf_haptic_with_force")
        },
        {
            "title": "4. No-Haptic Demo + WITH Force Input",
            "demo_path": os.path.expanduser("~/ros2_ws/raw_demonstrations/trajectory/no_haptic_wave"),
            "eval_path": os.path.join(base_eval, "trajectory_golf_no_haptic_with_force")
        }
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), dpi=140)
    fig.suptitle("2x2 Overview: Normalized Force Profiles (% of Task Duration - Trajectory)", fontsize=14, fontweight="bold")
    axes_flat = axes.flatten()

    for idx, cond in enumerate(conditions):
        ax = axes_flat[idx]
        print(f"\nVerwerken: {cond['title']}")
        
        demo_runs = load_lerobot_episodes(cond["demo_path"])
        eval_runs = load_lerobot_episodes(cond["eval_path"])

        # Blauw voor menselijke demonstratie, Rood voor AI Policy
        plot_group(ax, demo_runs, color='#1f77b4', label_prefix='Human Demo')
        plot_group(ax, eval_runs, color='#d62728', label_prefix='AI Policy')

        ax.set_title(cond["title"], fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Task Progress (%)" if idx >= 2 else "")
        ax.set_ylabel("Force |Fz| (N)" if idx % 2 == 0 else "")
        ax.set_ylim(0, 25.0)
        ax.set_xlim(0, 100)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.expanduser("~/ros2_ws/evaluation_results/2x2_force_normalized_trajectory_fixed.png")
    plt.savefig(output_path)
    print(f"\n✅ Nieuwe genormaliseerde trajectory grafiek opgeslagen op: {output_path}")

if __name__ == "__main__":
    trajectory_time_grid = generate_trajectory_grid_func = generate_trajectory_time_grid()