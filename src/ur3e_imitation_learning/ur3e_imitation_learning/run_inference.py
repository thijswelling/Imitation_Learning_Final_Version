#!/usr/bin/env python3
# Autonomous execution of a trained LeRobot Diffusion Policy on the UR3e.
# Policy output: 6 relative joint-position increments (radians/action).
# Actuation: a separate joint-space P tracker converts target joint error
# into velocity commands on /forward_velocity_controller/commands.
# IMPORTANT: never run this node and the teleop controller simultaneously.
import os
import sys
import csv
import select
import termios
import tty
import time
import threading
import argparse
from collections import deque
import numpy as np
import torch
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cv2.setNumThreads(1)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float64MultiArray, Float32
from cv_bridge import CvBridge

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from pathlib import Path

UR3E_JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint'
]


# Compute an approximate TCP position for logging/evaluation only.
# Adjust tcp_offset_z if the end-effector/gripper/tool geometry changes.
def ur3e_forward_kinematics(q: np.ndarray, tcp_offset_z: float = 0.18) -> np.ndarray:
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

    return T[:3, 3].astype(np.float64)


class PolicyInferenceNode(Node):
    def __init__(self, checkpoint_path: str, use_forces: bool, rate_hz: float = 30.0,
                 max_steps: int = 240, num_inference_steps: int = 24,
                 speed_scale: float = 1.0, kp_track: float = 12.0,
                 tcp_offset_z: float = 0.18, results_dir: str = None,
                 show_camera: bool = True, contact_fz_thresh: float = 1.5,
                 max_fz_limit: float = 22.0, max_lead_rad: float = 0.025,
                 dq_deadzone: float = 0.0022):
        super().__init__('policy_inference_node')
        self.use_forces = use_forces
        self.rate_hz = rate_hz
        self.dt = 1.0 / self.rate_hz
        self.max_steps = max_steps
        self.speed_scale = speed_scale
        self.kp_track = kp_track
        self.tcp_offset_z = tcp_offset_z
        self.show_camera = show_camera

        self.contact_fz_thresh = contact_fz_thresh
        self.max_fz_limit = max_fz_limit
        self.max_lead_rad = max_lead_rad
        self.dq_deadzone = dq_deadzone

        self.bridge = CvBridge()
        # GPU is preferred for real-time diffusion. The autocast code below
        # uses float16 and should be verified if attempting CPU-only execution.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_tag = "with_force" if self.use_forces else "no_force"
        if results_dir is None:
            self.results_dir = os.path.expanduser(f"~/ros2_ws/evaluation_results/policy_{model_tag}")
        else:
            self.results_dir = os.path.expanduser(results_dir)
        os.makedirs(self.results_dir, exist_ok=True)
        self.summary_csv_path = os.path.join(self.results_dir, "summary_results.csv")
        self.run_idx = self._get_next_run_idx()
        self.history_runs = self._load_existing_summary()

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_free, vram_total = torch.cuda.mem_get_info(0)
            self.get_logger().info(
                f"GPU actief: {gpu_name} (Vrij VRAM: {vram_free / 1e9:.2f} / {vram_total / 1e9:.2f} GB)"
            )

        self.get_logger().info(f"Model laden vanaf {checkpoint_path} op {self.device}...")

        real_checkpoint_path = os.path.realpath(checkpoint_path)
        if os.path.exists(os.path.join(real_checkpoint_path, "pretrained_model", "config.json")):
            real_checkpoint_path = os.path.join(real_checkpoint_path, "pretrained_model")

        os.environ["HF_HUB_OFFLINE"] = "1"

        # Load trained weights/config from LeRobot checkpoint. The matching
        # LeRobot code revision is needed for compatible checkpoint loading.
        self.policy = DiffusionPolicy.from_pretrained(Path(real_checkpoint_path))

        if num_inference_steps is not None and num_inference_steps > 0:
            self.policy.config.num_inference_steps = num_inference_steps
            self.policy.diffusion.num_inference_steps = num_inference_steps

        img_feat = self.policy.config.input_features.get("observation.images.top_view")
        if img_feat is not None and hasattr(img_feat, "shape"):
            _, self.expected_h, self.expected_w = img_feat.shape
        else:
            self.expected_h, self.expected_w = 240, 320

        self.use_gripper = True

        self.get_logger().info(
            f"Model verwacht beeldres (HxW): {self.expected_h}x{self.expected_w} | "
            f"Fixed Rollout: {self.max_steps} stappen | "
            f"Ruis-Deadzone ||dq|| < {self.dq_deadzone} rad (Geen kunstmatige kracht-rem!)"
        )

        self.policy.to(self.device)
        self.policy.eval()
        # Clear the policy action queue so this run starts from fresh state.
        self.policy.reset()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=real_checkpoint_path
        )
        home_degrees = [
            -54.65,     # Base
            -125.08,    # Shoulder
            120.29,     # Elbow
            -109.11,    # Wrist 1
            -61.30,     # Wrist 2
            315.64      # Wrist 3
        ]
        self.absolute_home_q = np.array(home_degrees, dtype=np.float32) * (np.pi / 180.0)

        self.warmup_gpu()

        self.data_lock = threading.Lock()
        self.latest_q = None
        self.joint_order_idx = None
        self.latest_img_rgb = None
        self.latest_gripper_state = 0.0
        # Keep 30 raw wrist force samples for start-of-run taring and
        # a 5-sample moving mean for the live force observation.
        self.wrench_calib_buffer = deque(maxlen=30)
        self.wrench_filter_buffer = deque(maxlen=5)
        self.force_offset = np.zeros(3, dtype=np.float32)

        self.mode = 'HOMING'
        self.step_count = 0
        self.had_contact = False
        self.contact_start_time = None
        self.last_dq_norm = 0.0

        # target_q accumulates predicted delta-q commands; each predicted
        # action can be tracked for up to ticks_per_action control callbacks.
        self.target_q = None
        self.substep_ticks = 0
        # At a 30 Hz timer, six ticks may occupy up to ~0.2 s per action;
        # max_steps is a count of policy actions, NOT a fixed wall time.
        self.ticks_per_action = 6
        self.max_delta_q = 0.04
        self.max_vel = 0.45

        self.run_start_wall_time = None
        self.active_motion_time = 0.0
        self.run_log = []

        self.create_subscription(JointState, '/joint_states', self.joint_cb, 1)
        self.create_subscription(
            WrenchStamped, '/force_torque_sensor_broadcaster/wrench',
            self.wrench_cb, qos_profile_sensor_data
        )
        self.create_subscription(Image, '/camera/image_raw', self.image_cb, 1)
        self.create_subscription(Float32, '/gripper/state', self.gripper_state_cb, 1)

        self.vel_pub = self.create_publisher(
            Float64MultiArray, '/forward_velocity_controller/commands', 10
        )
        self.gripper_pub = self.create_publisher(Float32, '/gripper/target', 10)

        self.wait_for_sensors()
        self.open_gripper()

        self.running = True
        self.orig_settings = termios.tcgetattr(sys.stdin)
        self.key_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.key_thread.start()

        self.timer = self.create_timer(self.dt, self.control_loop)
        self.print_instructions()

    def _get_next_run_idx(self):
        existing = [f for f in os.listdir(self.results_dir) if f.startswith("run_") and f.endswith("_dashboard.png")]
        if not existing:
            return 1
        indices = [int(f.split("_")[1]) for f in existing if f.split("_")[1].isdigit()]
        return max(indices) + 1 if indices else 1

    def _load_existing_summary(self):
        runs = []
        if os.path.exists(self.summary_csv_path):
            try:
                with open(self.summary_csv_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        runs.append({
                            "run_idx": int(row["run_idx"]),
                            "outcome": row["outcome"],
                            "wall_time_s": float(row["wall_time_s"]),
                            "active_time_s": float(row["active_time_s"]),
                            "mean_contact_fz": float(row.get("mean_contact_fz", 0.0)),
                            "std_contact_fz": float(row.get("std_contact_fz", 0.0)),
                            "peak_fz_n": float(row.get("peak_fz_n", row.get("peak_force_n", 0.0))),
                        })
            except Exception:
                pass
        return runs

    # Run the network before the first motion to reduce first-call latency.
    def warmup_gpu(self):
        self.get_logger().info("GPU warm-up uitvoeren (2 passes)...")
        base_state_dim = 6
        state_dim = base_state_dim + 3 if self.use_forces else base_state_dim

        dummy_img = torch.zeros((1, 3, self.expected_h, self.expected_w), dtype=torch.float32, device=self.device)
        dummy_state = torch.zeros((1, state_dim), dtype=torch.float32, device=self.device)
        batch = {
            "observation.images.top_view": dummy_img,
            "observation.state": dummy_state,
        }
        batch = self.preprocessor(batch)
        for i in range(2):
            self.policy.reset()
            t0 = time.perf_counter()
            with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.float16):
                _ = self.policy.select_action(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.get_logger().info(f"  Warm-up pass {i + 1}: {dt_ms:.1f} ms")
        self.policy.reset()

    def print_instructions(self):
        self.get_logger().info(
            "\n======================================================================\n"
            "  ZUIVERE END-TO-END EVALUATIE (Geen Kunstmatige Kracht-Rem):\n"
            "  [Spatiebalk] = Herkalibreer F/T-sensor & START VASTE RUN\n"
            "  [c] / [o]    = Gripper DICHT ([c]) of OPEN ([o])\n"
            "  [s]          = Label run als SUCCES (na afloop van vaste tijd)\n"
            "  [f]          = Label run als MISLUKT (of nood-afbreken tijdens run)\n"
            "  [r] of [h]   = Direct terug naar HOME-positie\n"
            "  [Ctrl+C]     = Stoppen\n"
            f"  Vaste duur   : {self.max_steps} stappen ({self.max_steps * self.dt:.1f} seconden)\n"
            f"  Ruisdrempel  : ||dq|| < {self.dq_deadzone} rad wordt gefilterd als stilstand\n"
            f"  Plots & CSVs : {self.results_dir}\n"
            "======================================================================\n"
            "--> Robot beweegt nu EERST automatisch naar de Home-positie..."
        )

    # Incoming JointState may list joints in any order; map by name once.
    def joint_cb(self, msg: JointState):
        if self.joint_order_idx is None:
            name_to_idx = {name: i for i, name in enumerate(msg.name)}
            if all(n in name_to_idx for n in UR3E_JOINT_NAMES):
                self.joint_order_idx = [name_to_idx[n] for n in UR3E_JOINT_NAMES]
            else:
                return
        pos = msg.position
        self.latest_q = np.array([pos[i] for i in self.joint_order_idx], dtype=np.float32)

    # Cache raw wrist F/T forces; only Fx, Fy, Fz are used here.
    def wrench_cb(self, msg: WrenchStamped):
        f = msg.wrench.force
        w = (f.x, f.y, f.z)
        self.wrench_calib_buffer.append(w)
        self.wrench_filter_buffer.append(w)

    def get_filtered_wrench(self):
        if not self.wrench_filter_buffer:
            return np.zeros(3, dtype=np.float32)
        return np.mean(self.wrench_filter_buffer, axis=0, dtype=np.float32)

    # Convert BGR ROS images to RGB, resized to model expected shape.
    def image_cb(self, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if bgr.shape[0] != self.expected_h or bgr.shape[1] != self.expected_w:
            bgr_resized = cv2.resize(bgr, (self.expected_w, self.expected_h), interpolation=cv2.INTER_LINEAR)
        else:
            bgr_resized = bgr
        self.latest_img_rgb = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB)

        if self.show_camera:
            vis = bgr.copy()
            color = (0, 255, 255) if self.mode == 'AWAIT_LABEL' else (0, 255, 0)
            cv2.putText(
                vis,
                f"Mode: {self.mode} | Step: {self.step_count}/{self.max_steps}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
            )
            cv2.imshow("Inference Camera", vis)
            cv2.waitKey(1)

    def gripper_state_cb(self, msg: Float32):
        self.latest_gripper_state = float(msg.data)

    def open_gripper(self):
        if self.use_gripper:
            g_msg = Float32()
            g_msg.data = 0.0
            self.gripper_pub.publish(g_msg)
            self.get_logger().info("🔓 Gripper OPEN gestuurd (0.0)")

    def close_gripper(self):
        if self.use_gripper:
            g_msg = Float32()
            g_msg.data = 1.0
            self.gripper_pub.publish(g_msg)
            self.get_logger().info("🔒 Gripper DICHT gestuurd (1.0)")

    def wait_for_sensors(self):
        self.get_logger().info("Wachten op eerste berichten van UR3e, F/T-sensor en Camera...")
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            q_ok = self.latest_q is not None
            img_ok = self.latest_img_rgb is not None
            w_ok = len(self.wrench_calib_buffer) >= 10
            if q_ok and img_ok and w_ok:
                break
            self.get_logger().info(
                f"Wachten op... q: {q_ok} | img: {img_ok} | wrench: {w_ok}",
                throttle_duration_sec=1.0
            )

    # Average recent force samples at home, with no external contact.
    # This is baseline subtraction, not full gravity compensation.
    def recalibrate_force_offset(self):
        if len(self.wrench_calib_buffer) > 0:
            self.force_offset = np.mean(self.wrench_calib_buffer, axis=0, dtype=np.float32)
        self.get_logger().info(
            f"⚖️  Krachtsensor gekalibreerd in Home-positie! Offset: {np.round(self.force_offset, 2)} N"
        )

    # Actuate the robot with six joint velocities (rad/s), not positions.
    def publish_vel(self, q_dot_list):
        cmd_msg = Float64MultiArray()
        cmd_msg.data = [float(x) for x in q_dot_list]
        self.vel_pub.publish(cmd_msg)

    def stop_robot(self):
        self.publish_vel([0.0] * 6)

    # Log real execution measurements for per-run CSVs and diagnostics.
    def record_datapoint(self, q_t: np.ndarray, f_cal: np.ndarray, dq_norm: float):
        if self.run_start_wall_time is None:
            self.run_start_wall_time = time.perf_counter()
        wall_t = time.perf_counter() - self.run_start_wall_time
        self.active_motion_time += self.dt

        xyz_m = ur3e_forward_kinematics(q_t, tcp_offset_z=self.tcp_offset_z)
        xyz_mm = xyz_m * 1000.0
        f_norm = float(np.linalg.norm(f_cal))

        self.run_log.append({
            "step": self.step_count,
            "wall_time_s": wall_t,
            "active_time_s": self.active_motion_time,
            "x_mm": float(xyz_mm[0]),
            "y_mm": float(xyz_mm[1]),
            "z_mm": float(xyz_mm[2]),
            "fx": float(f_cal[0]),
            "fy": float(f_cal[1]),
            "fz": float(f_cal[2]),
            "f_norm": f_norm,
            "dq_norm": float(dq_norm),
            "had_contact": int(self.had_contact),
            "gripper": float(self.latest_gripper_state),
            "q0": float(q_t[0]),
            "q1": float(q_t[1]),
            "q2": float(q_t[2]),
            "q3": float(q_t[3]),
            "q4": float(q_t[4]),
            "q5": float(q_t[5]),
        })

    # Stop motion, save raw CSV + summary metrics + diagnostic plot,
    # then return to HOMING before another run.
    def finish_run(self, outcome: str = "SUCCESS"):
        self.stop_robot()
        if not self.run_log:
            self.get_logger().warn("Geen meetdata verzameld in deze run om te plotten.")
            self.mode = 'HOMING'
            return

        wall_time_s = self.run_log[-1]["wall_time_s"]
        active_time_s = self.run_log[-1]["active_time_s"]
        peak_force_n = max(d["f_norm"] for d in self.run_log)
        peak_fz_n = max(abs(d["fz"]) for d in self.run_log)

        contact_fz = [abs(d["fz"]) for d in self.run_log if abs(d["fz"]) >= self.contact_fz_thresh]
        mean_contact_fz = float(np.mean(contact_fz)) if contact_fz else 0.0
        std_contact_fz = float(np.std(contact_fz)) if len(contact_fz) > 1 else 0.0

        plateau_samples = self.run_log[-60:] if len(self.run_log) >= 60 else self.run_log
        plateau_fz = [abs(d["fz"]) for d in plateau_samples]
        mean_plateau_fz = float(np.mean(plateau_fz)) if plateau_fz else 0.0
        std_plateau_fz = float(np.std(plateau_fz)) if len(plateau_fz) > 1 else 0.0

        end_x, end_y, end_z = self.run_log[-1]["x_mm"], self.run_log[-1]["y_mm"], self.run_log[-1]["z_mm"]

        run_csv_path = os.path.join(self.results_dir, f"run_{self.run_idx:04d}_data.csv")
        fieldnames = list(self.run_log[0].keys())
        with open(run_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.run_log)

        file_exists = os.path.exists(self.summary_csv_path)
        with open(self.summary_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "run_idx", "outcome", "steps", "wall_time_s", "active_time_s",
                    "mean_contact_fz", "std_contact_fz", "peak_fz_n", "peak_force_n",
                    "mean_plateau_fz", "std_plateau_fz",
                    "end_x_mm", "end_y_mm", "end_z_mm"
                ])
            writer.writerow([
                self.run_idx, outcome, self.step_count,
                round(wall_time_s, 3), round(active_time_s, 3),
                round(mean_contact_fz, 3), round(std_contact_fz, 3),
                round(peak_fz_n, 3), round(peak_force_n, 3),
                round(mean_plateau_fz, 3), round(std_plateau_fz, 3),
                round(end_x, 2), round(end_y, 2), round(end_z, 2)
            ])

        self.history_runs.append({
            "run_idx": self.run_idx,
            "outcome": outcome,
            "wall_time_s": wall_time_s,
            "active_time_s": active_time_s,
            "mean_contact_fz": mean_contact_fz,
            "std_contact_fz": std_contact_fz,
            "peak_fz_n": peak_fz_n,
        })

        plot_path = os.path.join(self.results_dir, f"run_{self.run_idx:04d}_dashboard.png")
        self.generate_dashboard_plot(
            plot_path, outcome, active_time_s, mean_contact_fz, std_contact_fz, peak_fz_n, peak_force_n
        )

        self.get_logger().info(
            f"\n📊 === RUN {self.run_idx:04d} ({outcome}) OPGESLAGEN ===\n"
            f"  • Vaste Evaluatietijd            : {active_time_s:.2f} s ({self.step_count} stappen)\n"
            f"  • Contactkracht |Fz| (Gem ± SD)  : {mean_contact_fz:.2f} ± {std_contact_fz:.2f} N\n"
            f"  • Piekkracht |Fz| (en ||F||)     : {peak_fz_n:.2f} N (Totaal: {peak_force_n:.2f} N)\n"
            f"  • Eindplateau laatste 2s (Gem±SD): {mean_plateau_fz:.2f} ± {std_plateau_fz:.2f} N\n"
            f"  • Plot opgeslagen in             : {plot_path}\n"
            f"=========================================================="
        )

        self.run_idx += 1
        self.run_log.clear()
        self.mode = 'HOMING'

    def generate_dashboard_plot(self, plot_path: str, outcome: str, active_time_s: float,
                                mean_fz: float, std_fz: float, peak_fz_n: float, peak_force_n: float):
        t = np.array([d["active_time_s"] for d in self.run_log])
        fx = np.array([d["fx"] for d in self.run_log])
        fy = np.array([d["fy"] for d in self.run_log])
        fz = np.array([d["fz"] for d in self.run_log])
        f_norm = np.array([d["f_norm"] for d in self.run_log])
        dq_norms = np.array([d["dq_norm"] for d in self.run_log])

        x = np.array([d["x_mm"] for d in self.run_log])
        y = np.array([d["y_mm"] for d in self.run_log])
        z = np.array([d["z_mm"] for d in self.run_log])

        fig = plt.figure(figsize=(15, 10), dpi=130)
        force_mode_str = "MET Force Input" if self.use_forces else "ZONDER Force Input"
        fig.suptitle(
            f"UR3e Injection Evaluatie — Run #{self.run_idx:04d} [{outcome}] ({force_mode_str})\n"
            f"Vaste Tijd: {active_time_s:.1f}s | Contact |Fz|: {mean_fz:.2f} ± {std_fz:.2f} N | Piek |Fz|: {peak_fz_n:.2f} N (||F||: {peak_force_n:.2f} N)",
            fontsize=13, fontweight='bold'
        )

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(t, np.abs(fz), label="|Fz| Insteekkracht (N)", color="#d62728", linewidth=2.2)
        ax1.plot(t, f_norm, label="||F|| Totale Kracht (N)", color="black", linestyle=":", linewidth=1.2)
        ax1.plot(t, np.abs(fx), label="|Fx| Lateraal (N)", color="#1f77b4", alpha=0.45)
        ax1.plot(t, np.abs(fy), label="|Fy| Lateraal (N)", color="#ff7f0e", alpha=0.45)

        if mean_fz > 0:
            ax1.axhline(mean_fz, color="green", linestyle="--", linewidth=1.8,
                        label=f"Gem. Contact |Fz|: {mean_fz:.2f} ± {std_fz:.2f} N")
            ax1.fill_between(t, max(0.0, mean_fz - std_fz), mean_fz + std_fz, color="green", alpha=0.15)

        peak_idx = int(np.argmax(np.abs(fz)))
        ax1.scatter([t[peak_idx]], [abs(fz[peak_idx])], color="darkred", s=70, zorder=5,
                    label=f"Piek |Fz|: {peak_fz_n:.2f} N")

        if self.contact_start_time is not None:
            ax1.axvline(self.contact_start_time, color="purple", linestyle="-.", linewidth=1.3,
                        label=f"Start Contact ({self.contact_start_time:.1f}s)")

        ax1.set_title("1. Krachtverloop, Piekkracht & Spreiding rond Gemiddelde")
        ax1.set_xlabel("Evaluatietijd (s)")
        ax1.set_ylabel("Kracht (N)")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="best", fontsize=8)

        ax2 = fig.add_subplot(2, 2, 2)
        color_z = "#2b5c8f"
        ax2.plot(t, z, color=color_z, linewidth=2.2, label="Z-hoogte TCP (mm)")
        ax2.set_xlabel("Evaluatietijd (s)")
        ax2.set_ylabel("Z-positie (mm)", color=color_z)
        ax2.tick_params(axis='y', labelcolor=color_z)
        ax2.grid(True, linestyle=":", alpha=0.6)

        ax2_twin = ax2.twinx()
        color_dq = "#2ca02c"
        ax2_twin.plot(t, dq_norms, color=color_dq, linestyle="--", linewidth=1.5, alpha=0.8, label="Model Actie ||dq||")
        ax2_twin.axhline(self.dq_deadzone, color="red", linestyle=":", linewidth=1.2, label=f"Deadzone ({self.dq_deadzone})")
        ax2_twin.set_ylabel("Model Actie-output ||dq|| (rad)", color=color_dq)
        ax2_twin.tick_params(axis='y', labelcolor=color_dq)
        ax2.set_title("2. Insteekdiepte (Z) & Model Stuur-activiteit (||dq||)")

        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax3.plot(x, y, z, color="gray", alpha=0.5, linewidth=1.2)
        sc3 = ax3.scatter(x, y, z, c=np.abs(fz), cmap="viridis", s=18)
        ax3.scatter([x[0]], [y[0]], [z[0]], color="green", s=60, label="Start")
        ax3.scatter([x[-1]], [y[-1]], [z[-1]], color="red", s=70, marker="^", label="Eind")
        plt.colorbar(sc3, ax=ax3, shrink=0.6, pad=0.1, label="|Fz| (N)")
        ax3.set_title("3. 3D XYZ-Traject & Krachtopbouw")
        ax3.set_xlabel("X (mm)")
        ax3.set_ylabel("Y (mm)")
        ax3.set_zlabel("Z (mm)")
        ax3.legend(loc="best", fontsize=8)

        ax4 = fig.add_subplot(2, 2, 4)
        run_labels = [f"R{r['run_idx']}" for r in self.history_runs]
        means = [r["mean_contact_fz"] for r in self.history_runs]
        stds = [r["std_contact_fz"] for r in self.history_runs]
        peaks = [r["peak_fz_n"] for r in self.history_runs]
        bar_colors = ["#2ca02c" if "SUCCESS" in r["outcome"] else "#d62728" for r in self.history_runs]

        x_pos = np.arange(len(run_labels))
        ax4.bar(x_pos, means, yerr=stds, capsize=5, color=bar_colors, alpha=0.75,
                edgecolor="black", label="Gem. Contact |Fz| ± SD")
        ax4.scatter(x_pos, peaks, color="darkred", marker="D", s=55, zorder=5, label="Piek |Fz| (N)")

        for i, (m, p) in enumerate(zip(means, peaks)):
            ax4.text(x_pos[i], p + 0.3, f"P:{p:.1f}N", ha='center', va='bottom', fontsize=7, color="darkred")

        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(run_labels)
        ax4.set_title("4. Overzicht Runs: Gemiddelde Kracht (± SD) & Piekkracht")
        ax4.set_xlabel("Run Nummer")
        ax4.set_ylabel("Kracht |Fz| (N)")
        ax4.grid(True, axis='y', linestyle=":", alpha=0.6)
        ax4.legend(loc="best", fontsize=8)

        plt.tight_layout(rect=[0, 0.01, 1, 0.93])
        plt.savefig(plot_path)
        plt.close(fig)

    # Clear contact detection, target and action counters for each trial.
    def _reset_run_state(self):
        self.step_count = 0
        self.had_contact = False
        self.contact_start_time = None
        self.last_dq_norm = 0.0
        self.target_q = None
        self.substep_ticks = 0

    # State machine: HOMING -> PAUSED -> RUNNING -> AWAIT_LABEL -> HOMING.
    # No task-space/Jacobian controller is used during policy execution.
    def control_loop(self):
        if self.latest_q is None or self.latest_img_rgb is None:
            return

        q_t = self.latest_q.copy()

        # Home using joint-space P tracking with a conservative velocity cap.
        if self.mode == 'HOMING':
            q_error = self.absolute_home_q - q_t
            max_err = float(np.max(np.abs(q_error)))

            if max_err < 0.012:
                self.stop_robot()
                time.sleep(0.3)
                self.recalibrate_force_offset()
                self.policy.reset()
                self._reset_run_state()
                self.mode = 'PAUSED'
                self.get_logger().info(
                    f">>> HOME BEREIKT! Klaar voor Run #{self.run_idx:04d}. Druk op [Spatiebalk] om te starten. <<<"
                )
            else:
                Kp_joint = 1.2
                q_dot = Kp_joint * q_error
                max_speed = float(np.max(np.abs(q_dot)))
                if max_speed > 0.35:
                    q_dot = q_dot * (0.35 / max_speed)
                self.publish_vel(q_dot)
            return

        if self.mode in ('PAUSED', 'AWAIT_LABEL'):
            self.stop_robot()
            return

        # Read current robot/sensor state and track or request a policy action.
        if self.mode == 'RUNNING':
            w_t = self.get_filtered_wrench()
            f_cal = w_t - self.force_offset

            # Software cutoff uses |Fz| ONLY, not the full force vector.
            if abs(f_cal[2]) > self.max_fz_limit:
                self.stop_robot()
                self.get_logger().warn(
                    f"⚠️ [NOOD-LIMIET] |Fz| = {abs(f_cal[2]):.2f} N > {self.max_fz_limit:.1f} N! Run gestopt en opgeslagen als OVERFORCE_FAIL."
                )
                self.finish_run(outcome="OVERFORCE_FAIL")
                return

            if self.target_q is None:
                self.target_q = q_t.copy()
                self.substep_ticks = 0

            # Continue tracking the prior action target without inference.
            # This makes policy-action rate lower than the ROS timer rate.
            if self.substep_ticks > 0:
                q_err = self.target_q - q_t
                if float(np.max(np.abs(q_err))) > 0.0004:
                    self.substep_ticks -= 1
                    q_dot = np.clip(self.kp_track * q_err, -self.max_vel, self.max_vel)
                    self.publish_vel(q_dot)
                    self.record_datapoint(q_t, f_cal, self.last_dq_norm)
                    return
                else:
                    self.substep_ticks = 0

            # LeRobot internally caches predicted action chunks. A new
            # denoising pass occurs when that queue is empty.
            if len(self.policy._queues["action"]) == 0:
                q_err = self.target_q - q_t
                if float(np.max(np.abs(q_err))) > 0.0004 and self.step_count > 0:
                    q_dot = np.clip(self.kp_track * q_err, -self.max_vel, self.max_vel)
                    self.publish_vel(q_dot)
                    self.record_datapoint(q_t, f_cal, self.last_dq_norm)
                    return
                self.stop_robot()

            img_t = self.latest_img_rgb.copy()

            # Match the dataset feature order: 6 joints then optional Fx/Fy/Fz.
            state_parts = [q_t]
            if self.use_forces:
                state_parts.append(f_cal)
            state_np = np.concatenate(state_parts, axis=0)

            img_tensor = (
                torch.from_numpy(img_t)
                .to(self.device, non_blocking=True)
                .permute(2, 0, 1)
                .contiguous()
                .float()
                .div_(255.0)
                .unsqueeze(0)
            )
            state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device, non_blocking=True)

            batch = {
                "observation.images.top_view": img_tensor,
                "observation.state": state_tensor,
            }
            batch = self.preprocessor(batch)

            inference_start = time.perf_counter()
            # Diffusion generates an action chunk from current observations.
            # This call can take longer than one 30 Hz control interval.
            with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.float16):
                action_tensor = self.policy.select_action(batch)
            inference_ms = (time.perf_counter() - inference_start) * 1000.0

            action_tensor = self.postprocessor(action_tensor)
            action_np = action_tensor.squeeze(0).cpu().numpy()

            dq = action_np[:6]
            # Clip each predicted joint increment before integrating it.
            dq_clipped = np.clip(dq, -self.max_delta_q, self.max_delta_q)
            raw_dq_norm = float(np.linalg.norm(dq_clipped))

            if not self.had_contact and abs(f_cal[2]) >= self.contact_fz_thresh:
                self.had_contact = True
                self.contact_start_time = self.active_motion_time
                self.get_logger().info(
                    f"📌 [CONTACT START] |Fz| = {abs(f_cal[2]):.2f} N op T = {self.active_motion_time:.2f}s"
                )

            # Als het model na contact aangeeft bijna stil te willen staan (onder de ruis-deadzone),
            # zet target_q op de huidige positie zodat de robot niet langzaam blijft doorkruipen door ruis.
            # Ignore tiny post-contact movements that would create creep.
            if self.had_contact and raw_dq_norm < self.dq_deadzone:
                dq_effective = np.zeros(6, dtype=np.float32)
                self.target_q = q_t.copy()
            else:
                dq_effective = dq_clipped
                self.target_q = self.target_q + (dq_effective * self.speed_scale)
                self.target_q = np.clip(self.target_q, q_t - self.max_lead_rad, q_t + self.max_lead_rad)

            self.last_dq_norm = raw_dq_norm
            self.substep_ticks = self.ticks_per_action - 1

            q_err = self.target_q - q_t
            # Closed-loop joint-position tracker outputs bounded velocities.
            q_dot = np.clip(self.kp_track * q_err, -self.max_vel, self.max_vel)
            self.publish_vel(q_dot)
            self.step_count += 1
            self.record_datapoint(q_t, f_cal, raw_dq_norm)

            self.get_logger().info(
                f"[RUN #{self.run_idx}] Step {self.step_count}/{self.max_steps} | "
                f"T: {self.active_motion_time:4.1f}s | Fz: {f_cal[2]:5.2f}N | "
                f"||dq||: {raw_dq_norm:.4f} | Inf: {inference_ms:4.1f}ms"
            )

            if self.step_count >= self.max_steps:
                self.stop_robot()
                self.mode = 'AWAIT_LABEL'
                peak_fz = max(abs(d["fz"]) for d in self.run_log)
                self.get_logger().info(
                    f"\n⏱️  === VASTE RUN-TIJD ({self.max_steps} STAPPEN) VOLTOOID! (Piek |Fz|: {peak_fz:.2f} N) ===\n"
                    f"👉 Druk nu op [s] als de naald RAAK zit (SUCCESS), of op [f] als hij MIS zat (FAILED)!"
                )

    # Keyboard: space=run/pause, s=success, f=failure, r/h=home, c/o=gripper.
    def process_key(self, key: str):
        if key == ' ':
            if self.mode == 'PAUSED':
                self.recalibrate_force_offset()
                self.policy.reset()
                self._reset_run_state()
                if self.latest_q is not None:
                    self.target_q = self.latest_q.copy()
                self.run_log.clear()
                self.run_start_wall_time = time.perf_counter()
                self.active_motion_time = 0.0
                self.mode = 'RUNNING'
                self.get_logger().info(
                    f"▶️  [START RUN #{self.run_idx:04d}] Robot draait nu zelfstandig {self.max_steps} stappen..."
                )
            elif self.mode == 'RUNNING':
                self.stop_robot()
                self.mode = 'PAUSED'
                self.get_logger().info("⏸️  [PAUZE] Robot stilgezet.")
        elif key in ('c', 'C'):
            self.close_gripper()
        elif key in ('o', 'O'):
            self.open_gripper()
        elif key in ('s', 'S'):
            if self.mode in ('AWAIT_LABEL', 'RUNNING', 'PAUSED') and len(self.run_log) > 0:
                self.get_logger().info("✅ [SUCCES GELABELD] Plots genereren & terug naar HOME...")
                self.finish_run(outcome="SUCCESS")
        elif key in ('f', 'F'):
            if self.mode in ('AWAIT_LABEL', 'RUNNING', 'PAUSED') and len(self.run_log) > 0:
                self.get_logger().info("❌ [FAIL GELABELD] Plots genereren & terug naar HOME...")
                self.finish_run(outcome="FAILED")
        elif key in ('r', 'h', 'R', 'H'):
            self.stop_robot()
            self.run_log.clear()
            self.get_logger().info("🏠 [HOME] Handmatige Return-to-Home geactiveerd!")
            self.mode = 'HOMING'

    def keyboard_loop(self):
        tty.setcbreak(sys.stdin.fileno())
        try:
            while self.running and rclpy.ok():
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    key = sys.stdin.read(1)
                    if key == '\x03':
                        break
                    self.process_key(key)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.orig_settings)

    def destroy_node(self):
        self.running = False
        try:
            self.stop_robot()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.orig_settings)
        except Exception:
            pass
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="Pad naar getrainde modelmap")
    parser.add_argument('--use-forces', action='store_true', help="Gebruik force-input")
    parser.add_argument('--rate', type=float, default=30.0, help="Control frequentie (Hz)")
    # max-steps counts policy actions; elapsed time also includes tracking
    # substeps, inference latency and callback delays (not exactly 8 s).
    parser.add_argument('--max-steps', type=int, default=240, help="Vast aantal stappen per run (240 = ~8s)")
    parser.add_argument('--num-inference-steps', type=int, default=24, help="Aantal DDIM denoising steps")
    parser.add_argument('--speed-scale', type=float, default=1.0, help="Stapgrootte-schaal")
    parser.add_argument('--kp', type=float, default=12.0, help="P-regelaar versterking")
    parser.add_argument('--tcp-offset', type=float, default=0.18, help="Lengte gripper + tool in m")
    parser.add_argument('--results-dir', type=str, default=None, help="Map voor plots en CSVs")
    parser.add_argument('--no-camera-window', action='store_true', help="Schakel OpenCV-venster uit")
    parser.add_argument('--contact-fz', type=float, default=1.5, help="Drempel |Fz| (N) voor contactberekening")
    parser.add_argument('--max-fz', type=float, default=22.0, help="Hardware veiligheidslimiet |Fz| (N)")
    parser.add_argument('--max-lead', type=float, default=0.025, help="Max voorsprong (rad) van target_q op q_t")
    parser.add_argument('--dq-deadzone', type=float, default=0.0022, help="Ruisdrempel ||dq|| bij contact")
    args = parser.parse_args()

    rclpy.init()
    node = PolicyInferenceNode(
        checkpoint_path=args.checkpoint,
        use_forces=args.use_forces,
        rate_hz=args.rate,
        max_steps=args.max_steps,
        num_inference_steps=args.num_inference_steps,
        speed_scale=args.speed_scale,
        kp_track=args.kp,
        tcp_offset_z=args.tcp_offset,
        results_dir=args.results_dir,
        show_camera=not args.no_camera_window,
        contact_fz_thresh=args.contact_fz,
        max_fz_limit=args.max_fz,
        max_lead_rad=args.max_lead,
        dq_deadzone=args.dq_deadzone
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop_robot()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()