#!/usr/bin/env python3
import os
import csv
import time
import shutil
import argparse
from collections import deque
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge

UR3E_JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint'
]

class DemonstrationRecorder(Node):
    def __init__(self, save_dir: str, rate_hz: float = 30.0):
        super().__init__('demonstration_recorder')
        self.save_dir = save_dir
        self.rate_hz = rate_hz
        self.bridge = CvBridge()

        os.makedirs(self.save_dir, exist_ok=True)
        self.episode_idx = self._get_next_episode_idx()

        # Exacte Home-positie ter controle bij de start van elke opname
        home_degrees = [
            -54.65,     # Base
            -125.08,    # Shoulder
            120.29,     # Elbow
            -109.11,    # Wrist 1
            -61.30,     # Wrist 2
            315.64      # Wrist 3
        ]
        self.absolute_home_q = np.array(home_degrees, dtype=np.float32) * (np.pi / 180.0)

        self.latest_q = None
        self.latest_wrench = None
        self.latest_img_bgr = None
        self.latest_gripper = 0.0  # 0.0 = helemaal open, 1.0 = helemaal dicht

        self.wrench_calib_buffer = deque(maxlen=30)
        self.wrench_filter_buffer = deque(maxlen=5)
        self.force_offset = np.zeros(3, dtype=np.float32)

        self.recording = False
        self.step_count = 0
        self.ram_buffer = []

        self.create_subscription(String, '/experiment/command', self.command_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.create_subscription(
            WrenchStamped, '/force_torque_sensor_broadcaster/wrench', self.wrench_cb, qos_profile_sensor_data
        )
        self.create_subscription(Image, '/camera/image_raw', self.image_cb, 10)
        self.create_subscription(Float32, '/gripper/state', self.gripper_cb, 10)

        self.timer = self.create_timer(1.0 / self.rate_hz, self.timer_cb)
        self.get_logger().info(
            f"Recorder gereed (incl. Gripper logging). Map: {self.save_dir} (Volgende: episode_{self.episode_idx:04d})"
        )

    def _get_next_episode_idx(self):
        existing = [d for d in os.listdir(self.save_dir) if d.startswith("episode_")]
        if not existing:
            return 0
        indices = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
        return max(indices) + 1 if indices else 0

    def joint_cb(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        if all(n in name_to_pos for n in UR3E_JOINT_NAMES):
            self.latest_q = np.array([name_to_pos[n] for n in UR3E_JOINT_NAMES], dtype=np.float32)

    def wrench_cb(self, msg: WrenchStamped):
        f = msg.wrench.force
        w = np.array([f.x, f.y, f.z], dtype=np.float32)
        self.wrench_calib_buffer.append(w)
        self.wrench_filter_buffer.append(w)
        self.latest_wrench = np.mean(self.wrench_filter_buffer, axis=0)

    def image_cb(self, msg: Image):
        self.latest_img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def gripper_cb(self, msg: Float32):
        self.latest_gripper = float(msg.data)

    def command_cb(self, msg: String):
        cmd = msg.data.strip().lower()
        if cmd == "toggle":
            if not self.recording:
                self.start_recording()
            else:
                self.stop_recording()
        elif cmd == "start" and not self.recording:
            self.start_recording()
        elif cmd == "stop" and not self.recording:
            pass
        elif cmd == "stop" and self.recording:
            self.stop_recording()
        elif cmd == "discard":
            self.discard_episode()

    def start_recording(self):
        missing = []
        if self.latest_q is None:
            missing.append("/joint_states")
        if self.latest_wrench is None:
            missing.append("/wrench")
        if self.latest_img_bgr is None:
            missing.append("/camera/image_raw")

        if missing:
            self.get_logger().warn(f"⚠️ Kan opname NIET starten! Wacht op: {', '.join(missing)}")
            return

        home_err = float(np.max(np.abs(self.latest_q - self.absolute_home_q)))
        if home_err > 0.06:
            self.get_logger().error(
                f"⛔ START GEWEIGERD! Robot staat niet in Home-positie (afwijking: {home_err * 180 / np.pi:.1f}°). "
                f"Druk eerst op [r] om naar Home te gaan en koppel daarna met [e]!"
            )
            return

        if len(self.wrench_calib_buffer) > 0:
            self.force_offset = np.mean(self.wrench_calib_buffer, axis=0)
        else:
            self.force_offset = self.latest_wrench.copy()

        self.ram_buffer.clear()
        self.step_count = 0
        self.recording = True
        self.get_logger().info(
            f"🟢 === OPNAME GESTART: episode_{self.episode_idx:04d} "
            f"(Force offset: {np.round(self.force_offset, 2)} N | Gripper: {self.latest_gripper:.2f}) ==="
        )

    def stop_recording(self):
        self.recording = False
        num_frames = len(self.ram_buffer)

        if num_frames < 15:
            self.get_logger().warn(f"⚠️ Episode te kort ({num_frames} frames), wordt genegeerd.")
            self.ram_buffer.clear()
            return

        ep_dir = os.path.join(self.save_dir, f"episode_{self.episode_idx:04d}")
        img_dir = os.path.join(ep_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        self.get_logger().info(f"💾 Bezig met opslaan van {num_frames} frames...")

        csv_path = os.path.join(ep_dir, "data.csv")
        with open(csv_path, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "timestamp", "q0", "q1", "q2", "q3", "q4", "q5",
                "gripper", "fx", "fy", "fz", "image_path"
            ])

            for idx, (ts, q_list, g_val, f_list, img_bgr) in enumerate(self.ram_buffer):
                img_filename = f"frame_{idx:06d}.jpg"
                img_path = os.path.join(img_dir, img_filename)
                cv2.imwrite(img_path, img_bgr)
                row = [ts] + q_list + [g_val] + f_list + [f"images/{img_filename}"]
                csv_writer.writerow(row)

        fz_vals = [abs(item[3][2]) for item in self.ram_buffer if abs(item[3][2]) > 1.0]
        mean_fz = float(np.mean(fz_vals)) if fz_vals else 0.0
        g_vals = [item[2] for item in self.ram_buffer]
        max_g = float(np.max(g_vals)) if g_vals else 0.0

        self.ram_buffer.clear()
        self.get_logger().info(
            f"🛑 === OPNAME OPGESLAGEN: episode_{self.episode_idx:04d} "
            f"({num_frames} stappen | Max Gripper: {max_g:.2f} | Gem. |Fz|: {mean_fz:.1f} N) ==="
        )
        self.episode_idx += 1

    def discard_episode(self):
        if self.recording:
            self.recording = False
            self.ram_buffer.clear()
            self.get_logger().info(f"🗑️  === HUIDIGE OPNAME GEANNULEERD: episode_{self.episode_idx:04d} ===")
            return

        if self.episode_idx == 0:
            self.get_logger().warn("Geen opgeslagen episode om te verwijderen.")
            return

        target_idx = self.episode_idx - 1
        ep_dir = os.path.join(self.save_dir, f"episode_{target_idx:04d}")
        if os.path.exists(ep_dir):
            shutil.rmtree(ep_dir)
        self.episode_idx = target_idx
        self.get_logger().info(
            f"🗑️  === EPISODE VERWIJDERD: episode_{target_idx:04d} (Volgende is weer {self.episode_idx:04d}) ==="
        )

    def timer_cb(self):
        if not self.recording:
            return

        q = self.latest_q
        w = self.latest_wrench
        img = self.latest_img_bgr
        g = self.latest_gripper

        if q is None or w is None or img is None:
            return

        f_calibrated = w - self.force_offset
        self.ram_buffer.append((
            time.time(),
            q.tolist(),
            float(g),
            f_calibrated.tolist(),
            img.copy()
        ))
        self.step_count += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-case', choices=['needle', 'pick_place', 'trajectory'], required=True)
    parser.add_argument('--demo-mode', choices=['haptic', 'no_haptic', 'haptic_wave', 'no_haptic_wave'], required=True)
    parser.add_argument('--rate', type=float, default=30.0)
    args = parser.parse_args()

    save_dir = os.path.expanduser(f"~/ros2_ws/raw_demonstrations/{args.use_case}/{args.demo_mode}")

    rclpy.init()
    node = DemonstrationRecorder(save_dir=save_dir, rate_hz=args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.recording:
            node.stop_recording()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()