#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Int32, String
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys, select, termios, tty, threading

class TouchPosePublisher(Node):
    def __init__(self):
        super().__init__('touch_publisher')
        
        self.pose_pub = self.create_publisher(PoseStamped, '/target_pose', 10)
        self.reset_pub = self.create_publisher(Bool, '/reset_home', 10)
        self.engage_pub = self.create_publisher(Bool, '/engage_orientation', 10)
        self.gripper_pub = self.create_publisher(Bool, '/gripper/cmd', 10)
        self.cmd_force_pub = self.create_publisher(WrenchStamped, '/touch/cmd_force', 10)
        self.command_pub = self.create_publisher(String, '/experiment/command', 10)
        
        self.vr_key_sub = self.create_subscription(String, '/vr_keyboard', self.vr_key_cb, 10)
        self.raw_pose_sub = self.create_subscription(PoseStamped, '/touch/raw_pose', self.raw_pose_cb, 10)
        # self.btn_sub = self.create_subscription(Int32, '/touch/buttons', self.button_cb, 10)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.wrench_sub = self.create_subscription(WrenchStamped, '/force_torque_sensor_broadcaster/wrench', self.wrench_cb, qos_profile_sensor_data)

        self.d = np.array([0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921])
        self.a = np.array([0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0])
        self.alpha = np.array([np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0])
        self.R_base_tool = np.eye(3)
        
        self.tare_external_force = np.zeros(3)
        self.is_tared = False
        self.tare_samples = []
        self.net_external_force = np.zeros(3)

        self.clutched = False
        self.dock_pos = None
        self.dock_rot = None
        
        self.clutch_offset_pos = np.zeros(3)
        self.clutch_offset_rot = R.identity()
        self.clutch_start_pos = None
        self.clutch_start_rot = None
        
        self.scale_pos = 2.0  # Verhoog dit van 1.0 naar bijv. 2.0 als je de beweging van de Omni wilt versterken
        self.prev_btn_state = 0
        self.running = True
        self.engaged = False
        self.raw_pos = None
        self.raw_rot = None
        
        self.startup_samples = []
        self.is_initialized = False
        self.raw_wrench_force = np.zeros(3)
        self.filtered_force = np.zeros(3)

        self.declare_parameter('enable_force_feedback', True)
        self.enable_force_feedback = self.get_parameter('enable_force_feedback').get_parameter_value().bool_value

        if not self.enable_force_feedback:
            self.get_logger().info(">>> FORCE FEEDBACK IS UITGESCHAKELD VOOR DEZE RUN <<<")

        self.orig_settings = termios.tcgetattr(sys.stdin)
        self.key_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.key_thread.start()

    def get_dh_matrix(self, theta, d, a, alpha):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([[ct, -st*ca, st*sa, a*ct], [st, ct*ca, -ct*sa, a*st], [0, sa, ca, d], [0, 0, 0, 1.0]])

    def joint_cb(self, msg):
        try:
            names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
            q = [msg.position[msg.name.index(n)] for n in names]
            T = np.eye(4)
            for i in range(6): T = T @ self.get_dh_matrix(q[i], self.d[i], self.a[i], self.alpha[i])
            self.R_base_tool = T[0:3, 0:3]
        except (ValueError, IndexError): pass

    def raw_pose_cb(self, msg):
        pos_raw = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        curr_pos = np.array([-pos_raw[2] / 1000.0, -pos_raw[0] / 1000.0, pos_raw[1] / 1000.0])
        
        q_raw = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        rot_raw = R.from_quat(q_raw)
        
        swap = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        curr_rot = R.from_matrix(swap @ rot_raw.as_matrix() @ swap.T)

        self.raw_pos = curr_pos
        self.raw_rot = curr_rot
        
        if not self.is_initialized:
            self.startup_samples.append(curr_pos)
            if len(self.startup_samples) >= 25:
                self.dock_pos = np.mean(self.startup_samples, axis=0)
                self.dock_rot = curr_rot
                self.is_initialized = True
                self.get_logger().info('>>> Haptic pen zero-point locked <<<')
            return
            
        self.publish_haptic_force()
        
        if not self.engaged:
            out_msg = PoseStamped()
            out_msg.header.stamp = self.get_clock().now().to_msg()
            out_msg.header.frame_id = 'base_link'
            out_msg.pose.orientation.w = 1.0
            self.pose_pub.publish(out_msg)
            return

        if self.clutched: return
        
        effective_pos = curr_pos - self.clutch_offset_pos
        effective_rot = self.clutch_offset_rot.inv() * curr_rot

        delta_pos = (effective_pos - self.dock_pos) * self.scale_pos
        delta_rot = self.dock_rot.inv() * effective_rot
        delta_q = delta_rot.as_quat()
        
        out_msg = PoseStamped()
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = 'base_link'
        out_msg.pose.position.x = float(delta_pos[0])
        out_msg.pose.position.y = float(delta_pos[1])
        out_msg.pose.position.z = float(delta_pos[2])
        out_msg.pose.orientation.x = float(delta_q[0])
        out_msg.pose.orientation.y = float(delta_q[1])
        out_msg.pose.orientation.z = float(delta_q[2])
        out_msg.pose.orientation.w = float(delta_q[3])
        self.pose_pub.publish(out_msg)

    def button_cb(self, msg):
        btn_val = msg.data
        if (btn_val & 1) and not (self.prev_btn_state & 1): self.gripper_pub.publish(Bool(data=True))
        elif (btn_val & 2) and not (self.prev_btn_state & 2): self.gripper_pub.publish(Bool(data=False))
        self.prev_btn_state = btn_val

    def wrench_cb(self, msg):
        sample = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        self.raw_wrench_force = 0.15 * sample + 0.85 * self.raw_wrench_force
        
        if not self.is_tared:
            self.tare_samples.append(self.raw_wrench_force)
            if len(self.tare_samples) >= 30:
                self.tare_external_force = np.mean(self.tare_samples, axis=0)
                self.is_tared = True
                self.get_logger().info('>>> Sensor zero-measurement completed <<<')

        self.net_external_force = self.raw_wrench_force - self.tare_external_force

    def publish_haptic_force(self):
        f_msg = WrenchStamped()
        f_msg.header.stamp = self.get_clock().now().to_msg()
        f_msg.header.frame_id = 'touch'
        
        if not self.enable_force_feedback or self.clutched:
            f_msg.wrench.force.x = 0.0
            f_msg.wrench.force.y = 0.0
            f_msg.wrench.force.z = 0.0
            self.cmd_force_pub.publish(f_msg)
            return

        target_force = np.zeros(3)
        if self.is_tared and self.engaged:
            net_force_base = self.R_base_tool @ self.net_external_force
            mag = np.linalg.norm(net_force_base)
            deadband = 2.0  # Verhoog dit van 1.0 naar bijv. 2.0 als je de Omni minder gevoelig wilt maken 
            
            if mag > deadband:
                eff = mag - deadband
                # Verhoog 0.2 naar bijv. 0.8 zodat de Omni de kracht veel sterker doorgeeft
                f_raw = -(net_force_base / mag) * (eff * 0.8) 
                f_mag = np.linalg.norm(f_raw)
                
                if f_mag > 0.001:
                    max_force = 5.0  # Verhoog eventueel de maximale grens (was 2.5)
                    scale = (max_force * np.tanh(f_mag / max_force)) / f_mag
                    target_force = f_raw * scale
                    
        current_mag = np.linalg.norm(self.filtered_force)
        target_mag = np.linalg.norm(target_force)
        
        if target_mag > current_mag:
            self.filtered_force = 0.4 * target_force + 0.6 * self.filtered_force
        else:
            self.filtered_force = 0.9 * target_force + 0.1 * self.filtered_force
            
        lateral_scale = 0.1  # Zet dit van 0.0 naar bijv. 0.3 als je ook zijdelingse krachten wilt voelen
        axial_scale = 0.3    # De hoofdas (loodrecht op het doek)
        
        f_msg.wrench.force.x = float(self.filtered_force[1]) * lateral_scale
        f_msg.wrench.force.y = -float(self.filtered_force[2]) * axial_scale
        f_msg.wrench.force.z = float(self.filtered_force[0]) * lateral_scale

        self.cmd_force_pub.publish(f_msg)
        
    def vr_key_cb(self, msg: String):
        self.process_key(msg.data)

    def process_key(self, key):
        if key == 'e':
            if self.raw_pos is None or self.raw_rot is None:
                self.get_logger().warn("⚠️ [e] ingedrukt, maar ik ontvang nog geen data van de Omni (/touch/raw_pose)!")
                return
            self.dock_pos = self.raw_pos.copy()
            self.dock_rot = self.raw_rot
            self.clutched = False
            self.clutch_offset_pos = np.zeros(3)
            self.clutch_offset_rot = R.identity()
            self.engaged = True
            self.is_tared = False
            self.tare_samples = []
            self.filtered_force = np.zeros(3)
            self.engage_pub.publish(Bool(data=True))
            self.get_logger().info('>>> [ENGAGED] Koppeling actief! Robot volgt nu de pen. <<<')

        elif key == 'r':
            if self.raw_pos is None or self.raw_rot is None:
                self.get_logger().warn("⚠️ [r] ingedrukt, maar ik ontvang nog geen data van de Omni!")
                return
            # Als hij nog niet engaged was, engage hem automatisch zodat Return-to-Home altijd werkt!
            if not self.engaged:
                self.engaged = True
                self.engage_pub.publish(Bool(data=True))
            self.dock_pos = self.raw_pos.copy()
            self.dock_rot = self.raw_rot
            self.clutch_offset_pos = np.zeros(3)
            self.clutch_offset_rot = R.identity()
            if self.clutched:
                self.clutch_start_pos = self.raw_pos.copy()
                self.clutch_start_rot = self.raw_rot
            self.filtered_force = np.zeros(3)
            self.reset_pub.publish(Bool(data=True))
            self.get_logger().info('>>> [HOME] Return-to-Home signaal verzonden! <<<')

        elif key == ' ':
            if not self.engaged:
                self.get_logger().warn("⚠️ Druk eerst op [e] om te koppelen voordat je de clutch (Spatiebalk) gebruikt.")
                return
            self.clutched = not self.clutched
            if self.clutched and self.raw_pos is not None:
                self.clutch_start_pos = self.raw_pos.copy()
                self.clutch_start_rot = self.raw_rot
                self.filtered_force = np.zeros(3)
                self.get_logger().info('⏸️  [CLUTCH AAN] Robot gepauzeerd (pen is vrij).')
            elif not self.clutched and self.clutch_start_pos is not None and self.raw_pos is not None:
                self.clutch_offset_pos += (self.raw_pos - self.clutch_start_pos)
                self.clutch_offset_rot = self.raw_rot * self.clutch_start_rot.inv() * self.clutch_offset_rot
                self.get_logger().info('▶️  [CLUTCH UIT] Robot volgt de pen weer.')

        elif key == 'c': 
            self.gripper_pub.publish(Bool(data=True))
            self.get_logger().info('✊ Gripper DICHT')
        elif key == 'o': 
            self.gripper_pub.publish(Bool(data=False))
            self.get_logger().info('🖐️ Gripper OPEN')

        elif key in ('\r', '\n'):
            self.get_logger().info('📩 [ENTER] ingedrukt -> Start/Stop signaal naar recorder gestuurd...')
            msg = String()
            msg.data = "toggle"
            self.command_pub.publish(msg)
        elif key == '1':
            self.get_logger().info('📩 [1] ingedrukt -> START signaal naar recorder gestuurd...')
            msg = String()
            msg.data = "start"
            self.command_pub.publish(msg)
        elif key == '2':
            self.get_logger().info('📩 [2] ingedrukt -> STOP signaal naar recorder gestuurd...')
            msg = String()
            msg.data = "stop"
            self.command_pub.publish(msg)
        elif key == 'x':
            self.get_logger().info('📩 [x] ingedrukt -> Laatste episode verwijderen...')
            msg = String()
            msg.data = "discard"
            self.command_pub.publish(msg)

    def keyboard_loop(self):
        self.get_logger().info("⌨️  Toetsenbord-luisteraar ACTIEF! Druk op [e], [r], [Spatie] of [ENTER].")
        tty.setcbreak(sys.stdin.fileno())
        try:
            while self.running and rclpy.ok():
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    key = sys.stdin.read(1)
                    if key == '\x03':  # Ctrl+C
                        break
                    self.process_key(key)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.orig_settings)

    def destroy_node(self):
        self.running = False
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.orig_settings)
        except Exception:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = TouchPosePublisher()
    try: 
        rclpy.spin(node)
    except KeyboardInterrupt: 
        pass
    finally:
        node.destroy_node()
        if rclpy.ok(): 
            rclpy.shutdown()

if __name__ == '__main__': 
    main()