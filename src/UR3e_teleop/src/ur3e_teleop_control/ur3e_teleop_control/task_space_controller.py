#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from scipy.spatial.transform import Rotation as R

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float64MultiArray

class UR3eClosedLoopPoseController(Node):
    def __init__(self):
        super().__init__("ur3e_task_space_controller")

        # Define the exact joint names of the UR3e robot arm in correct order
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        # Denavit-Hartenberg (DH) parameters for UR3e Forward Kinematics calculations
        self.d = np.array([0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921])
        self.a = np.array([0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0])
        self.alpha = np.array([np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0])

        self.current_q = None
        self.reference_q = None 
        self.reference_T = None 

        self.target_delta_pos = np.zeros(3)
        self.target_delta_rot = np.zeros(3)
        self.filtered_delta_pos = np.zeros(3)
        self.filtered_delta_rot = np.zeros(3)

        self.has_received_target = False
        self.is_engaged = False 

        self.tcp_pub = self.create_publisher(PoseStamped, "/tcp_pose", 10)

        # =======================================================================
        # VUL HIER JOUW IDEALE START-HOEKEN IN (LEES AF VAN HET UR TEACH SCHERM)
        # =======================================================================
        home_degrees = [
            -54.65,     # Base
            -125.08,    # Shoulder
            120.29,     # Elbow
            -109.11,    # Wrist 1
            -61.30,     # Wrist 2
            315.64      # Wrist 3
        ]
        # De code rekent het nu zelf netjes om naar radialen
        self.absolute_home_q = np.array(home_degrees) * (np.pi / 180.0)

        # --- Return to Home variabelen ---
        self.is_returning_home = False

        self.dt = 0.02 # Control loop time step (50 Hz)
        
        self.Kp_lin_max = 10.0
        self.Kp_ang_max = 12.0
        
        self.Kp_lin = 0.0
        self.Kp_ang = 0.0
        
        # PI-controller variables
        self.Ki_lin = 1.2
        self.error_lin_integral = np.zeros(3)

        self.K_posture = 0.2

        # ROS 2 Subscribers and Publishers
        self.create_subscription(JointState, "/joint_states", self.joint_state_cb, 10)
        self.create_subscription(PoseStamped, "/target_pose", self.pose_cb, 10)
        self.create_subscription(Bool, "/reset_home", self.reset_cb, 10)
        self.create_subscription(Bool, "/engage_orientation", self.engage_cb, 10)
        
        self.cmd_pub = self.create_publisher(Float64MultiArray, "/forward_velocity_controller/commands", 10)
        self.timer = self.create_timer(self.dt, self.control_loop)
        
        self.get_logger().info("UR3e Controller Active (Veilige Joint-Space Home ingeschakeld)")

    def joint_state_cb(self, msg):
        try:
            q = [0.0] * 6
            for idx, name in enumerate(self.joint_names):
                i = msg.name.index(name)
                q[idx] = msg.position[i]
            self.current_q = np.array(q)

            if not self.is_engaged:
                self.reference_q = self.current_q.copy()
                self.reference_T, _ = self.forward_kinematics(self.reference_q)
        except (ValueError, IndexError):
            pass

    def pose_cb(self, msg):
        if self.is_returning_home:
            return

        raw_pos = np.clip(np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ]), -0.25, 0.25)

        raw_q = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        raw_rotvec = R.from_quat(raw_q).as_rotvec()
        
        norm = np.linalg.norm(raw_rotvec)
        if norm > 1.5: raw_rotvec = raw_rotvec * (1.5 / norm)

        if not self.has_received_target:
            self.filtered_delta_pos = raw_pos.copy()
            self.filtered_delta_rot = raw_rotvec.copy()
            self.has_received_target = True

        self.target_delta_pos = raw_pos
        self.target_delta_rot = raw_rotvec

    def engage_cb(self, msg):
        if msg.data and self.current_q is not None:
            self.is_engaged = True
            self.is_returning_home = False
            self.reference_q = self.current_q.copy()
            self.reference_T, _ = self.forward_kinematics(self.reference_q)
            
            self.Kp_lin = 0.0
            self.Kp_ang = 0.0
            self.error_lin_integral = np.zeros(3)
            self.get_logger().info("Zero-point locked! (PI Control Active)")

    def reset_cb(self, msg):
        if msg.data and self.is_engaged and not self.is_returning_home:
            self.get_logger().info(">>> Return to ABSOLUTE Home geactiveerd (Veilige Joint-Space) <<<")
            self.is_returning_home = True
            self.error_lin_integral = np.zeros(3)

    def get_dh_matrix(self, theta, d, a, alpha):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,   sa,       ca,      d],
            [0,   0,        0,       1.0]
        ])

    def forward_kinematics(self, q):
        T_all = []
        T = np.eye(4)
        for i in range(6):
            Ti = self.get_dh_matrix(q[i], self.d[i], self.a[i], self.alpha[i])
            T = T @ Ti
            T_all.append(T)
        return T, T_all

    def compute_full_jacobian(self, T_all):
        J = np.zeros((6, 6))
        p_end = T_all[-1][0:3, 3] 
        T_prev = np.eye(4)
        for i in range(6):
            if i > 0: T_prev = T_all[i-1]
            z_i = T_prev[0:3, 2] 
            p_i = T_prev[0:3, 3] 
            J[0:3, i] = np.cross(z_i, p_end - p_i)
            J[3:6, i] = z_i 
        return J

    def get_orientation_error(self, R_target, R_curr):
        R_err = R_target @ R_curr.T
        angle = np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
        if abs(angle) < 1e-5: return np.zeros(3)
        axis = np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]
        ]) / (2.0 * np.sin(angle))
        return angle * axis

    def control_loop(self):
        if self.current_q is None: return

        # ==============================================================================
        # 1. AUTONOME RETURN TO HOME LOOP (PURE JOINT-SPACE = 100% VEILIG)
        # ==============================================================================
        if self.is_returning_home:
            # Bereken het verschil tussen waar we zijn en de absolute home hoeken
            q_error = self.absolute_home_q - self.current_q
            max_err = np.max(np.abs(q_error))
            
            # Als we er (bijna) zijn:
            if max_err < 0.02:
                self.is_returning_home = False
                self.get_logger().info(">>> Absolute Home Pose bereikt. Klaar voor de volgende trial. <<<")
                
                # Reset alle teleoperatie parameters naadloos
                self.reference_q = self.current_q.copy()
                self.reference_T, _ = self.forward_kinematics(self.reference_q)
                self.target_delta_pos = np.zeros(3)
                self.target_delta_rot = np.zeros(3)
                self.filtered_delta_pos = np.zeros(3)
                self.filtered_delta_rot = np.zeros(3)
                q_dot = np.zeros(6)
            else:
                # Proportionele controle: Beweeg de gewrichten direct naar het doel
                Kp_joint = 1.0
                q_dot = Kp_joint * q_error
                
                # Snelheidslimiet voor een vloeiende, zachte beweging (0.4 rad/s)
                max_speed = np.max(np.abs(q_dot))
                if max_speed > 0.4: 
                    q_dot = q_dot * (0.4 / max_speed)
                    
            # Stuur direct naar de robot en sla het complexe Cartesiaanse gedeelte over
            cmd_msg = Float64MultiArray()
            cmd_msg.data = q_dot.tolist()
            self.cmd_pub.publish(cmd_msg)
            return

        # ==============================================================================
        # 2. NORMALE TELEOPERATIE LOOP
        # ==============================================================================
        if self.reference_T is None: return

        if self.is_engaged:
            if self.Kp_lin < self.Kp_lin_max: self.Kp_lin += self.Kp_lin_max / (1.0 / self.dt)
            if self.Kp_ang < self.Kp_ang_max: self.Kp_ang += self.Kp_ang_max / (1.0 / self.dt)
        else:
            self.Kp_lin, self.Kp_ang = 0.0, 0.0

        if not self.has_received_target or not self.is_engaged:
            cmd_msg = Float64MultiArray(); cmd_msg.data = [0.0] * 6
            self.cmd_pub.publish(cmd_msg)
            return

        self.filtered_delta_pos = 0.30 * self.filtered_delta_pos + 0.70 * self.target_delta_pos
        self.filtered_delta_rot = 0.60 * self.filtered_delta_rot + 0.40 * self.target_delta_rot

        curr_T, T_all = self.forward_kinematics(self.current_q)
        curr_pos = curr_T[0:3, 3]
        curr_R = curr_T[0:3, 0:3]

        ref_pos = self.reference_T[0:3, 3]
        ref_R = self.reference_T[0:3, 0:3]

        # Zend de TCP Pose uit voor de logger!
        tcp_msg = PoseStamped()
        tcp_msg.pose.position.x = curr_pos[0]
        tcp_msg.pose.position.y = curr_pos[1]
        tcp_msg.pose.position.z = curr_pos[2]
        quat = R.from_matrix(curr_R).as_quat() # [x, y, z, w]
        tcp_msg.pose.orientation.x = quat[0]
        tcp_msg.pose.orientation.y = quat[1]
        tcp_msg.pose.orientation.z = quat[2]
        tcp_msg.pose.orientation.w = quat[3]
        self.tcp_pub.publish(tcp_msg)
        
        target_pos = ref_pos + self.filtered_delta_pos
        R_delta = R.from_rotvec(self.filtered_delta_rot).as_matrix()
        target_R = ref_R @ R_delta

        shoulder_pos = np.array([0.0, 0.0, self.d[0]])
        vec_from_shoulder = target_pos - shoulder_pos
        dist_from_shoulder = np.linalg.norm(vec_from_shoulder)
        max_reach = 0.5
        if dist_from_shoulder > max_reach:
            target_pos = shoulder_pos + vec_from_shoulder * (max_reach / dist_from_shoulder)

        error_lin = target_pos - curr_pos
        err_norm = np.linalg.norm(error_lin)
        if err_norm > 0.025:
            error_lin = error_lin * (0.025 / err_norm)
        
        if self.is_engaged:
            self.error_lin_integral += error_lin * self.dt
            self.error_lin_integral = np.clip(self.error_lin_integral, -0.03, 0.03)
            
        error_ang = self.get_orientation_error(target_R, curr_R)
        
        V_lin = (self.Kp_lin * error_lin) + (self.Ki_lin * self.error_lin_integral)
        V_ang = self.Kp_ang * error_ang
        V_task = np.hstack([V_lin, V_ang])

        J_6x6 = self.compute_full_jacobian(T_all)
        det_J = abs(np.linalg.det(J_6x6))
        
        base_damping = 0.02
        damping = base_damping + 0.2 * ((0.03 - det_J) / 0.03) if det_J < 0.03 else base_damping

        A = J_6x6 @ J_6x6.T + (damping ** 2) * np.eye(6)
        J_dls = J_6x6.T @ np.linalg.inv(A)

        q_null = self.K_posture * (self.absolute_home_q - self.current_q)
        N = np.eye(6) - (J_dls @ J_6x6)
        q_dot = (J_dls @ V_task) + (N @ q_null)

        max_speed = np.max(np.abs(q_dot))
        if max_speed > 3.0: q_dot = q_dot * (3.0 / max_speed)

        if np.linalg.norm(error_lin) < 0.002 and np.linalg.norm(error_ang) < 0.01:
            q_dot = np.zeros(6)

        cmd_msg = Float64MultiArray()
        cmd_msg.data = q_dot.tolist()
        self.cmd_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = UR3eClosedLoopPoseController()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == "__main__": main()