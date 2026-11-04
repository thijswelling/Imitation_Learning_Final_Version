#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import pandas as pd
import time
import os
from datetime import datetime

class TrialLogger(Node):
    def __init__(self):
        super().__init__('trial_logger')
        
        # --- Configuratie ---
        self.record_rate = 50.0  # 50 Hz log frequentie
        self.is_recording = False
        self.data_buffer = []
        self.trial_number = 1
        
        # Exacte, onveranderlijke volgorde van de joints!
        self.joint_names = [
            'shoulder_pan_joint', 
            'shoulder_lift_joint', 
            'elbow_joint', 
            'wrist_1_joint', 
            'wrist_2_joint', 
            'wrist_3_joint'
        ]
        
        self.save_dir = os.path.expanduser('~/teleop_experiment_data')
        os.makedirs(self.save_dir, exist_ok=True)

        # --- Laatst bekende states (Nu None om te checken of er echt data is!) ---
        self.latest_wrench = None
        self.latest_joints = None
        self.latest_tcp = None 
        
        # --- Subscribers ---
        self.sub_force = self.create_subscription(
            WrenchStamped, 
            '/force_torque_sensor_broadcaster/wrench', 
            self.force_cb, 10)
            
        self.sub_joints = self.create_subscription(
            JointState, 
            '/joint_states', 
            self.joints_cb, 10)
            
        # NIEUW: TCP Pose subscriber (Pas deze topic naam evt. aan naar jouw setup)
        self.sub_tcp = self.create_subscription(
            PoseStamped,
            '/tcp_pose', 
            self.tcp_cb, 10)

        self.sub_command = self.create_subscription(
            String,
            '/experiment/command',
            self.command_cb, 10)

        # --- Timer voor wegschrijven ---
        self.timer = self.create_timer(1.0 / self.record_rate, self.log_step)
        
        self.get_logger().info(f"Datalogger gestart (Optie B gereed). Wacht op 'start' commando op /experiment/command.")

    # --- Callbacks ---
    def force_cb(self, msg):
        self.latest_wrench = [
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        ]

    def joints_cb(self, msg):
        # Koppel de binnengekomen namen veilig aan hun waarden via zip()
        joint_dict = dict(zip(msg.name, msg.position))
        
        # Controleer of alle benodigde joints in het bericht zitten
        if all(name in joint_dict for name in self.joint_names):
            self.latest_joints = [joint_dict[name] for name in self.joint_names]

    def tcp_cb(self, msg):
        # Haal X, Y, Z en de rotaties (quaternions) op
        self.latest_tcp = [
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
            msg.pose.orientation.x, msg.pose.orientation.y, 
            msg.pose.orientation.z, msg.pose.orientation.w
        ]

    def command_cb(self, msg):
        cmd = msg.data.lower()
        if cmd == "start" and not self.is_recording:
            self.start_recording()
        elif cmd == "stop" and self.is_recording:
            self.stop_recording()

    # --- Start / Stop logica met Veiligheidscheck ---
    def start_recording(self):
        # 🚨 BEVEILIGING: Controleer of alle sensoren data sturen
        if self.latest_wrench is None or self.latest_joints is None or self.latest_tcp is None:
            self.get_logger().error("Kan niet starten! Een van de sensoren (Force, Joints of TCP) stuurt nog geen data.")
            return

        self.is_recording = True
        self.data_buffer = []
        self.start_time = time.time()
        self.get_logger().info(f"🔴 TRIAL {self.trial_number} GESTART! Neemt op met {self.record_rate}Hz.")

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        duration = time.time() - self.start_time
        self.get_logger().info(f"⏹️ TRIAL {self.trial_number} GESTOPT. Duur: {duration:.2f}s. Opslaan...")
        self.save_data()
        self.trial_number += 1

    # --- De 50Hz Loop ---
    def log_step(self):
        if not self.is_recording:
            return
            
        current_time = time.time() - self.start_time
        
        # Alles veilig samenvoegen in één rij
        row = [current_time] + self.latest_wrench + self.latest_joints + self.latest_tcp
        self.data_buffer.append(row)

    # --- Opslaan naar CSV ---
    def save_data(self):
        if len(self.data_buffer) == 0:
            self.get_logger().warn("Geen data verzameld.")
            return
            
        # Kolomnamen bijgewerkt met TCP data
        cols = ['time', 
                'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 
                'j1', 'j2', 'j3', 'j4', 'j5', 'j6',
                'tcp_x', 'tcp_y', 'tcp_z', 'tcp_qx', 'tcp_qy', 'tcp_qz', 'tcp_qw']
                
        df = pd.DataFrame(self.data_buffer, columns=cols)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trial_{self.trial_number}_{timestamp}.csv"
        filepath = os.path.join(self.save_dir, filename)
        
        df.to_csv(filepath, index=False)
        self.get_logger().info(f"✅ Data succesvol opgeslagen in: {filepath} ({len(df)} rijen)")

def main(args=None):
    rclpy.init(args=args)
    logger_node = TrialLogger()
    try:
        rclpy.spin(logger_node)
    except KeyboardInterrupt:
        pass
    finally:
        if logger_node.is_recording:
            logger_node.stop_recording()
        logger_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()