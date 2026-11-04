#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
import socket
import time

class RobotiqBridge(Node):
    def __init__(self):
        super().__init__('robotiq_bridge')
        
        self.declare_parameter('robot_ip', '192.168.1.102')
        self.robot_ip = self.get_parameter('robot_ip').get_parameter_value().string_value
        self.port = 63352
        self.sock = None

        # 0 = helemaal open, 255 = helemaal dicht
        self.gripper_pos = 0

        self.connect_gripper()

        # 1. Subscriber voor teleoperatie (True = direct helemaal dicht, False = direct helemaal open)
        self.sub_cmd = self.create_subscription(
            Bool,
            '/gripper/cmd',
            self.cmd_callback,
            10
        )

        # 2. Subscriber voor Policy Inference (0.0 = open, 1.0 = dicht)
        self.sub_target = self.create_subscription(
            Float32,
            '/gripper/target',
            self.target_callback,
            10
        )

        # 3. Publisher voor huidige gripper status (50 Hz) voor opname & inference
        self.state_pub = self.create_publisher(Float32, '/gripper/state', 10)
        self.timer = self.create_timer(0.02, self.publish_state)

        self.get_logger().info(f"Robotiq Bridge ready on {self.robot_ip}:{self.port} (Direct Open/Close Mode)")

    def connect_gripper(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)
            self.sock.connect((self.robot_ip, self.port))
            
            self.sock.sendall(b"SET ACT 1\n")
            time.sleep(0.1)
            self.sock.sendall(b"SET GTO 1\n")
            self.sock.sendall(b"SET SPE 255\n")
            self.sock.sendall(b"SET FOR 255\n")
            self.sock.sendall(b"SET POS 0\n")
            
            self.get_logger().info("Hand-E successfully activated and connected!")
        except Exception as e:
            self.get_logger().error(f"Error connecting to Robotiq gripper: {e}")

        # NIEUW: Geheugenvariabele (als hij eenmaal dicht is, blijft hij dicht tot de reset)
        self.gripper_latched = False

    def send_pos_to_hardware(self, new_pos: int):
        new_pos = int(max(0, min(255, new_pos)))
        if new_pos == self.gripper_pos:
            return
        self.gripper_pos = new_pos
        if self.sock:
            try:
                cmd_str = f"SET POS {self.gripper_pos}\n"
                self.sock.sendall(cmd_str.encode('utf-8'))
            except Exception as e:
                self.get_logger().error(f"Error sending gripper command: {e}")

    def cmd_callback(self, msg: Bool):
        # 1 druk op de knop: True = volledig sluiten (255), False = volledig openen (0)
        new_pos = 255 if msg.data else 0
        self.send_pos_to_hardware(new_pos)
        status_str = "DICHT (255)" if msg.data else "OPEN (0)"
        self.get_logger().info(f"[Teleop] Gripper direct naar: {status_str}")

    def target_callback(self, msg: Float32):
        norm_val = max(0.0, min(1.0, float(msg.data)))
        if norm_val >= 0.90:
            self.close_counter = getattr(self, 'close_counter', 0) + 1
        else:
            self.close_counter = 0

        # Sluit pas als het model 6 frames achter elkaar zeker is (voorkomt te vroeg sluiten in vlucht)
        if self.close_counter >= 1:
            self.gripper_latched = True
        if norm_val < 0.05:
            self.gripper_latched = False
            self.close_counter = 0

        new_pos = 255 if self.gripper_latched else 0
        self.send_pos_to_hardware(new_pos)

    def publish_state(self):
        msg = Float32()
        msg.data = float(self.gripper_pos) / 255.0
        self.state_pub.publish(msg)

    def destroy_node(self):
        if self.sock:
            self.sock.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RobotiqBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()