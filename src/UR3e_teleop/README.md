# UR3e Bilateral Haptic Teleoperation with 3D Systems Touch (ROS 2)

Complete bilateral teleoperation pipeline for a Universal Robots UR3e and Robotiq Hand-E gripper with real-time force feedback via a 3D Systems Touch haptic device in ROS 2 Humble.

---

## System Architecture

1. **Hardware Bridge (`touch_haptics/touch_haptic_bridge` - C++)**:
   - Runs an asynchronous 1000 Hz OpenHaptics scheduler loop for stable haptic force rendering.
   - Includes a startup hardware interlock to prevent streaming zero-vectors before device initialisation.
   - Topics: `/touch/raw_pose` (20 Hz), `/touch/buttons` (20 Hz), `/touch/cmd_force` (in).

2. **Gripper Bridge (`ur3e_teleop_control/robotiq_bridge` - Python)**:
   - Connects directly to the UR controller Robotiq daemon via TCP/IP socket (port 63352).
   - Translates high-level gripper commands into native Hand-E actuation commands with a conservative grasping force preset (~20–25 N) to protect delicate objects like cables.
   - Topic: `/gripper/cmd` (in).

3. **Teleop Publisher (`ur3e_teleop_control/touch_publisher` - Python)**:
   - Handles stylus offsets, clutching, filtering, workspace scaling, and gripper button events.
   - Safety Interlock: locks pose streaming until the tare/engage key is triggered by the user.
   - Rotates wrist F/T sensor wrenches into the robot's `base_link` frame using forward kinematics (DH parameters).
   - Maps 3D interaction forces to counter-forces on the Touch stylus with a 2.2 N deadband and EWMA smoothing.
   - Topics: `/target_pose` (out), `/gripper/cmd` (out), `/touch/cmd_force` (out).

4. **Task-Space Controller (`ur3e_teleop_control/task_space_controller` - Python)**:
   - Closed-loop DLS (Damped Least Squares) Jacobian inverse kinematics.
   - Startup interlock: commands zero velocity until a valid target pose is received.
   - Postural null-space projection towards the home configuration.
   - Topic: `/forward_velocity_controller/commands`.

---

## Quick Start (Single Command)

Execute the startup script from the workspace root (pass your robot IP as an argument):

~/ros2_ws/start_teleop.sh <Robot_IP>    # <- change "<Robot_IP" to 192.168.0.134 (On teach pendant, find/change in Settings-Network-IP address)

Press `Ctrl+C` in this terminal to terminate all nodes and processes cleanly.

---

## Manual Startup Sequence (Alternative)

If debugging individual modules, launch each in a separate terminal:

1. **Terminal 1 (UR Driver)**:
   ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3e robot_ip:=<ROBOT_IP> launch_rviz:=false initial_joint_controller:=forward_velocity_controller

2. **Terminal 2 (Gripper Socket Bridge)**:
   ros2 run ur3e_teleop_control robotiq_bridge --ros-args -p robot_ip:=<ROBOT_IP>

3. **Terminal 3 (Kinematics Controller)**:
   ros2 run ur3e_teleop_control task_space_controller

4. **Terminal 4 (1000 Hz Haptic Bridge)**:
   ros2 run touch_haptics touch_haptic_bridge

5. **Terminal 5 (Interactive Teleop Node)**:
   ros2 run ur3e_teleop_control touch_publisher

---

## Controls & Keybindings (Active in Teleop Terminal)

- [e]       : Engage & Tare (Locks origin, activates orientation tracking, tares F/T sensor)
- [SPACE]   : Clutch toggle (Freezes robot pose; allows repositioning stylus)
- [c]       : Close gripper (or Stylus Button 1)
- [o]       : Open gripper (or Stylus Button 2)
- [r]       : Reset Home (Returns target reference to home position)
- [Ctrl+C]  : Safe shutdown
