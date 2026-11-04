#!/usr/bin/env bash
# INJECTION TASK / NEEDLE: FORCE FEEDBACK + RECORDER

if [ -z "$1" ]; then
    echo "Gebruik: bash start_c1.sh <ROBOT_IP> [VIDEO_DEVICE]"
    exit 1
fi

ROBOT_IP=$1
VIDEO_DEV=${2:-/dev/video0}
WS_DIR="$HOME/ros2_ws"

# Ruim eventuele hangende processen op
pkill -f v4l2_camera_node 2>/dev/null
pkill -f robotiq_bridge 2>/dev/null
pkill -f touch_haptic_bridge 2>/dev/null
pkill -f touch_publisher 2>/dev/null
pkill -f task_space_controller 2>/dev/null
pkill -f record_episodes 2>/dev/null
sleep 1

source /opt/ros/humble/setup.bash
source "${WS_DIR}/install/setup.bash"

echo "=== Teleoperatie & Opname starten voor UR3e Injection op IP: ${ROBOT_IP} ==="

cleanup() {
    echo -e "\nAlle teleop- en opname-nodes veilig afsluiten..."
    kill $(jobs -p) 2>/dev/null
    pkill -f v4l2_camera_node 2>/dev/null
    pkill -f robotiq_bridge 2>/dev/null
    pkill -f touch_haptic_bridge 2>/dev/null
    pkill -f record_episodes 2>/dev/null
    wait $(jobs -p) 2>/dev/null
    echo "Alles netjes afgesloten."
}
trap cleanup SIGINT SIGTERM EXIT

# 1. Start de Camera
echo "1. Camera starten op ${VIDEO_DEV}..."
ros2 run v4l2_camera v4l2_camera_node --ros-args \
  -p video_device:="${VIDEO_DEV}" \
  -p image_size:="[640,480]" \
  -p time_per_frame:="[1,30]" \
  -r __ns:=/camera < /dev/null &
sleep 2

# 2. Start de UR Robot Driver
echo "2. UR Driver starten... (Druk op Play op het Teach Pendant!)"
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3e robot_ip:=${ROBOT_IP} launch_rviz:=false initial_joint_controller:=forward_velocity_controller < /dev/null &
sleep 5 

# 3. Start de Robotiq Gripper Bridge
echo "3. Robotiq bridge starten..."
ros2 run ur3e_teleop_control robotiq_bridge --ros-args -p robot_ip:=${ROBOT_IP} < /dev/null &
sleep 2

# 4. Start de Task Space Controller
echo "4. Task Space Controller starten..."
ros2 run ur3e_teleop_control task_space_controller < /dev/null &
sleep 1

# 5. Start de Haptic Bridge
echo "5. Haptic bridge starten..."
ros2 run touch_haptics touch_haptic_bridge < /dev/null &
sleep 2

# 6. Start de Demonstration Recorder (voor needle / injection)
echo "6. Demonstration Recorder starten (needle - haptic)..."
ros2 run ur3e_imitation_learning record_episodes --use-case needle --demo-mode haptic < /dev/null &
sleep 1

echo -e "\n================================================================"
echo "  BEDIENING IN DEZE TERMINAL:"
echo "  [e]          = Engage (Koppel robot aan pen)"
echo "  [Spatiebalk] = Clutch AAN/UIT (Pauzeer robot / pen vrij bewegen)"
echo "  [r]          = Veilig terug naar Home-positie (Joint-Space)"
echo "  [ENTER]      = START / STOP opname van een episode"
echo "  [x]          = Verwijder huidige/laatste mislukte episode"
echo "  [Ctrl+C]     = Alles afsluiten"
echo -e "================================================================\n"

# 7. Start Haptic Publisher
ros2 run ur3e_teleop_control touch_publisher --ros-args -p enable_force_feedback:=true