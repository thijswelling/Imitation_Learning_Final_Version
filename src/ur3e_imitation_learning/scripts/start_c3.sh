#!/usr/bin/env bash

# 1. Controleer argumenten en laad ROS 2
if [ -z "$1" ]; then echo "Usage: ./start_c3.sh <ROBOT_IP>"; exit 1; fi
ROBOT_IP=$1
WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
source "${WS_DIR}/install/setup.bash"

echo "=== Starting C3: VR View | Haptic ON ==="

# Trap om ALLES netjes af te sluiten (inclusief de OpenCV VR camera)
cleanup() { 
    echo -e "\nShutting down robot and VR stream..."; 
    # Zet schermbeveiliging weer AAN voor je laptop als je stopt
    xset s on +dpms 
    kill $(jobs -p) 2>/dev/null; 
    wait $(jobs -p) 2>/dev/null; 
}
trap cleanup SIGINT SIGTERM EXIT

# ==========================================
# DEEL 1: Robot & Haptics Opstarten
# ==========================================
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3e robot_ip:=${ROBOT_IP} launch_rviz:=false initial_joint_controller:=forward_velocity_controller &
sleep 5
ros2 run ur3e_teleop_control robotiq_bridge --ros-args -p robot_ip:=${ROBOT_IP} &
sleep 2
ros2 run ur3e_teleop_control task_space_controller &
sleep 1
ros2 run touch_haptics touch_haptic_bridge &
sleep 2

# ==========================================
# DEEL 2: DK2 & Logitech Brio Opstarten
# ==========================================
echo "Configuring Oculus DK2 and Camera..."
# 1. Zet Linux schermbeveiliging en beeldscherm-slaapstand UIT
xset s off
xset s noblank
xset -dpms

# 2. Probeer USB autosuspend uit te zetten
for d in /sys/bus/usb/devices/*/power/control; do [ -w "$d" ] && echo on > "$d" 2>/dev/null; done

HDMI_PORT=$(xrandr | grep -E "HDMI-[0-9] connected" | awk '{print $1}' | head -n 1)
if [ -n "$HDMI_PORT" ]; then
    xrandr --output "$HDMI_PORT" --mode 1080x1920 --rate 60.00 --rotate normal --pos 1920x0
    sleep 1
fi

BRIO_DEV=""
for dev in $(ls -v /dev/video* 2>/dev/null); do
    INFO=$(v4l2-ctl -d "$dev" --info 2>/dev/null)
    if echo "$INFO" | grep -qi "brio" && echo "$INFO" | grep -qi "video capture"; then
        if v4l2-ctl -d "$dev" --stream-mmap --stream-count=1 >/dev/null 2>&1; then
            BRIO_DEV="$dev"; break
        fi
    fi
done

if [ -n "$BRIO_DEV" ]; then
    CAM_INDEX=$(echo "$BRIO_DEV" | grep -o -E '[0-9]+')
    echo "Starting VR Stream on /dev/video${CAM_INDEX} (Background)..."
    # Start de headtracking in de achtergrond (&)
    python3 "${WS_DIR}/test_headtracking_clean.py" --cam "$CAM_INDEX" &
else
    echo "[WAARSCHUWING] Brio niet gevonden, VR stream start niet. Controleer USB!"
fi

sleep 2

# ==========================================
# DEEL 3: Start Teleoperatie Interface
# ==========================================
echo "Starting Teleop Interface (HAPTICS = ON)"
ros2 run ur3e_teleop_control touch_publisher --ros-args -p enable_force_feedback:=true