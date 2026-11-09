# UR3e Imitation Learning with Haptic Teleoperation and Diffusion Policy

This repository implements a demonstration-to-autonomous-execution workflow for a **Universal Robots UR3e**, **3D Systems Touch / Phantom Omni** haptic interface, wrist force/torque feedback, a camera, and **LeRobot Diffusion Policy**.

It is intended as a reference for collecting teleoperated demonstrations, converting them to the LeRobot format, training four policy conditions, and evaluating the trained policies on the robot. The two research tasks are **needle insertion** and **trajectory following**. Some launch scripts also include a preliminary `pick_place` configuration.

> **Read before controlling hardware.** The robot moves on the basis of external Python ROS 2 nodes and learned commands. Use an approved test environment, validate the robot/tool geometry, keep the workspace clear, verify the UR safety limits and teach-pendant emergency stop, and test slowly without a human in the robot workspace. The code has not been certified for safe operation. Never start teleoperation and policy inference simultaneously: both publish to `/forward_velocity_controller/commands`.
>
> **Reproducibility status.** The ROS 2 code and the three principal Python scripts are included. The proprietary OpenHaptics SDK, the Universal Robots driver, the LeRobot **source code/commit**, training data, camera devices, and trained checkpoints are **not** included. This repository does not pin a tested LeRobot commit. Follow the checks below and record the exact dependency versions before expecting bit-for-bit reproduction.

## 1. How the system works

```text
                          DEMONSTRATION COLLECTION
3D Systems Touch --> touch_haptic_bridge --> /touch/raw_pose --> touch_publisher
                                                              |
                                                       /target_pose
                                                              |
UR3e /joint_states ----------------------------------> task_space_controller
                                                              |
                                               /forward_velocity_controller/commands
                                                              |
                                                            UR3e

UR3e F/T sensor --> /force_torque_sensor_broadcaster/wrench --> touch_publisher
                          |                                         |
                          |                                  /touch/cmd_force
                          |                                         |
                          +--> demonstration_recorder        touch_haptic_bridge
                                                                    |
                                                               Touch motors

Camera /camera/image_raw ------> demonstration_recorder <------ /joint_states
                                         |
                                     Record 30 Hz
                                         |
                               CSV + JPEG episodes
                                         |
                                 Convert to LeRobot
                                         |
                                  Diffusion training
                                         |
                                Saved policy checkpoint
                                         |
                                 run_inference.py
                                         |
                        /forward_velocity_controller/commands
                                         |
                                  UR3e + evaluation
```

**During teleoperation:** the Touch's measured end-effector pose is mapped into a desired UR3e pose. A task-space feedback controller calculates joint-velocity commands using a damped Jacobian inverse. In the opposite direction, the UR3e wrist F/T readings are filtered, tared, transformed and scaled into haptic force commands.

**During autonomous execution:** Diffusion Policy sees an image, six joint positions, and optionally the three wrist force components. It predicts **six joint-position increments** (`delta_q`), not torques or velocities. A separate closed-loop P controller converts joint-position error into bounded joint-velocity commands.

## 2. Hardware and prerequisites

**Tested development context documented in the project:** Ubuntu **22.04**, ROS 2 **Humble**, UR3e, Phantom Omni / 3D Systems Touch, a Robotiq Hand-E gripper, and a USB camera. Actual installed dependency revisions are not recorded in this archive.

You will need:

1. A UR3e reachable from the Linux computer by Ethernet; the official **Universal Robots ROS 2 Driver**, configured with the **External Control URCap** and the proper robot network settings.
2. An active `/forward_velocity_controller` plus the joint-state and force/torque broadcasters. The UR driver's startup must activate the velocity controller rather than the default trajectory controller.
3. A 3D Systems Touch / Phantom Omni device **and** an installed, licensed/usable **OpenHaptics** SDK with the `HD`/`HDU` headers and libraries. `touch_haptics` will not link without these proprietary libraries.
4. A USB camera publishing `sensor_msgs/msg/Image` on `/camera/image_raw` at approximately 30 Hz. The launch scripts use the ROS 2 `v4l2_camera` package; a separate `webcam_publisher.py` is also included, but has a hard-coded `/dev/video2` device index.
5. A Robotiq gripper if you want gripper control (the bridge connects to the robot IP at TCP port **63352**). If your setup has no gripper, disable the bridge and check any commands that assume a gripper is present.
6. A CUDA-capable NVIDIA GPU for the included training script. Live inference strongly benefits from a GPU. CPU-only operation of the existing `torch.autocast(..., dtype=torch.float16)` path has not been validated.

Official driver setup, including URCap preparation: [Universal Robots ROS 2 driver documentation](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_robot_driver/ur_robot_driver/doc/usage/startup.html).

### 2.1 Install general ROS 2 tools

Install ROS 2 Humble following the official ROS instructions for Ubuntu 22.04. Then:

```bash
source /opt/ros/humble/setup.bash
sudo apt update
sudo apt install -y git python3-pip python3-venv python3-colcon-common-extensions \
    python3-rosdep python3-opencv ros-humble-cv-bridge \
    ros-humble-ros2-control ros-humble-ros2-controllers \
    ros-humble-v4l2-camera v4l-utils
```

Install the official Universal Robots ROS 2 driver according to its instructions for Humble. If installed separately, its environment must be sourced before using this project. Confirm it is available:

```bash
ros2 pkg prefix ur_robot_driver
ros2 pkg prefix v4l2_camera
```

Install OpenHaptics separately. Confirm the header and libraries are discoverable by your compiler/linker before building `touch_haptics`. Their install locations depend on your SDK package. The `touch_haptics/CMakeLists.txt` links against `HD` and `HDU`; it does not contain a portable SDK locator.

### 2.2 Clone and build this ROS 2 workspace

The scripts assume the workspace is located at **`~/ros2_ws`**. If you choose a different directory, update the hard-coded paths in the recorder, converter, launch scripts and training script.

On a **fresh machine** without an existing `~/ros2_ws`:

```bash
cd ~
git clone https://github.com/thijswelling/Imitation_Learning.git ros2_ws
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
sudo rosdep init             # only if rosdep has never been initialized
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro humble -r -y
colcon list
colcon build --symlink-install
source install/setup.bash
ros2 pkg prefix ur3e_imitation_learning
ros2 pkg prefix ur3e_teleop_control
ros2 pkg prefix touch_haptics
```

If you **already have** a ROS 2 workspace, merge the repository sources into that workspace carefully. Do not overwrite your local `src/`, launch configurations, or data without checking differences. The project packages are nested as follows:

```text
ros2_ws/
├── README.md
├── webcam_publisher.py
├── src/
│   ├── ur3e_imitation_learning/
│   │   ├── scripts/
│   │   │   ├── start_traj_haptic.sh
│   │   │   ├── start_traj_no_haptic.sh
│   │   │   ├── start_pnp_haptic.sh
│   │   │   ├── start_pnp_no_haptic.sh
│   │   │   ├── start_c1.sh ... start_c4.sh  (VR-oriented, see note below)
│   │   │   └── train_injection_augmented.sh
│   │   └── ur3e_imitation_learning/
│   │       ├── record_episodes.py
│   │       ├── convert_to_lerobot.py
│   │       └── run_inference.py
│   └── UR3e_teleop/src/
│       ├── ur3e_teleop_control/
│       └── touch_haptics/
├── raw_demonstrations/          (generated; ignored by Git)
├── lerobot_source/             (external, not in this repository)
├── evaluation_results/        (generated; ignored by Git)
├── build/ install/ log/       (generated; ignored by Git)
└── .venv/                    (optional environment; ignored by Git)
```

If the build fails with `cannot find -lHD` or `-lHDU`, OpenHaptics is not installed or its library path is missing. If the UR driver is unavailable, install/source it before launching the robot. Both are external dependencies.

## 3. Install the LeRobot training environment

The folder `lerobot_source/` in the original development workspace is **not shipped with this repository**. The source is needed for training and the `lerobot` Python imports used by conversion and inference.

The exact LeRobot commit originally used was not preserved. **Do not assume that the current LeRobot `main` branch accepts every flag or checkpoint from this project.** Obtain the compatible revision from the original development environment where possible, and record its commit hash.

Illustrative setup after choosing a compatible revision:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

git clone https://github.com/huggingface/lerobot.git lerobot_source
cd lerobot_source
# IMPORTANT: replace the next placeholder with the tested commit/tag:
# git checkout <TESTED_LEROBOT_COMMIT_OR_TAG>
python3 -m pip install --upgrade pip
python3 -m pip install -e .
python3 -m pip install pandas pillow opencv-python matplotlib

# Verify the imports expected by this particular project:
python3 - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
print('Required LeRobot imports are available')
PY

test -f src/lerobot/scripts/lerobot_train.py && echo 'Legacy training entry point found'
git rev-parse HEAD
```

If an import or training flag fails, the LeRobot version is **not compatible**. Select the tested LeRobot revision or adapt the project intentionally. Avoid silently changing model or dataset interfaces. In terminals running ROS nodes, source ROS, the workspace overlay and the same Python environment.

## 4. Start ROS 2, sensors and teleoperation

Start the driver with the real robot IP (substitute your own):

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur3e robot_ip:=<ROBOT_IP> launch_rviz:=false \
  initial_joint_controller:=forward_velocity_controller
```

Prepare and start the **External Control** program on the teach pendant according to the UR driver instructions. Verify:

```bash
ros2 control list_controllers
ros2 topic list | grep -E 'joint_states|wrench|forward_velocity_controller'
ros2 topic hz /joint_states
ros2 topic echo /force_torque_sensor_broadcaster/wrench --once
```

Open separate terminals and source the environment in each one:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/.venv/bin/activate    # if you created the venv
source ~/ros2_ws/install/setup.bash
```

Then launch components (in their own terminals):

```bash
# Terminal B. Camera, change video device for your setup.
ros2 run v4l2_camera v4l2_camera_node --ros-args \
  -p video_device:=/dev/video4 \
  -p image_size:="[640,480]" \
  -p time_per_frame:="[1,30]" \
  -r __ns:=/camera

# Terminal C. Robotiq gripper bridge, optional if no gripper.
ros2 run ur3e_teleop_control robotiq_bridge --ros-args -p robot_ip:=<ROBOT_IP>

# Terminal D. Joint velocity teleoperation controller.
ros2 run ur3e_teleop_control task_space_controller

# Terminal E. OpenHaptics device bridge.
ros2 run touch_haptics touch_haptic_bridge

# Terminal F. Start ONE recorder (example: needle, haptic demos).
ros2 run ur3e_imitation_learning record_episodes --use-case needle --demo-mode haptic

# Terminal G. Interactive teleoperation keyboard and force-feedback processing.
ros2 run ur3e_teleop_control touch_publisher --ros-args -p enable_force_feedback:=true
```

**Important:** these code blocks describe **separate terminals**. Do not run the teleop controller (`task_space_controller`) while running `run_inference` later. You can skip the separate recorder/teleop setup if a task-specific launch script already starts it. The provided `start_traj_*.sh` scripts run their own process cleanup via `pkill`; read them before executing in a shared ROS environment. The `start_c1.sh`–`start_c4.sh` files target a VR setup and reference a `test_headtracking_clean.py` script **not included** in the archive.

Check topic availability and rate:

```bash
ros2 topic hz /camera/image_raw
ros2 topic echo /joint_states --once
ros2 topic echo /touch/raw_pose --once
ros2 topic echo /target_pose --once
ros2 topic echo /touch/cmd_force --once
```

### 4.1 Record a demonstration

In the **touch_publisher terminal**:

| Key | Effect |
|---|---|
| `r` | Return robot to the configured home pose. |
| `e` | Engage teleoperation and restart haptic-force tare. |
| `Space` | Clutch/unclutch the Touch while preserving the robot goal. |
| `Enter` | Toggle start/stop of the episode recorder. |
| `1` / `2` | Explicitly start / stop recording. |
| `x` | Cancel current episode or delete most recently saved episode. |
| `c` / `o` | Close / open gripper. |

The recorder **refuses to start** if sensors are missing or the robot's six joints differ from the configured home values by more than `0.06 rad`. The values are specified in `record_episodes.py`; adapt them to your calibrated hardware setup if needed. Make sure there is **no external contact while taring**.

The recording timer is nominally **30 Hz**. The UR3e teleoperation controller operates separately at **50 Hz**. The recorder combines the latest received image, joint positions and forces, not exactly timestamp-synchronized measurements. Raw episodes are stored in:

```text
~/ros2_ws/raw_demonstrations/<use-case>/<demo-mode>/episode_0000/
    data.csv
    images/frame_000000.jpg
    images/frame_000001.jpg
    ...
```

Each CSV row contains a timestamp, six joint positions, a gripper command state, three tared wrist-force components, and the matching image path. Episodes shorter than 15 frames are discarded.

### 4.2 Collect both demonstration conditions

For each task (`needle` and `trajectory`), collect:

- `--demo-mode haptic`, with `enable_force_feedback:=true`.
- `--demo-mode no_haptic`, with `enable_force_feedback:=false`.

Stop the current recorder before starting one for a different task/condition. The project author reports 50 demonstrations per haptic condition for each of the two tasks. Collect enough demonstrations for your experimental plan and log exclusions separately. The **"30% deviation from the mean" filter is not implemented in `convert_to_lerobot.py`**; apply and document any such screening outside this converter.

## 5. Convert recordings to LeRobot datasets

Conversion is an **offline** step. It does not move the robot. Ensure your Python environment can import `lerobot` and `pandas`.

For each of the four conditions, use the **same demonstration folder** once with force observations and once without them:

```bash
cd ~/ros2_ws
source .venv/bin/activate
# Example: needle insertion, haptic demonstrations, with force observations
python3 src/ur3e_imitation_learning/ur3e_imitation_learning/convert_to_lerobot.py \
  --use-case needle --demo-mode haptic --use-forces

# Needle, haptic demonstrations, no force observations
python3 src/ur3e_imitation_learning/ur3e_imitation_learning/convert_to_lerobot.py \
  --use-case needle --demo-mode haptic

# Needle, no-haptic demonstrations, with force observations
python3 src/ur3e_imitation_learning/ur3e_imitation_learning/convert_to_lerobot.py \
  --use-case needle --demo-mode no_haptic --use-forces

# Needle, no-haptic demonstrations, no force observations
python3 src/ur3e_imitation_learning/ur3e_imitation_learning/convert_to_lerobot.py \
  --use-case needle --demo-mode no_haptic
```

For trajectory datasets, change `--use-case needle` to `--use-case trajectory` in these commands.

The dataset IDs have the form:

```text
local/ur3e_needle_haptic_with_force
local/ur3e_needle_haptic_no_force
local/ur3e_needle_no_haptic_with_force
local/ur3e_needle_no_haptic_no_force
```

The converter uses these steps:

1. Read every episode's `data.csv` and corresponding JPEG images; ignore episodes with fewer than 15 frames.
2. Smooth force measurements with a five-frame rolling mean.
3. Estimate the **mean terminal axial force and tool-tip depth** across demonstrations, using the last 30% of each episode.
4. Trim idle frames from the beginning; detect a task-completion point from force/depth thresholds and trim the original episode after that point.
5. Convert images to RGB at **320 × 240** pixels. Set observations to `image + 6 joints` or `image + 6 joints + Fx,Fy,Fz`.
6. Create six-dimensional action labels: **`q[t+1] - q[t]`** in radians/frame.
7. Add **150 synthetic terminal frames** by default. Without forces, actions are zero. With forces, synthetic observations increase toward 200% of estimated goal force; above 120% a small reverse action is labeled.
8. Save each processed demonstration as a LeRobot episode, with a nominal 30 fps dataset.

**Converter limitations:** the completion heuristic and hard-coded LeRobot task name `ur3e_injection_task` are currently **needle-specific but are used for every `--use-case`**. Adapt/validate this logic before claiming equivalence on trajectory following or pick-and-place. Synthetic high-force/retraction frames are generated training labels, not physically observed behavior. Conversion also **deletes an older local dataset with the same ID** in the LeRobot cache; back up anything you want to preserve.

## 6. Train Diffusion Policies

The provided `src/ur3e_imitation_learning/scripts/train_injection_augmented.sh` trains the **needle** use case. It selects the demo condition and force observation flag:

```bash
cd ~/ros2_ws
source .venv/bin/activate

# IMPORTANT: First open this script and fix its output path:
nano src/ur3e_imitation_learning/scripts/train_injection_augmented.sh
```

Specifically, **replace** the hard-coded setting:

```bash
out_dir="/home/thijs/outputs/injection_${DEMO_MODE}_${FORCE_MODE}_aug"
```

with a location on your own machine, for example:

```bash
out_dir="$HOME/ros2_ws/outputs/injection_${DEMO_MODE}_${FORCE_MODE}_aug"
```

The script executes `rm -rf "${out_dir}"` before starting, so it **overwrites previously trained outputs** with the same name. Check the path carefully. It also expects LeRobot's training file at `~/ros2_ws/lerobot_source/src/lerobot/scripts/lerobot_train.py` and requires a CUDA GPU.

After the path and LeRobot revision are checked, run:

```bash
bash src/ur3e_imitation_learning/scripts/train_injection_augmented.sh haptic no_force
bash src/ur3e_imitation_learning/scripts/train_injection_augmented.sh haptic with_force
bash src/ur3e_imitation_learning/scripts/train_injection_augmented.sh no_haptic no_force
bash src/ur3e_imitation_learning/scripts/train_injection_augmented.sh no_haptic with_force
```

Main training settings:

| Setting | Value |
|---|---:|
| Policy | LeRobot Diffusion Policy |
| Optimizer steps | 12,000 |
| Batch size | 8 |
| Input frames | 2 observations |
| Action prediction horizon | 16 |
| Actions per chunk | 4 |
| Diffusion training timesteps | 100 |
| Denoising steps set during training | 16 |
| Noise scheduler | DDIM |
| Image crop | Random **224 × 304** from 240 × 320 |
| Checkpoint save interval | 3,000 steps |

**Camera augmentation is random cropping only** in the included script, not brightness or camera-angle augmentation. Camera images were standardized during conversion; the random crop is applied during training.

For **trajectory**, the provided training script does **not** automatically switch use cases. Before training trajectory policies, make a copy of it and replace the `repo_id` and `out_dir` patterns with `ur3e_trajectory_...`. Ensure that the trajectory dataset uses an appropriate goal-detection/conversion rule and a compatible LeRobot version.

## 7. Execute a trained policy on the UR3e

**Stop the teleoperation controller and Touch publisher before inference.** Only the UR driver, velocity controller, sensor broadcasters, camera, and (if used) Robotiq bridge should remain active. Keep the robot workspace clear and verify safety settings. Start with conservative speeds and no obstacle/contact.

Locate a valid LeRobot checkpoint: it must contain a `config.json` in the model directory, or an enclosing checkpoint directory with `pretrained_model/config.json`.

```bash
# In a fresh terminal, after starting UR driver + camera and sourcing env:
source /opt/ros/humble/setup.bash
source ~/ros2_ws/.venv/bin/activate
source ~/ros2_ws/install/setup.bash

# Substitute the actual trained checkpoint directory:
ros2 run ur3e_imitation_learning run_inference \
  --checkpoint "$HOME/ros2_ws/outputs/injection_haptic_with_force_aug/checkpoints/last" \
  --use-forces \
  --results-dir "$HOME/ros2_ws/evaluation_results/needle_haptic_with_force" \
  --max-steps 240 --kp 12 --max-fz 22
```

The checkpoint path above is **an example**. Inspect the output folders generated by your installed LeRobot version; do not assume a directory named `last` always exists. Omit `--use-forces` for a no-force policy, otherwise its expected observation dimension will not match.

Key controls in the **inference** terminal:

| Key | Effect |
|---|---|
| `Space` | Start a run from the paused/home state, or pause an active run. |
| `c` / `o` | Close/open the Robotiq gripper (when connected). |
| `s` | Mark run as SUCCESS, save CSV and diagnostics, return home. |
| `f` | Mark run as FAILED, save CSV and diagnostics, return home. |
| `r` or `h` | Return to home; clears current unsaved run. |
| `Ctrl+C` | Exit the node and send a stop command. |

At startup the robot automatically tries to reach a fixed joint home configuration. After homing, the node averages recent F/T samples to define a new zero-force offset. Do not press start while the robot is contacting the environment. The execution loop:

1. Reads the latest RGB camera image and six measured joint angles, plus filtered and tared forces when enabled.
2. Gets the next `delta_q` from the Diffusion Policy's action queue. A new denoising pass occurs when the queue empties.
3. Clips joint increments to **0.04 rad/action** and limits the desired-position lead to **0.025 rad**.
4. Tracks the desired joint position with a P controller (default `Kp=12`), outputting joint velocities bounded to **0.45 rad/s**.
5. Allows up to **6 controller ticks per policy action**, depending on tracking error. Therefore `--max-steps 240` is **NOT guaranteed to finish in 8 seconds** despite the old help string.
6. Detects contact when `|Fz| >= 1.5 N` and suppresses tiny post-contact actions below the `0.0022 rad` norm threshold.
7. Aborts with `OVERFORCE_FAIL` when the tared **absolute axial force** exceeds the configured `--max-fz` limit (default 22 N). This is a software limit on **Fz**, not a general collision monitor.
8. When finished, waits for an operator success/failure label, saves per-run data and graphs, and returns home.

Results are stored in the selected `--results-dir`, including `summary_results.csv`, `run_XXXX_data.csv` and `run_XXXX_dashboard.png`. These output files are not intended for Git commits.

## 8. Experimental conditions and evaluation

For **each task**, build and test four independent policies:

| Condition | Haptic feedback while recording? | Force in policy observation? |
|---|---|---|
| C1 | Yes | No |
| C2 | No | No |
| C3 | Yes | Yes |
| C4 | No | Yes |

The original project author described **50 demonstrations per haptic condition per task** and **five autonomous trials per policy per task**. These are reported experiment plans/counts, not counts automatically verified by the source code. The CSV files and saved episodes are needed to confirm actual totals.

Useful evaluation outputs are trial success/failure, peak `|Fz|`, peak total force `||F||`, mean/standard deviation of contact force, final TCP position and execution time. The force sensor is at the **robot wrist**, not at the needle tip or inside the gripper jaws. Force comparisons should use consistent tare conditions and robot/tool orientations.

## 9. Troubleshooting checklist

| Symptom | Check |
|---|---|
| `ros2 pkg prefix ur3e_imitation_learning` fails | Build and source the workspace overlay; inspect `colcon list`. |
| `cannot find -lHD` / `-lHDU` | OpenHaptics SDK/64-bit libraries and compiler/linker paths. |
| No robot movement | External Control URCap running, controller active, robot in Remote Control (if required), no competing publishers, safety state OK. |
| No gripper command | Correct IP, TCP port 63352, Robotiq daemon, `/gripper/cmd` / `/gripper/target`. |
| No camera images | Check `/dev/video*`, camera node namespace, `ros2 topic hz /camera/image_raw`. |
| Recorder refuses to start | Return to the configured home joints; check image, joint and wrench topics. |
| Image/joint misalignment | Recorder uses latest-value 30 Hz sampling, **not precise timestamp synchronization**; measure timestamp skew before fast contact tasks. |
| Conversion says no episodes | Check exact directory `~/ros2_ws/raw_demonstrations/<use-case>/<demo-mode>`. |
| `ModuleNotFoundError: lerobot` | Activate the same compatible LeRobot environment in the terminal. |
| LeRobot training argument error | LeRobot checkout does not match this legacy CLI; find/install a compatible commit. |
| Checkpoint cannot load | Verify directory and matching policy state dimensions, features and LeRobot code version. |
| Policy too slow / high contact force | Review GPU inference latency, six-tick action tracking, limits and force-tare conditions; validate without contact first. |

## 10. Repository hygiene and known limitations

- Build/install/log folders, datasets, recorded images, model weights, evaluation output, virtual environments and `lerobot_source/` are excluded by `.gitignore`.
- Check that **no credentials, SSH keys, proprietary SDK redistributables, private images or sensitive recordings** are staged before publishing.
- `run_inference.py` uses LeRobot's private `_queues["action"]` implementation detail. A LeRobot upgrade may break it.
- No exact LeRobot commit, full Python dependency lockfile, automatic dataset quality filter or hardware test suite is provided.
- The recorder does not synchronize sensor timestamps; it records latest available values at 30 Hz.
- Training includes synthetic terminal frames that may not represent physically consistent trajectories. The conversion script's needle-specific completion detection currently applies to other use cases as well.
- The task-specific launch scripts presume fixed hardware paths, installed system tools and a fixed `~/ros2_ws` location; audit them before running.
- The default home joint angles and TCP offset (0.18 m) are **configuration-specific**, not universal UR3e calibration values.
- The `.py` files' explanatory comments are documentation; code behavior was not altered by the commenting pass.

## 11. Relevant source files

- `src/ur3e_imitation_learning/ur3e_imitation_learning/record_episodes.py`: teleoperation dataset recording.
- `src/ur3e_imitation_learning/ur3e_imitation_learning/convert_to_lerobot.py`: data conversion and synthetic stop frames.
- `src/ur3e_imitation_learning/scripts/train_injection_augmented.sh`: original Diffusion Policy training configuration.
- `src/ur3e_imitation_learning/ur3e_imitation_learning/run_inference.py`: autonomous robot control and evaluation.
- `src/UR3e_teleop/src/ur3e_teleop_control/ur3e_teleop_control/touch_publisher.py`: Touch mapping and force-feedback scaling.
- `src/UR3e_teleop/src/ur3e_teleop_control/ur3e_teleop_control/task_space_controller.py`: cartesian closed-loop velocity-command control with damped inverse kinematics.
- `src/UR3e_teleop/src/touch_haptics/src/touch_haptic_bridge.cpp`: OpenHaptics hardware bridge.
- `src/UR3e_teleop/src/ur3e_teleop_control/ur3e_teleop_control/robotiq_bridge.py`: Robotiq TCP bridge.

---

*Project source documentation, based on the supplied code snapshot. Hardware control, dependency installation and full training have not been tested in a clean environment as part of this documentation pass.*
