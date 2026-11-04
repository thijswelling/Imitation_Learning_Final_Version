#!/usr/bin/env bash
# TRAININGSCRIPT MET CAMERA AUGMENTATION VOOR INJECTION TASK

set -e

DEMO_MODE=${1:-haptic}
FORCE_MODE=${2:-with_force}
LEROBOT_DIR="$HOME/ros2_ws/lerobot_source"

repo_id="local/ur3e_needle_${DEMO_MODE}_${FORCE_MODE}"
out_dir="/home/thijs/outputs/injection_${DEMO_MODE}_${FORCE_MODE}_aug"

echo "===================================================================="
echo "🚀 Start Training (MET Camera Augmentation): ${repo_id}"
echo "📂 Output map: ${out_dir}"
echo "===================================================================="

rm -rf "${out_dir}"
cd "${LEROBOT_DIR}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 src/lerobot/scripts/lerobot_train.py \
  --policy.type=diffusion \
  --policy.device="cuda" \
  --dataset.repo_id="${repo_id}" \
  --output_dir="${out_dir}" \
  --batch_size=8 \
  --steps=12000 \
  --save_freq=3000 \
  --policy.crop_shape="[224, 304]" \
  --policy.crop_is_random=true \
  --policy.noise_scheduler_type="DDIM" \
  --policy.num_train_timesteps=100 \
  --policy.num_inference_steps=16 \
  --policy.horizon=16 \
  --policy.n_action_steps=4 \
  --policy.n_obs_steps=2 \
  --policy.down_dims="[64, 128, 256]" \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --dataset.streaming=false

echo "✅ Training met augmentatie voltooid voor: ${repo_id}"