#!/bin/bash

set -e

CONFIG_FILE="/home/ubuntu/workspace/ruogu/maskpolicy_train/train_config/mask_act/try_mask_act.json"
TRAIN_SCRIPT="/home/ubuntu/workspace/ruogu/Mask_DP/train.py"
OUTPUT_BASE_DIR="/home/ubuntu/workspace/ruogu/maskpolicy_train/outputs/mask_act/mask_act_1214_usemask"
STAGE1_OUTPUT_DIR="${OUTPUT_BASE_DIR}/mask_act_224_stage1"
STAGE2_OUTPUT_DIR="${OUTPUT_BASE_DIR}/mask_act_224_stage2"
STAGE1_STEPS=3000
STAGE2_STEPS=300000
STAGE1_SAVE_STEPS=1000
STAGE2_SAVE_STEPS=50000
STAGE1_BATCH_SIZE=64
STAGE2_BATCH_SIZE=64
use_mask=True

source $(conda info --base)/etc/profile.d/conda.sh
conda activate maskpolicy

echo "Starting Stage 1: Mask Prediction Training"
STAGE1_CONFIG="/tmp/mask_dp_stage1_config_$$.json"
cp "$CONFIG_FILE" "$STAGE1_CONFIG"
python3 << EOF
import json
with open("$STAGE1_CONFIG", "r") as f:
    config = json.load(f)
config["policy"]["training_stage"] = "stage1"
config["output_dir"] = "$STAGE1_OUTPUT_DIR"
config["job_name"] = "mask_act_224_stage1"
config["steps"] = $STAGE1_STEPS
config["save_freq"] = $STAGE1_SAVE_STEPS
with open("$STAGE1_CONFIG", "w") as f:
    json.dump(config, f, indent=2)
EOF

if [ "$use_mask" = True ]; then
python "$TRAIN_SCRIPT" --config "$STAGE1_CONFIG" || {
    echo "Stage 1 training failed"
    rm -f "$STAGE1_CONFIG"
    exit 1
}
fi

echo "Stage 1 training completed"

echo "Starting Stage 2: Joint Training (Action + Mask)"
STAGE2_CONFIG="/tmp/mask_dp_stage2_config_$$.json"
cp "$CONFIG_FILE" "$STAGE2_CONFIG"
python3 << EOF
import json
with open("$STAGE2_CONFIG", "r") as f:
    config = json.load(f)
config["policy"]["training_stage"] = "stage2"
config["output_dir"] = "$STAGE2_OUTPUT_DIR"
config["job_name"] = "mask_act_224_stage2"
config["steps"] = $STAGE2_STEPS
config["save_freq"] = $STAGE2_SAVE_STEPS
config["batch_size"] = $STAGE2_BATCH_SIZE
config["policy"]["use_mask"] = $use_mask
with open("$STAGE2_CONFIG", "w") as f:
    json.dump(config, f, indent=2)
EOF

STAGE1_PRETRAINED_PATH=""
if [ -L "$STAGE1_OUTPUT_DIR/checkpoints/last" ]; then
    checkpoint_dir=$(readlink -f "$STAGE1_OUTPUT_DIR/checkpoints/last")
    [ -d "$checkpoint_dir/pretrained_model" ] && STAGE1_PRETRAINED_PATH="$checkpoint_dir/pretrained_model"
elif [ -d "$STAGE1_OUTPUT_DIR/checkpoints" ]; then
    latest_checkpoint=$(ls -td "$STAGE1_OUTPUT_DIR/checkpoints"/*/ 2>/dev/null | head -1)
    [ -n "$latest_checkpoint" ] && [ -d "$latest_checkpoint/pretrained_model" ] && STAGE1_PRETRAINED_PATH="$latest_checkpoint/pretrained_model"
fi

if [ -n "$STAGE1_PRETRAINED_PATH" ]; then
    echo "Using pretrained model: $STAGE1_PRETRAINED_PATH"
    PRETRAINED_ARG="--policy.path=$STAGE1_PRETRAINED_PATH"
else
    echo "Stage 1 pretrained model not found, starting stage 2 from scratch"
    PRETRAINED_ARG=""
fi

python "$TRAIN_SCRIPT" --config "$STAGE2_CONFIG" $PRETRAINED_ARG || {
    echo "Stage 2 training failed"
    rm -f "$STAGE1_CONFIG" "$STAGE2_CONFIG"
    exit 1
}

echo "Stage 2 training completed"

rm -f "$STAGE1_CONFIG" "$STAGE2_CONFIG"

echo "All training stages completed"
echo "Stage 1 output: $STAGE1_OUTPUT_DIR"
echo "Stage 2 output: $STAGE2_OUTPUT_DIR"
