#!/bin/bash
# Local test script for GRPO-Gym training
# This script runs a quick sanity test with a tiny model

set -e

echo "=================================="
echo "GRPO-Gym Local Test"
echo "=================================="

# Configuration
MODEL_PATH="${NEMO_SKILLS_TEST_HF_MODEL:-/home/wedu/Qwen3-0.6B}"
MODEL_TYPE="${NEMO_SKILLS_TEST_MODEL_TYPE:-qwen}"
OUTPUT_DIR="/tmp/nemo-skills-tests/${MODEL_TYPE}/test-grpo-gym-local/fsdp"
BACKEND="fsdp"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Backend: $BACKEND"
echo "  Output: $OUTPUT_DIR"
echo ""

# Clean up previous runs
echo "🧹 Cleaning up previous runs..."
rm -rf /tmp/ray/ray_current_cluster
rm -rf /mnt/datadrive/nemo-skills-test-data/hf-cache/nemo_rl/ 2>/dev/null || true
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo ""
echo "📂 Directories:"
echo "  Script: $SCRIPT_DIR"
echo "  Project: $PROJECT_DIR"
echo ""

# Run GRPO-Gym training
echo "🚀 Starting GRPO-Gym training..."
echo ""

python -c "
from pathlib import Path
from nemo_skills.pipeline.cli import grpo_gym_nemo_rl, wrap_arguments

grpo_gym_nemo_rl(
    ctx=wrap_arguments(
        '++grpo.max_num_steps=3 '
        '++grpo.num_prompts_per_step=2 '
        '++policy.max_total_sequence_length=256 '
        '++policy.dtensor_cfg.tensor_parallel_size=1 '
        '++checkpointing.save_period=2 '
        '++policy.train_global_batch_size=2 '
        '++policy.train_micro_batch_size=1 '
        '++policy.optimizer.kwargs.lr=1e-6 '
    ),
    cluster='test-local',
    config_dir=Path('${SCRIPT_DIR}').absolute(),
    expname='test-grpo-gym-local',
    output_dir='${OUTPUT_DIR}',
    hf_model='${MODEL_PATH}',
    num_nodes=1,
    num_gpus=1,
    training_data='/nemo_run/code/tests/data/small-grpo-gym-data-train.test',
    validation_data='/nemo_run/code/tests/data/small-grpo-gym-data-val.test',
    backend='${BACKEND}',
    disable_wandb=True,
)
"

echo ""
echo "=================================="
echo "✅ Test completed!"
echo "=================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To check the output:"
echo "  ls -lh $OUTPUT_DIR"
echo "  cat $OUTPUT_DIR/training-logs/*.out"
