#!/bin/bash
# Local test script for GRPO-Gym training with LLM-as-Judge
# 
# Requires 2 GPUs:
#   - GPU 0: External Judge vLLM Server (Docker container)
#   - GPU 1: Policy vLLM + GRPO Training (nemo-rl container)

set -e

echo "=================================="
echo "GRPO-Gym Local Test with LLM-as-Judge"
echo "=================================="

# Get script directory first
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
MODEL_PATH="${NEMO_SKILLS_TEST_HF_MODEL:-Qwen/Qwen3-0.6B}"
MODEL_TYPE="${NEMO_SKILLS_TEST_MODEL_TYPE:-qwen}"
OUTPUT_DIR="${HOME}/nemo-skills-tests/${MODEL_TYPE}/test-grpo-gym-judge-local/fsdp"
BACKEND="fsdp"

# Judge vLLM configuration
JUDGE_PORT=8100
JUDGE_GPU=0
JUDGE_CONTAINER="judge-vllm-test"
VLLM_IMAGE="vllm/vllm-openai:v0.10.2"

# Get host IP for container-to-host communication
HOST_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "Configuration:"
echo "  Policy Model: $MODEL_PATH (GPU 1)"
echo "  Judge Model: $MODEL_PATH (GPU $JUDGE_GPU, port $JUDGE_PORT)"
echo "  Host IP: $HOST_IP"
echo "  Output: $OUTPUT_DIR"
echo ""

# Preflight: this script expects you to run from an environment that has nemo_run
# (used by the NeMo-Skills pipeline launcher). Fail early before starting containers.
python -c "import nemo_run" >/dev/null 2>&1 || {
    echo "  ❌ ERROR: Python module 'nemo_run' not found."
    echo "     Please activate the NeMo-Skills environment (the one you used for the non-judge test) and retry."
    exit 1
}

# Clean up (skip output_dir - user can delete manually if needed)
echo "🧹 Cleaning up previous runs..."
rm -rf /tmp/ray/ray_current_cluster 2>/dev/null || true
# IMPORTANT: NeMo-RL/Ray isolated worker venvs can be cached here; stale caches can
# lead to dependency drift (e.g., numpy major mismatch) and cryptic import errors.
rm -rf /mnt/datadrive/nemo-skills-test-data/hf-cache/nemo_rl/ 2>/dev/null || true
mkdir -p "$OUTPUT_DIR"

# Stop any existing judge container
docker stop $JUDGE_CONTAINER 2>/dev/null || true
docker rm $JUDGE_CONTAINER 2>/dev/null || true

# ============================================================================
# Step 1: Start Judge vLLM Server (GPU 0)
# ============================================================================
echo ""
echo "🤖 Step 1: Starting Judge vLLM server on GPU $JUDGE_GPU..."

docker run -d \
    --name $JUDGE_CONTAINER \
    --gpus "\"device=$JUDGE_GPU\"" \
    -v /home/binhu/code/hf_models:/root/.cache/huggingface \
    -p $JUDGE_PORT:$JUDGE_PORT \
    --shm-size=16g \
    $VLLM_IMAGE \
    --model "$MODEL_PATH" \
    --port $JUDGE_PORT \
    --trust-remote-code \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.5

echo "  Container: $JUDGE_CONTAINER"
echo "  Waiting for Judge vLLM..."

for i in $(seq 1 90); do
    if curl -s "http://localhost:$JUDGE_PORT/v1/models" > /dev/null 2>&1; then
        echo "  ✅ Judge vLLM ready at http://$HOST_IP:$JUDGE_PORT/v1"
        break
    fi
    if [ $((i % 15)) -eq 0 ]; then
        echo "  ... waiting ($i/90)"
    fi
    sleep 2
done

if ! curl -s "http://localhost:$JUDGE_PORT/v1/models" > /dev/null 2>&1; then
    echo "  ❌ ERROR: Judge vLLM failed to start"
    docker logs $JUDGE_CONTAINER 2>&1 | tail -30
    docker rm -f $JUDGE_CONTAINER 2>/dev/null || true
    exit 1
fi

# ============================================================================
# Step 2: Run GRPO-Gym Training with Judge (GPU 1)
# ============================================================================
echo ""
echo "🚀 Step 2: Starting GRPO-Gym training on GPU 1..."

# Judge vLLM URL (accessible from nemo-rl container via host IP)
JUDGE_URL="http://${HOST_IP}:${JUDGE_PORT}/v1"
echo "  Judge URL: $JUDGE_URL"

set +e
export MODEL_PATH SCRIPT_DIR OUTPUT_DIR BACKEND JUDGE_URL

# NOTE: nemo-run needs to be able to inspect source code of the entry module.
# Using `python -` (stdin) breaks that (OSError: could not get source code), so we
# write a temporary file and execute it.
TMP_PY="$(mktemp -t grpo_gym_with_judge.XXXXXX.py)"
cat > "$TMP_PY" <<'PY'
import os
from pathlib import Path

from nemo_skills.pipeline.cli import grpo_gym_nemo_rl, wrap_arguments

model_path = os.environ["MODEL_PATH"]
judge_url = os.environ["JUDGE_URL"]

ctx = (
    "++grpo.max_num_steps=3 "
    "++grpo.num_prompts_per_step=2 "
    "++grpo.num_generations_per_prompt=4 "
    "++policy.max_total_sequence_length=256 "
    "++policy.dtensor_cfg.tensor_parallel_size=1 "
    "++checkpointing.save_period=2 "
    "++policy.train_global_batch_size=8 "
    "++policy.train_micro_batch_size=1 "
    "++policy.optimizer.kwargs.lr=1e-6 "
    "++policy.generation.vllm_cfg.gpu_memory_utilization=0.3 "
    "++env.nemo_gym.config_paths=[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,resources_servers/math_with_judge/configs/math_with_judge.yaml,responses_api_agents/simple_agent/configs/simple_agent.yaml] "
    "++env.nemo_gym.judge_model.responses_api_models.openai_model.entrypoint=app.py "
    f"++env.nemo_gym.judge_model.responses_api_models.openai_model.openai_base_url={judge_url} "
    "++env.nemo_gym.judge_model.responses_api_models.openai_model.openai_api_key=EMPTY "
    f"++env.nemo_gym.judge_model.responses_api_models.openai_model.openai_model={model_path} "
    "++env.nemo_gym.math_with_judge.resources_servers.math_with_judge.should_use_judge=true "
    "++env.nemo_gym.math_with_judge.resources_servers.math_with_judge.judge_model_server.type=responses_api_models "
    "++env.nemo_gym.math_with_judge.resources_servers.math_with_judge.judge_model_server.name=judge_model "
    "++env.nemo_gym.math_with_judge.resources_servers.math_with_judge.judge_threshold=1.1 "
)

grpo_gym_nemo_rl(
    ctx=wrap_arguments(ctx),
    cluster="test-local-gpu1",
    config_dir=Path(os.environ["SCRIPT_DIR"]).absolute(),
    expname="test-grpo-gym-judge-local",
    output_dir=os.environ["OUTPUT_DIR"],
    hf_model=model_path,
    num_nodes=1,
    num_gpus=1,
    training_data="/nemo_run/code/tests/data/small-grpo-gym-data-train.test",
    validation_data="/nemo_run/code/tests/data/small-grpo-gym-data-val.test",
    backend=os.environ["BACKEND"],
    disable_wandb=True,
)
PY
python "$TMP_PY"
PY_EXIT_CODE=$?
rm -f "$TMP_PY"
exit $PY_EXIT_CODE
TRAINING_EXIT_CODE=$?

# If nemo-run prints "finished: FAILED" but the wrapper returned 0, force a failure here
# so this script reliably fails in CI/local debugging.
if grep -R "finished: FAILED" "$OUTPUT_DIR/training-logs" 2>/dev/null | head -n 1 >/dev/null; then
    echo "  ❌ Detected nemo-run failure in training logs."
    TRAINING_EXIT_CODE=1
fi
set -e

# ============================================================================
# Cleanup
# ============================================================================
echo ""
echo "🧹 Stopping Judge vLLM container..."
docker stop $JUDGE_CONTAINER 2>/dev/null || true
docker rm $JUDGE_CONTAINER 2>/dev/null || true

echo ""
echo "=================================="
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Test completed!"
else
    echo "❌ Test failed (exit code: $TRAINING_EXIT_CODE)"
fi
echo "=================================="
echo ""
echo "Logs:"
echo "  Training: cat $OUTPUT_DIR/training-logs/*.out"

exit $TRAINING_EXIT_CODE
