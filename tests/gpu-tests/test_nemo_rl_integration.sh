#!/bin/bash
# NeMo-Skills + NeMo-RL Integration Test

set -e

MODEL="${NEMO_SKILLS_TEST_HF_MODEL:-Qwen/Qwen3-0.6B}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_FILE="/tmp/nemo-rl-test-data.jsonl"
PROXY_PORT=7000
VLLM_PORT=8000
VLLM_URL="http://localhost:${VLLM_PORT}/v1"
PROXY_URL="http://localhost:${PROXY_PORT}/v1"

echo '{"question": "What is 2+2?", "expected_answer": "4"}' > "$DATA_FILE"

cleanup() {
    echo "Cleaning up..."
    jobs -p | xargs -r kill 2>/dev/null || true
    rm -f "$DATA_FILE" /tmp/nemo-rl.log /tmp/nemo-skills.log
}
trap cleanup EXIT

# Start NeMo-Skills proxy (handles vLLM discovery internally)
echo "Starting NeMo-Skills proxy..."
ns run_cmd --cluster test-local --config_dir "$SCRIPT_DIR" --container nemo-skills --num_gpus 0 \
    "NEMO_RL_VLLM_URL=$VLLM_URL python -m nemo_skills.inference.generate \
        ++start_server=True \
        ++generate_port=$PROXY_PORT \
        ++server_wait_timeout=300 \
        ++prompt_format=openai" \
    2>&1 | tee /tmp/nemo-skills.log &

sleep 3

# Start NeMo-RL
echo "Starting NeMo-RL..."
ns run_cmd --cluster test-local --config_dir "$SCRIPT_DIR" --container nemo-rl --num_gpus 1 \
    "python -u -m nemo_skills.training.nemo_rl.start_grpo \
        ++policy.model_name=$MODEL \
        ++data.train_data_path=$DATA_FILE \
        ++policy.generation.vllm_cfg.expose_http_server=true \
        ++policy.generation.vllm_cfg.http_server_port=$VLLM_PORT \
        ++policy.generation.vllm_cfg.async_engine=true \
        ++policy.megatron_cfg.enabled=false \
        ++nemo_gym.policy_backend=openai \
        ++nemo_gym.openai_cfg.base_url=$PROXY_URL \
        ++nemo_gym.openai_cfg.api_key=EMPTY \
        ++grpo.num_iterations=3" \
    2>&1 | tee /tmp/nemo-rl.log

echo "✅ Done"
