#!/bin/bash
# NeMo-Skills + NeMo-RL Integration Test

set -e

MODEL="${NEMO_SKILLS_TEST_HF_MODEL:-Qwen/Qwen3-0.6B}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_FILE="/tmp/nemo-rl-test-data.jsonl"
PROXY_PORT=7000

echo '{"question": "What is 2+2?", "expected_answer": "4"}' > "$DATA_FILE"

cleanup() {
    echo "Cleaning up..."
    kill %1 %2 2>/dev/null || true
    rm -f "$DATA_FILE"
}
trap cleanup EXIT

# Start NeMo-RL (vLLM with HTTP exposed) in background
echo "Starting NeMo-RL..."
ns run_cmd --cluster test-local --config_dir "$SCRIPT_DIR" --container nemo-rl --num_gpus 1 \
    "python -m nemo_skills.training.nemo_rl.start_grpo \
        ++policy.model_name=$MODEL \
        ++data.train_data_path=$DATA_FILE \
        ++policy.generation.vllm_cfg.expose_http_server=true \
        ++policy.generation.vllm_cfg.async_engine=true \
        ++policy.megatron_cfg.enabled=false \
        ++grpo.num_iterations=1" \
    2>&1 | tee /tmp/nemo-rl.log &

# Wait for vLLM URL to appear in logs
echo "Waiting for vLLM URL..."
VLLM_URL=""
for i in {1..120}; do
    VLLM_URL=$(grep -oP 'http://[0-9.]+:[0-9]+/v1' /tmp/nemo-rl.log 2>/dev/null | head -1 || true)
    [ -n "$VLLM_URL" ] && break
    sleep 2
done
[ -z "$VLLM_URL" ] && { echo "ERROR: vLLM URL not found"; exit 1; }
echo "vLLM at: $VLLM_URL"

# Start NeMo-Skills proxy (will auto-wait for vLLM via server_wait_timeout)
echo "Starting NeMo-Skills proxy..."
ns run_cmd --cluster test-local --config_dir "$SCRIPT_DIR" --container nemo-skills \
    "NEMO_RL_VLLM_URL=$VLLM_URL python -m nemo_skills.inference.generate \
        ++start_server=True \
        ++generate_port=$PROXY_PORT \
        ++server_wait_timeout=120 \
        ++prompt_config=generic/math" \
    2>&1 | tee /tmp/nemo-skills.log &

# Wait for proxy
echo "Waiting for proxy..."
for i in {1..60}; do
    curl -s "http://localhost:$PROXY_PORT/health" && break
    sleep 2
done

# Test
echo "Testing..."
RESPONSE=$(curl -s -X POST "http://localhost:$PROXY_PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"nemo-skills","messages":[{"role":"user","content":"What is 2+2?"}]}')

echo "Response: $RESPONSE"
echo "$RESPONSE" | grep -q "choices" && echo "✅ PASS" || { echo "❌ FAIL"; exit 1; }
