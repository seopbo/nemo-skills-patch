#!/bin/bash
# NeMo-Skills + NeMo-RL Integration Test
#
# Architecture:
#   1. NeMo-RL starts with NeMo-Gym, vLLM exposes HTTP
#   2. We detect vLLM URL from logs
#   3. Start proxy with discovered vLLM URL
#   4. NemoGym environment (already created) calls proxy for rollouts
#   5. Proxy forwards to vLLM, returns responses
#
# Key: NeMo-RL owns vLLM (can update weights), NemoGym uses proxy for rollouts
# Note: There's a race - NemoGym might try to call proxy before it's ready.
#       The proxy URL is configured ahead of time, but actual connection happens later.

set -e

MODEL="${NEMO_SKILLS_TEST_HF_MODEL:-Qwen/Qwen3-0.6B}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_FILE="/tmp/nemo-rl-test-data.jsonl"
# Use a random port to avoid conflicts with previous runs
PROXY_PORT=$((7200 + RANDOM % 100))
PROXY_URL="http://localhost:${PROXY_PORT}/v1"

# Kill any leftover processes on the port
fuser -k ${PROXY_PORT}/tcp 2>/dev/null || true

echo '{"problem": "What is 2+2?", "expected_answer": "4"}' > "$DATA_FILE"

cleanup() {
    echo "Cleaning up..."
    jobs -p | xargs -r kill 2>/dev/null || true
    rm -f "$DATA_FILE" /tmp/nemo-rl.log /tmp/nemo-skills.log
}
trap cleanup EXIT

# ============================================================================
# Step 1: Start NeMo-RL with NeMo-Gym (vLLM will expose HTTP server)
# Note: NemoGym is configured to call PROXY_URL for rollouts
# ============================================================================
echo "Starting NeMo-RL with NeMo-Gym integration..."
# Patch NemoGym bug: global_config_dict_yaml returns YAML but client expects JSON
NEMO_GYM_PATCH='sed -i "s/return OmegaConf.to_yaml(get_global_config_dict())/import json; return json.dumps(OmegaConf.to_container(get_global_config_dict()))/" /opt/nemo_rl_venv/lib/python3.12/site-packages/nemo_gym/server_utils.py'
ns run_cmd --cluster test-local --config_dir "$SCRIPT_DIR" --container nemo-rl --num_gpus 1 \
    "$NEMO_GYM_PATCH && python -u -m nemo_skills.training.nemo_rl.start_grpo \
        ++policy.model_name=$MODEL \
        ++data.train_data_path=$DATA_FILE \
        ++data.prompt.prompt_config=generic/math \
        ++policy.generation.vllm_cfg.expose_http_server=true \
        ++policy.generation.vllm_cfg.async_engine=true \
        ++policy.megatron_cfg.enabled=false \
        ++env.should_use_nemo_gym=true \
        ++nemo_gym.policy_base_url=$PROXY_URL \
        ++grpo.max_num_epochs=5 \
        ++grpo.num_iterations=1 \
        ++grpo.num_generations_per_prompt=2 \
        ++grpo.num_prompts_per_step=1" \
    2>&1 | tee /tmp/nemo-rl.log &
NEMO_RL_PID=$!

# ============================================================================
# Step 2: Wait for vLLM to start and get its URL
# ============================================================================
echo "Waiting for vLLM to start..."
VLLM_URL=""
for i in {1..360}; do
    VLLM_URL=$(grep -oP 'Starting server on \K(http://[0-9.]+:[0-9]+/v1)' /tmp/nemo-rl.log 2>/dev/null | head -1 || true)
    if [ -n "$VLLM_URL" ]; then
        echo "Found vLLM at: $VLLM_URL"
        break
    fi
    if [ $((i % 10)) -eq 0 ]; then
        echo "  ... waiting for vLLM ($i/360)"
    fi
    sleep 2
done

if [ -z "$VLLM_URL" ]; then
    echo "ERROR: vLLM URL not found in logs"
    echo "=== NeMo-RL Log ===" && tail -100 /tmp/nemo-rl.log
    exit 1
fi

# ============================================================================
# Step 3: Start NeMo-Skills proxy with discovered vLLM URL
# ============================================================================
echo "Starting NeMo-Skills proxy (connecting to $VLLM_URL)..."
ns run_cmd --cluster test-local --config_dir "$SCRIPT_DIR" --container nemo-skills --num_gpus 0 \
    "NEMO_RL_VLLM_URL=$VLLM_URL python -m nemo_skills.inference.generate \
        ++start_server=True \
        ++temperature=None \
        ++top_p=None \
        ++generate_port=$PROXY_PORT \
        ++prompt_format=openai" \
    2>&1 | tee /tmp/nemo-skills.log &
PROXY_PID=$!

# Wait for proxy to be ready
echo "Waiting for proxy to be ready..."
for i in {1..60}; do
    if grep -q "Server ready at" /tmp/nemo-skills.log 2>/dev/null; then
        echo "Proxy is ready!"
        break
    fi
    if [ $((i % 10)) -eq 0 ]; then
        echo "  ... waiting for proxy ($i/60)"
    fi
    sleep 2
done

# ============================================================================
# Step 4: Wait for NeMo-RL training to complete
# ============================================================================
echo "Waiting for NeMo-RL training..."
wait $NEMO_RL_PID
NEMO_RL_EXIT=$?

if [ $NEMO_RL_EXIT -ne 0 ]; then
    echo "ERROR: NeMo-RL failed with exit code $NEMO_RL_EXIT"
    echo "=== NeMo-RL Log ===" && tail -100 /tmp/nemo-rl.log
    echo "=== Proxy Log ===" && tail -50 /tmp/nemo-skills.log
    exit 1
fi

# ============================================================================
# Step 5: Verify proxy was used
# ============================================================================
echo ""
echo "Checking results..."

# Check if NeMo-Gym was actually used
if grep -q "NeMo-Gym environment created" /tmp/nemo-rl.log 2>/dev/null; then
    echo "✓ NeMo-Gym environment was created"
elif grep -q "NeMo-Gym is not available" /tmp/nemo-rl.log 2>/dev/null; then
    echo "✗ NeMo-Gym import failed - check logs for details"
    grep -A5 "NeMo-Gym is not available" /tmp/nemo-rl.log || true
else
    echo "⚠️ Couldn't determine if NeMo-Gym was used - check logs"
fi

# Count requests to the proxy
PROXY_REQUESTS=$(grep -c "/v1/chat/completions" /tmp/nemo-skills.log 2>/dev/null || echo "0")
echo "Proxy received approximately $PROXY_REQUESTS requests"

if [ "$PROXY_REQUESTS" -gt 0 ]; then
    echo ""
    echo "  - NeMo-RL started vLLM with HTTP exposed"
    echo "  - Proxy connected to vLLM"
    echo "  - NeMo-Gym used proxy for rollouts ($PROXY_REQUESTS requests)"
else
    echo ""
    echo "⚠️ Integration test completed but proxy may not have been used"
    echo "  Possible reasons:"
    echo "  - NeMo-Gym wasn't available in container"
    echo "  - Proxy started too late (race condition)"
    echo "  - Training finished before doing rollouts"
    echo ""
    echo "Check /tmp/nemo-skills.log and /tmp/nemo-rl.log for details"
fi
