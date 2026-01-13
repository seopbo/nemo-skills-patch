#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeMo-RL + NeMo-Skills Integration Test with LLM-as-Judge (Single Node)

Simplified single-node version for testing:
  - Judge vLLM: GPU 0
  - Judge Proxy: CPU
  - Policy vLLM (NeMo-RL managed): GPU 1+
  - Policy Proxy: CPU
  - Training: GPU 1+

All components run on the same node using localhost.

Uses the new nemo-rl-judge image with all components pre-installed.
"""

from nemo_skills.pipeline.utils.declarative import Command, CommandGroup, HardwareConfig, Pipeline
from nemo_skills.pipeline.utils import get_cluster_config

# ============================================================================
# Configuration
# ============================================================================
cluster = 'eos'
partition = 'batch'

# Model paths - 使用小模型快速验证
policy_model = '/lustre/fsw/llmservice_nemo_reasoning/wedu/models/Qwen2.5-1.5b-genrm-cot-tir'
judge_model = '/lustre/fsw/llmservice_nemo_reasoning/wedu/models/Qwen2.5-1.5b-genrm-cot-tir'
test_data = '/lustre/fsw/llmservice_nemo_reasoning/bihu/test-data/nemo-rl-integration-test-with-question.jsonl'
output_dir = '/lustre/fsw/llmservice_nemo_reasoning/bihu/test-nemo-rl-integration-judge'

# 单节点配置
num_nodes = 1
num_gpus = 8  # 总共 8 GPU

# GPU 分配
judge_gpu = 0           # Judge vLLM 使用 GPU 0
policy_gpus = "1,2,3,4"  # Policy/Training 使用 GPU 1-4 (batch_size 8 能被 4 整除)
policy_num_gpus = 4     # 用于 NeMo-RL

# 端口策略（避免硬编码 + 避免抢占/冲突）：
# - 所有服务端口从同一个 SLURM_JOB_ID 派生，保证同一 job 内四个 component 计算一致
# - 基础端口范围 20000-39999（与你之前 cluster 脚本保持一致）
# - 避免占用 Ray worker 端口段（你在训练里设置的是 30000-40000），因此尽量把服务端口落在 20000-29999
#
# 端口分配（BASE = 20000 + (SLURM_JOB_ID % 10000)  -> 20000-29999）：
# - judge_vllm:               BASE + 80
# - judge_proxy (skills):     BASE + 81
# - policy_proxy (skills):    BASE + 180
# - judge_model (Gym wrapper):BASE + 82
# - policy_model (Gym wrapper):BASE + 83
# - gym_agent:                BASE + 90

# 新镜像：包含新版 NeMo-RL + NeMo-Gym（支持 LLM-as-Judge）
CONTAINER = "nemo-rl-judge"

print(f"[config] cluster={cluster}")
print(f"[config] container={CONTAINER}")
print(f"[config] policy_model={policy_model}")
print(f"[config] judge_model={judge_model}")
print(f"[config] Single node: {num_gpus} GPUs (judge=GPU{judge_gpu}, policy=GPU{policy_gpus})")
print("[config] Ports are derived from SLURM_JOB_ID at runtime (see script comments).")

# ============================================================================
# Load cluster configuration
# ============================================================================
cluster_config = get_cluster_config(cluster, None)
nemo_rl_log = f"{output_dir}/nemo-rl-$SLURM_JOB_ID.log"
judge_model_name = judge_model.split('/')[-1]

# ============================================================================
# Step 1: Judge vLLM Server (GPU 0)
# ============================================================================
print("\n🤖 Step 1: Setting up Judge vLLM server (GPU 0)...")

judge_vllm_cmd = f"""bash -lc 'set -euo pipefail
export CUDA_VISIBLE_DEVICES={judge_gpu}
BASE_PORT=$((20000 + (SLURM_JOB_ID % 10000)))
JUDGE_VLLM_PORT=$((BASE_PORT + 80))
echo "[judge_vllm] Starting on GPU {judge_gpu}, port $JUDGE_VLLM_PORT (BASE_PORT=$BASE_PORT)"

python3 -m nemo_skills.inference.server.serve_vllm \\
    --model {judge_model} \\
    --num_gpus 1 \\
    --port $JUDGE_VLLM_PORT
'"""

judge_vllm_server = Command(
    command=judge_vllm_cmd,
    gpus=0,  # 在 Command 层面不分配，内部用 CUDA_VISIBLE_DEVICES
    name="judge_vllm",
    container="vllm",
)

# ============================================================================
# Step 2: Judge Proxy (CPU, connects to local judge vLLM)
# ============================================================================
print("🔌 Step 2: Setting up Judge Proxy...")

judge_proxy_cmd = f"""bash -lc 'set -euo pipefail
HOST=$(hostname -f)
BASE_PORT=$((20000 + (SLURM_JOB_ID % 10000)))
JUDGE_VLLM_PORT=$((BASE_PORT + 80))
JUDGE_PROXY_PORT=$((BASE_PORT + 81))
echo "[judge_proxy] Starting on port $JUDGE_PROXY_PORT, backend: http://$HOST:$JUDGE_VLLM_PORT/v1 (BASE_PORT=$BASE_PORT)"

# Wait for judge vLLM
until curl -s http://$HOST:$JUDGE_VLLM_PORT/v1/models > /dev/null 2>&1; do
    echo "[judge_proxy] Waiting for judge vLLM..."
    sleep 5
done
echo "[judge_proxy] Judge vLLM ready!"

python3 -m nemo_skills.inference.generate \\
    ++start_server=True \\
    ++prompt_format=openai \\
    ++inference.temperature=0.0 \\
    ++inference.top_p=-1 \\
    ++generate_port=$JUDGE_PROXY_PORT \\
    ++server.base_url=http://$HOST:$JUDGE_VLLM_PORT/v1 \\
    ++server.model={judge_model} \\
    ++server_wait_timeout=900
'"""

judge_proxy = Command(
    command=judge_proxy_cmd,
    gpus=0,
    name="judge_proxy",
    container=CONTAINER,
)

# ============================================================================
# Step 3: Policy Proxy (CPU, waits for NeMo-RL's vLLM from logs)
# ============================================================================
print("🧭 Step 3: Setting up Policy Proxy...")

policy_proxy_cmd = f"""bash -lc 'set -euo pipefail
BASE_PORT=$((20000 + (SLURM_JOB_ID % 10000)))
POLICY_PROXY_PORT=$((BASE_PORT + 180))
echo "[policy_proxy] Starting on port $POLICY_PROXY_PORT (BASE_PORT=$BASE_PORT)"
echo "[policy_proxy] Waiting for vLLM URL in {nemo_rl_log}..."

VLLM_URL=""
for i in $(seq 1 360); do
    if [ -f "{nemo_rl_log}" ]; then
        VLLM_URL=$(grep -oP "Starting server on \\K(http://[0-9.]+:[0-9]+/v1)" {nemo_rl_log} 2>/dev/null | head -1 || true)
        if [ -n "$VLLM_URL" ]; then
            echo "[policy_proxy] Found vLLM at: $VLLM_URL"
            break
        fi
    fi
    if [ $((i % 10)) -eq 0 ]; then
        echo "[policy_proxy] ... waiting ($i/360)"
    fi
    sleep 2
done

if [ -z "$VLLM_URL" ]; then
    echo "[policy_proxy] ERROR: vLLM URL not found after 12 minutes"
    tail -100 {nemo_rl_log} 2>/dev/null || echo "Log not found"
    exit 1
fi

python3 -m nemo_skills.inference.generate \\
    ++start_server=True \\
    ++prompt_format=openai \\
    ++inference.temperature=-1 \\
    ++inference.top_p=-1 \\
    ++generate_port=$POLICY_PROXY_PORT \\
    ++server_wait_timeout=900 \\
    ++server.base_url=$VLLM_URL \\
    ++server.model={policy_model} \\
    ++evaluator.type=math
'"""

policy_proxy = Command(
    command=policy_proxy_cmd,
    gpus=0,
    name="policy_proxy",
    container=CONTAINER,
)

# ============================================================================
# Step 4: Training (GPU 1-7)
# ============================================================================
print("🚀 Step 4: Setting up NeMo-RL training...")

training_cmd = f"""bash -lc 'set -euo pipefail
export CUDA_VISIBLE_DEVICES={policy_gpus}
HOST=$(hostname -f)

BASE_PORT=$((20000 + (SLURM_JOB_ID % 10000)))
JUDGE_VLLM_PORT=$((BASE_PORT + 80))
JUDGE_PROXY_PORT=$((BASE_PORT + 81))
JUDGE_MODEL_SERVER_PORT=$((BASE_PORT + 82))
POLICY_MODEL_SERVER_PORT=$((BASE_PORT + 83))
GYM_AGENT_PORT=$((BASE_PORT + 90))
POLICY_PROXY_PORT=$((BASE_PORT + 180))

JUDGE_PROXY_URL="http://$HOST:$JUDGE_PROXY_PORT"
POLICY_PROXY_URL="http://$HOST:$POLICY_PROXY_PORT/v1"

echo "[training] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[training] judge_proxy: $JUDGE_PROXY_URL"
echo "[training] policy_proxy: $POLICY_PROXY_URL"
echo "[training] BASE_PORT=$BASE_PORT ports: judge_vllm=$JUDGE_VLLM_PORT judge_proxy=$JUDGE_PROXY_PORT judge_model=$JUDGE_MODEL_SERVER_PORT policy_model=$POLICY_MODEL_SERVER_PORT gym_agent=$GYM_AGENT_PORT policy_proxy=$POLICY_PROXY_PORT"

# Wait for judge proxy
until curl -s $JUDGE_PROXY_URL/health > /dev/null 2>&1; do
    echo "[training] Waiting for judge proxy..."
    sleep 5
done
echo "[training] Judge proxy ready!"

unset RAY_ADDRESS 2>/dev/null || true

# Apply NeMo-Gym patches
echo "[training] Applying NeMo-Gym patches..."
# Patch 1: fix global_config_dict_yaml returning YAML instead of JSON
sed -i "s/return OmegaConf.to_yaml(get_global_config_dict())/import json; return json.dumps(OmegaConf.to_container(get_global_config_dict()))/" /opt/nemo_rl_venv/lib/python3.12/site-packages/nemo_gym/server_utils.py 2>/dev/null || true
# Patch 2: fix ALL requirements.txt files (replace invalid local nemo-gym install with proper package)
# Check both source dir and site-packages
for search_dir in /opt/NeMo-Gym /opt/nemo_rl_venv/lib; do
    echo "[training] Searching for requirements.txt in $search_dir..."
    for f in $(find "$search_dir" -name "requirements.txt" 2>/dev/null | grep -E "(resources_servers|responses_api)" || true); do
        if grep -q "^-e nemo-gym" "$f" 2>/dev/null; then
            echo "[training] Patching $f"
            # Replace invalid local path with proper nemo-gym install from /opt/NeMo-Gym
            sed -i "s|^-e nemo-gym.*|nemo-gym @ file:///opt/NeMo-Gym|" "$f"
        fi
    done
done
echo "[training] NeMo-Gym patches applied"

python -u -m nemo_skills.training.nemo_rl.start_grpo \\
    ++policy.model_name={policy_model} \\
    ++data.train_data_path={test_data} \\
    ++data.prompt.prompt_config=generic/math \\
    ++data.prompt.prompt_template=qwen-instruct \\
    ++policy.generation.vllm_cfg.expose_http_server=true \\
    ++policy.generation.vllm_cfg.async_engine=true \\
    ++policy.generation.vllm_cfg.skip_tokenizer_init=false \\
    ++policy.generation.vllm_cfg.gpu_memory_utilization=0.7 \\
    ++policy.generation.vllm_cfg.tensor_parallel_size=1 \\
    ++env.should_use_nemo_gym=true \\
    ++nemo_gym.initial_global_config_dict.config_paths=[] \\
    ++nemo_gym.policy_base_url=$POLICY_PROXY_URL \\
    ++nemo_gym.initial_global_config_dict.disallowed_ports=[$POLICY_PROXY_PORT,$JUDGE_PROXY_PORT,$JUDGE_VLLM_PORT,$JUDGE_MODEL_SERVER_PORT,$POLICY_MODEL_SERVER_PORT,$GYM_AGENT_PORT] \\
    ++nemo_gym.initial_global_config_dict.math_with_judge.resources_servers.math_with_judge.entrypoint=app.py \\
    ++nemo_gym.initial_global_config_dict.math_with_judge.resources_servers.math_with_judge.should_use_judge=true \\
    ++nemo_gym.initial_global_config_dict.math_with_judge.resources_servers.math_with_judge.domain=math \\
    ++nemo_gym.initial_global_config_dict.math_with_judge.resources_servers.math_with_judge.judge_model_server.type=responses_api_models \\
    ++nemo_gym.initial_global_config_dict.math_with_judge.resources_servers.math_with_judge.judge_model_server.name=judge_model \\
    ++nemo_gym.initial_global_config_dict.math_with_judge.resources_servers.math_with_judge.judge_responses_create_params.input=[] \\
    ++nemo_gym.initial_global_config_dict.math_with_judge_simple_agent.responses_api_agents.simple_agent.entrypoint=app.py \\
    ++nemo_gym.initial_global_config_dict.math_with_judge_simple_agent.responses_api_agents.simple_agent.host=$HOST \\
    ++nemo_gym.initial_global_config_dict.math_with_judge_simple_agent.responses_api_agents.simple_agent.port=$GYM_AGENT_PORT \\
    ++nemo_gym.initial_global_config_dict.math_with_judge_simple_agent.responses_api_agents.simple_agent.resources_server.type=resources_servers \\
    ++nemo_gym.initial_global_config_dict.math_with_judge_simple_agent.responses_api_agents.simple_agent.resources_server.name=math_with_judge \\
    ++nemo_gym.initial_global_config_dict.math_with_judge_simple_agent.responses_api_agents.simple_agent.model_server.type=responses_api_models \\
    ++nemo_gym.initial_global_config_dict.math_with_judge_simple_agent.responses_api_agents.simple_agent.model_server.name=policy_model \\
    ++nemo_gym.initial_global_config_dict.policy_model.responses_api_models.openai_model.entrypoint=app.py \\
    ++nemo_gym.initial_global_config_dict.policy_model.responses_api_models.openai_model.host=$HOST \\
    ++nemo_gym.initial_global_config_dict.policy_model.responses_api_models.openai_model.port=$POLICY_MODEL_SERVER_PORT \\
    ++nemo_gym.initial_global_config_dict.policy_model.responses_api_models.openai_model.openai_base_url=$POLICY_PROXY_URL \\
    ++nemo_gym.initial_global_config_dict.policy_model.responses_api_models.openai_model.openai_api_key=dummy \\
    ++nemo_gym.initial_global_config_dict.policy_model.responses_api_models.openai_model.openai_model={policy_model} \\
    ++nemo_gym.initial_global_config_dict.judge_model.responses_api_models.openai_model.entrypoint=app.py \\
    ++nemo_gym.initial_global_config_dict.judge_model.responses_api_models.openai_model.host=$HOST \\
    ++nemo_gym.initial_global_config_dict.judge_model.responses_api_models.openai_model.port=$JUDGE_MODEL_SERVER_PORT \\
    ++nemo_gym.initial_global_config_dict.judge_model.responses_api_models.openai_model.openai_base_url=http://$HOST:$JUDGE_PROXY_PORT/v1 \\
    ++nemo_gym.initial_global_config_dict.judge_model.responses_api_models.openai_model.openai_api_key=dummy \\
    ++nemo_gym.initial_global_config_dict.judge_model.responses_api_models.openai_model.openai_model={judge_model} \\
    ++cluster.ray.min_worker_port=30000 \\
    ++cluster.ray.max_worker_port=40000 \\
    ++grpo.max_num_epochs=2 \\
    ++grpo.num_iterations=1 \\
    ++grpo.num_generations_per_prompt=4 \\
    ++grpo.num_prompts_per_step=2 \\
    ++policy.train_global_batch_size=8 \\
    ++policy.train_micro_batch_size=1 \\
    ++policy.max_total_sequence_length=512 \\
    ++policy.optimizer.kwargs.lr=1e-6 \\
    ++policy.weight_decay=0.01 \\
    ++checkpointing.save_period=1 \\
    ++checkpointing.checkpoint_dir={output_dir}/checkpoints \\
    ++logger.log_dir={output_dir}/logs \\
    ++policy.context_parallel_size=1 \\
    ++policy.tensor_model_parallel_size=1 \\
    ++policy.megatron_cfg.enabled=false \\
    ++cluster.gpus_per_node={policy_num_gpus} \\
    ++cluster.num_nodes=1 \\
    2>&1 | tee {nemo_rl_log}
'"""

training_task = Command(
    command=training_cmd,
    gpus=0,  # 内部用 CUDA_VISIBLE_DEVICES
    name="nemo_rl_training",
    container=CONTAINER,
)

# ============================================================================
# Step 5: Create Single-Node Pipeline
# ============================================================================
print("📦 Step 5: Creating single-node pipeline...")

# All commands in one group on one node
all_commands = CommandGroup(
    commands=[judge_vllm_server, judge_proxy, policy_proxy, training_task],
    hardware=HardwareConfig(
        partition=partition,
        num_nodes=num_nodes,
        num_gpus=num_gpus,
    ),
    name="nemo_rl_with_judge",
    log_dir=f"{output_dir}/slurm_logs",
)

job = {
    "name": "test-nemo-rl-integration-judge",
    "groups": [all_commands]
}

pipeline = Pipeline(
    name="test-nemo-rl-integration-judge",
    cluster_config=cluster_config,
    jobs=[job],
    # 关键：避免复用旧的 nemo-run packaging，确保把本地最新改动打包到远端
    reuse_code=False,
)

# ============================================================================
# Step 6: Run Pipeline
# ============================================================================
print("\n" + "=" * 80)
print("Submitting single-node pipeline...")
print(f"  Container: {CONTAINER}")
print(f"  Node: 1 x {num_gpus} GPUs")
print(f"  Judge: GPU {judge_gpu}")
print(f"  Policy/Training: GPU {policy_gpus}")
print("=" * 80)

result = pipeline.run(dry_run=False, sequential=False)

print("\n✅ Single-node integration test with LLM-as-Judge submitted!")
