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
NeMo-RL + NeMo-Skills Integration Test for Remote Cluster

This script adapts the local integration test (test_nemo_rl_integration.sh)
to run on remote SLURM clusters like EOS.

Architecture (same as sh script):
  1. NeMo-RL starts with NeMo-Gym, vLLM exposes HTTP (NeMo-RL internally starts Ray)
  2. We detect vLLM URL from logs (using grep, not Ray discovery)
  3. Start proxy with discovered vLLM URL (via environment variable)
  4. NemoGym environment (already created) calls proxy for rollouts
  5. Proxy forwards to vLLM, returns responses

Key differences from local test:
  - Uses cluster='eos' instead of 'test-local'
  - All paths are on the cluster filesystem
  - Submits actual SLURM jobs instead of Docker containers
"""

from nemo_skills.pipeline.utils import get_cluster_config
from nemo_skills.pipeline.utils.declarative import Command, CommandGroup, HardwareConfig, Pipeline

# ============================================================================
# Configuration - Adapt to your cluster
# ============================================================================
# # DFW cluster configuration (commented out - cluster too slow)
# cluster = 'dfw'
# partition = 'batch'
# model = '/lustre/fsw/portfolios/llmservice/users/wedu/models/Qwen2.5-1.5B'
# test_data = '/lustre/fsw/portfolios/llmservice/users/bihu/test-data/nemo-rl-integration-test.jsonl'
# output_dir = '/lustre/fsw/portfolios/llmservice/users/bihu/test-nemo-rl-integration'

# EOS cluster configuration
cluster = 'eos'
partition = 'batch'
model = '/lustre/fsw/llmservice_nemo_reasoning/wedu/models/Qwen2.5-1.5b-genrm-cot-tir'
test_data = '/lustre/fsw/llmservice_nemo_reasoning/bihu/test-data/nemo-rl-integration-test.jsonl'
output_dir = '/lustre/fsw/llmservice_nemo_reasoning/bihu/test-nemo-rl-integration'

# Test parameters (small for quick testing)
# 对齐本地测试：单节点 1 卡
num_nodes = 1
num_gpus = 1
backend = 'fsdp'

print("=" * 80)
print("NeMo-RL + NeMo-Skills Integration Test on Remote Cluster")
print("=" * 80)
print(f"Cluster: {cluster}")
print(f"Model: {model}")
print(f"Test data: {test_data}")
print(f"Output dir: {output_dir}")
print(f"Resources: {num_nodes} nodes × {num_gpus} GPUs")
print("=" * 80)

# ============================================================================
# Step 1: Verify test data exists
# ============================================================================
print("\n📝 Step 1: Verifying test data...")
print(f"Test data file: {test_data}")
print("✅ Data should already be created on the cluster")
print("   (3 simple math problems for testing)")

# ============================================================================
# Step 2: Build a single SLURM job with 2 components (training + proxy)
# ============================================================================
print("\n🚀 Step 2: Submitting a single SLURM job (training + proxy)...")

cluster_config = get_cluster_config(cluster, None)

model_name = model.rstrip("/").split("/")[-1]
# NOTE: On shared clusters, fixed ports frequently collide.
# We derive a per-job port from SLURM_JOB_ID at runtime (same formula used by training + proxy).
# Range: 20000-39999.
proxy_port_formula = "20000 + (SLURM_JOB_ID % 20000)"
proxy_url_for_nemo_gym = "http://$SLURM_MASTER_NODE:$PROXY_PORT/v1"

# 日志文件路径（用于在两个命令之间传递 vLLM URL）
nemo_rl_log = f"{output_dir}/nemo-rl-$SLURM_JOB_ID.log"

print(f"[config] proxy_port_formula={proxy_port_formula} (range 20000-39999)")
print(f"[config] nemo_gym.policy_base_url={proxy_url_for_nemo_gym}")


def create_training_cmd():
    """
    训练命令 - 和 sh 脚本思路一致：
    - 不手动设置 RAY_ADDRESS，让 NeMo-RL 内部自己初始化 Ray
    - 训练日志输出到文件，供 proxy 命令解析 vLLM URL
    """
    return f"""bash -lc 'set -euo pipefail
export PROXY_PORT=$(({proxy_port_formula}))
echo "[training] using PROXY_PORT=$PROXY_PORT (policy_base_url={proxy_url_for_nemo_gym})"

# 关键：确保不设置 RAY_ADDRESS，让 NeMo-RL 自己初始化本地 Ray
# 这和 sh 脚本的思路一致
# unset 以防容器内有预设值
unset RAY_ADDRESS 2>/dev/null || true

# 临时方案：镜像中 NeMo-RL 版本太旧，需要补全（镜像更新后可删除）
# 1) 安装 nemo_gym 包
if ! python3 -c "import nemo_gym" 2>/dev/null; then
    echo "[training] Installing nemo_gym package from GitHub..."
    uv pip install git+https://github.com/NVIDIA-NeMo/Gym.git --quiet 2>/dev/null || \
    pip install git+https://github.com/NVIDIA-NeMo/Gym.git --quiet
    echo "[training] nemo_gym package installed"
else
    echo "[training] nemo_gym package already exists"
fi

# 2) 下载 nemo_rl/environments/nemo_gym.py（如果不存在）
NEMO_RL_ENV_DIR=$(python3 -c "import nemo_rl.environments; print(nemo_rl.environments.__path__[0])")
if [ ! -f "$NEMO_RL_ENV_DIR/nemo_gym.py" ]; then
    echo "[training] Downloading nemo_gym.py from NeMo-RL main branch..."
    curl -sL "https://raw.githubusercontent.com/NVIDIA-NeMo/RL/main/nemo_rl/environments/nemo_gym.py" -o "$NEMO_RL_ENV_DIR/nemo_gym.py"
    echo "[training] nemo_gym.py installed to $NEMO_RL_ENV_DIR/"
else
    echo "[training] nemo_gym.py already exists"
fi

python -u -m nemo_skills.training.nemo_rl.start_grpo \\
  ++policy.model_name={model} \\
  ++cluster.num_nodes={num_nodes} \\
  ++cluster.gpus_per_node={num_gpus} \\
  ++data.train_data_path={test_data} \\
  ++data.prompt.prompt_config=generic/math \\
  ++data.prompt.prompt_template=qwen-instruct \\
  ++policy.generation.vllm_cfg.expose_http_server=true \\
  ++policy.generation.vllm_cfg.async_engine=true \\
  ++policy.generation.vllm_cfg.skip_tokenizer_init=false \\
  ++policy.generation.vllm_cfg.gpu_memory_utilization=0.7 \\
  ++policy.generation.vllm_cfg.tensor_parallel_size=1 \\
  ++policy.megatron_cfg.enabled=false \\
  ++env.should_use_nemo_gym=true \\
  ++nemo_gym.policy_base_url={proxy_url_for_nemo_gym} \\
  ++grpo.max_num_epochs=5 \\
  ++grpo.num_iterations=1 \\
  ++grpo.num_generations_per_prompt=2 \\
  ++grpo.num_prompts_per_step=1 \\
  ++policy.train_global_batch_size=2 \\
  ++policy.train_micro_batch_size=1 \\
  ++policy.max_total_sequence_length=512 \\
  ++policy.optimizer.kwargs.lr=1e-6 \\
  ++policy.weight_decay=0.01 \\
  ++checkpointing.save_period=1 \\
  ++checkpointing.checkpoint_dir={output_dir}/checkpoints \\
  ++logger.log_dir={output_dir}/logs \\
  ++policy.context_parallel_size=1 \\
  ++policy.tensor_model_parallel_size=1 \\
  2>&1 | tee {nemo_rl_log}
'"""


policy_training = Command(
    command=create_training_cmd,
    gpus=num_gpus,  # first component controls sbatch GPU allocation
    nodes=num_nodes,
    name="nemo_rl_training",
    container="nemo-rl",
)


# Proxy 命令 - 和 sh 脚本思路一致：
# 从日志中用 grep 解析 vLLM URL（不是用 Ray 服务发现）
policy_proxy_cmd = f"""bash -lc 'set -euo pipefail
export PROXY_PORT=$(({proxy_port_formula}))
echo "[policy_proxy] binding to :$PROXY_PORT"

# ============================================================================
# 和 sh 脚本一样：从日志中解析 vLLM URL
# grep -oP "Starting server on \\K(http://[0-9.]+:[0-9]+/v1)"
# ============================================================================
echo "[policy_proxy] waiting for vLLM URL in {nemo_rl_log}..."
VLLM_URL=""
for i in $(seq 1 360); do
    if [ -f "{nemo_rl_log}" ]; then
        VLLM_URL=$(grep -oP "Starting server on \\K(http://[0-9.]+:[0-9]+/v1)" {nemo_rl_log} 2>/dev/null | head -1 || true)
        if [ -n "$VLLM_URL" ]; then
            echo "[policy_proxy] found vLLM at: $VLLM_URL"
            break
        fi
    fi
    if [ $((i % 10)) -eq 0 ]; then
        echo "[policy_proxy] ... waiting for vLLM ($i/360)"
    fi
    sleep 2
done

if [ -z "$VLLM_URL" ]; then
    echo "[policy_proxy] ERROR: vLLM URL not found in logs after 12 minutes"
    tail -100 {nemo_rl_log} 2>/dev/null || echo "Log file not found"
    exit 1
fi

# 显式传递 server.base_url 和 server.model 参数
# 环境变量可能不被新版 generate.py 支持，需要用命令行参数
echo "[policy_proxy] starting proxy with server.base_url=$VLLM_URL"
python3 -m nemo_skills.inference.generate \\
  ++start_server=True \\
  ++prompt_format=openai \\
  ++inference.temperature=-1 \\
  ++inference.top_p=-1 \\
  ++generate_port=$PROXY_PORT \\
  ++server_wait_timeout=900 \\
  ++server.base_url=$VLLM_URL \\
  ++server.model={model_name} \\
  ++evaluator.type=math
'"""

policy_proxy = Command(
    command=policy_proxy_cmd,
    gpus=0,
    nodes=1,
    name="policy_proxy",
    container="nemo-rl",
)

group = CommandGroup(
    commands=[policy_training, policy_proxy],
    hardware=HardwareConfig(
        partition=partition,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        # EOS 未配置 GRES，保持默认（不加 gpus-per-node/gres），让调度选择 GPU 节点。
        sbatch_kwargs=None,
    ),
    name="nemo_rl_integration",
    log_dir=f"{output_dir}/slurm_logs",
)

job = {"name": "test-nemo-rl-integration", "group": group}

pipeline = Pipeline(
    name="test-nemo-rl-integration",
    cluster_config=cluster_config,
    jobs=[job],
    # 不复用旧实验打包，避免沿用含 gres 的旧 sbatch。
    reuse_code=False,
)

pipeline.run(dry_run=False, sequential=False)

print("\n✅ Integration test job submitted!")
print("\n📊 Monitor job status:")
print("   nemo_run list")
print("\n📁 Check outputs:")
print(f"   ls -lh {output_dir}")
print("\n📝 View logs:")
print(f"   ls -lh {output_dir}/slurm_logs")
