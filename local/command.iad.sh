#!/bin/bash

# Example command for running generation on IAD cluster
# Uses the cluster config at: nemo-skills-configs/cluster_configs/latest/iad.yaml
#
# Key differences from local execution:
# 1. --cluster specifies which cluster config to use
# 2. All paths must reference mounted locations (e.g., /workspace instead of ./)
# 3. Jobs are submitted to Slurm via SSH tunnel
# 4. Code is automatically packaged and uploaded to the cluster

# Set required environment variables before running:
# export NEMO_SKILLS_CONFIG_DIR=/path/to/nemo-skills-configs/cluster_configs/latest
# export NV_INFERENCE_URL=https://your-inference-endpoint-url
# export NV_API_KEY=your-api-key

# Verify required environment variables are set
if [ -z "$NEMO_SKILLS_CONFIG_DIR" ]; then
    echo "Error: NEMO_SKILLS_CONFIG_DIR is not set"
    echo "Please set it to your cluster configs directory:"
    echo "  export NEMO_SKILLS_CONFIG_DIR=/path/to/nemo-skills-configs/cluster_configs/latest"
    exit 1
fi

if [ -z "$NV_INFERENCE_URL" ]; then
    echo "Error: NV_INFERENCE_URL is not set"
    echo "Please set it to your inference endpoint:"
    exit 1
fi

# GPU allocation:
# --server_gpus: Number of GPUs per node (default: 1)
# --server_nodes: Number of nodes for distributed inference (default: 1)
ns generate \
    --cluster=iad \
    --expname=test-iad \
    --server_type=openai \
    --model=nvidia/openai/gpt-oss-20b \
    --server_gpus=8 \
    --server_nodes=1 \
    ++server.base_url="$NV_INFERENCE_URL" \
    --output_dir=/workspace/generation \
    --input_file=/nemo_run/code/local/input.jsonl \
    ++prompt_config=/nemo_run/code/local/prompt.yaml \
    ++server.api_key_env_var=NV_API_KEY

