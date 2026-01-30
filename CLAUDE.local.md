# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Nemo-Skills is a collection of pipelines to improve "skills" of large language models (LLMs). The system supports the full lifecycle: synthetic data generation → model training → evaluation on 80+ benchmarks. It scales from local workstations to large Slurm clusters with minimal code changes.

**Documentation:** https://nvidia-nemo.github.io/Skills/
- Basics: https://nvidia-nemo.github.io/Skills/basics
- Pipelines: https://nvidia-nemo.github.io/Skills/pipelines
- Tutorials: https://nvidia-nemo.github.io/Skills/tutorials

## Quick Start

```bash
# Install
pip install -e .[dev]

# Setup cluster config (required for local/cluster execution)
ns setup

# For development: Install pre-commit hooks (required - enforces DCO sign-off)
pre-commit install

# All commits MUST be signed off
git commit -s -m "Your message"

# Run tests
pytest                    # Unit tests (CPU only)
pytest -m gpu            # GPU tests
pytest tests/gpu-tests/  # Specific GPU tests
```

**Code Quality:**
- Line length: 119 characters
- Target: Python 3.10+
- Ruff for formatting/linting (pre-commit handles this)
- Do not add arbitrary defaults for configs - be explicit

## Key Environment Variables

**Cluster Configuration:**
- `NEMO_SKILLS_CONFIG` - Set default cluster (instead of `--cluster`)
- `NEMO_SKILLS_CONFIG_DIR` - Override default config directory
- `NEMO_SKILLS_DATA_DIR` - Cluster data directory path
- `NEMO_SKILLS_EXTRA_DATASETS` - Additional dataset search paths

**API Keys:**
- `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, `NVIDIA_API_KEY` - For API providers
- `HF_TOKEN` - For gated models (e.g., Llama 3.1)
- `WANDB_API_KEY` - For training logging

**Critical for Cluster:**
- `HF_HOME` - Must be set in cluster config's `env_vars` for HuggingFace models

**SSH & Code Packaging:**
- `NEMO_SKILLS_SSH_KEY_PATH`, `NEMO_SKILLS_SSH_SERVER` - SSH tunnel config
- `NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1` - Allow uncommitted changes (but won't be packaged!)

## Command Basics

### Argument Convention
- `--arg_name` - Wrapper arguments (job config: cluster, GPUs, expname, etc.)
- `++arg_name` - Main arguments (passed to underlying script)
- Run `ns <command> --help` to see all options and underlying script path

### Common Wrapper Arguments
- `--cluster=local|slurm` - Execution environment
- `--expname=my-exp` - Experiment name
- `--run_after=other-exp` - Job dependency (SLURM only)
- `--num_chunks=N` - Split into N parallel jobs
- `--num_random_seeds=N` - Generate N samples per input
- `--num_jobs=N` - Parallelize across N SLURM jobs
- `--reuse_code_exp=exp-name` - Reuse code from previous experiment

### Basic Commands

**Generation:**
```bash
# API-based
ns generate --server_type=openai --model=gpt-4o-mini --output_dir=./output --input_file=./input.jsonl ++prompt_config=generic/math

# Local with vLLM
ns generate --cluster=local --server_type=vllm --model=Qwen/Qwen2.5-1.5B-Instruct --server_gpus=1 --output_dir=/workspace/output --input_file=/workspace/input.jsonl ++prompt_config=generic/math
```

**Evaluation:**
```bash
ns prepare_data gsm8k math aime24
ns eval --cluster=local --benchmarks=gsm8k:4,math:8 --model=/models/my-model --server_type=vllm --output_dir=/workspace/results
ns summarize_results /workspace/results
```

**Training:**
```bash
ns sft_nemo_rl --cluster=slurm --expname=my-sft --model=/models/base-model --training_data=/data/train.jsonl
ns grpo_nemo_rl --cluster=slurm --expname=my-grpo --model=/models/sft-model --training_data=/data/train.jsonl
```

**Server:**
```bash
ns start_server --cluster=local --server_type=vllm --model=meta-llama/Llama-3.1-8B-Instruct --num_gpus=1
```

**Run Arbitrary Commands:**
```bash
ns run_cmd --cluster=slurm --cmd="python scripts/my_script.py" --expname=custom-job
```

**Find detailed examples:** Run `ns <command> --help` or check documentation at https://nvidia-nemo.github.io/Skills/pipelines

### Important Command Options

**Generation:**
- `++code_execution=true` - Enable sandboxed code execution
- `++parse_reasoning=true ++end_reasoning_string='</think>'` - For reasoning models
- `++prompt_format=openai` - When data is already formatted as OpenAI messages
- `++server.enable_soft_fail=true` - Continue on errors (vLLM/SGLang)
- `++skip_filled=false` - Force rerun even if output exists

**Evaluation:**
- `--benchmarks=gsm8k:10,math:10` - `:N` specifies number of repeats (computes variance)
- `--data_dir=/workspace/ns-data` - Use cluster-prepared data
- `--judge_model=gpt-4o --judge_server_type=openai` - LLM-as-a-judge
- `++eval_config.timeout=60` - Custom code execution timeout

**Training:**
- `++backend=megatron` - Use Megatron backend (faster than FSDP)
- `++policy.train_micro_batch_size=1` - Keep at 1 for sequence packing!
- `++policy.context_parallel_size=4` - For sequences > 4k tokens
- `--disable_wandb` - Disable WandB logging

**Server:**
- `--create_tunnel` - Create SSH tunnel for remote SLURM
- `--get_random_port` - Avoid port conflicts on shared nodes
- `--launch_chat_interface` - Start chat UI (default port: 7860)

## Architecture Overview

### Core Patterns

**1. Registry + Factory Pattern** - Used everywhere for pluggability
```python
models = {"vllm": VLLMModel, "openai": OpenAIModel, ...}
def get_model(server_type, **kwargs):
    return models[server_type](**kwargs)
```
- Model backends: `nemo_skills/inference/model/__init__.py`
- Evaluators: `nemo_skills/evaluation/evaluator/__init__.py`
- Datasets: `nemo_skills/dataset/`

**2. Declarative Pipeline System** - Layered job specification
```
Pipeline → Jobs → CommandGroups → Commands → Scripts
```
- Key files: `nemo_skills/pipeline/utils/declarative.py`, `scripts.py`, `exp.py`, `cluster.py`
- Cross-component references for coordination (server hostname/port auto-shared with clients)
- Script types: `ServerScript`, `SandboxScript`, `GenerationClientScript`, `BaseJobScript`

**3. Cluster Abstraction** - Same code runs locally or on SLURM
- Executor types: `none` (direct), `local` (Docker), `slurm` (SLURM+containers)
- Config location: `cluster_configs/`
- SSH tunnel support for remote job submission

**4. Async Generation** - High concurrency via async/await
- Default: 512 concurrent requests (configurable)
- Unified interface via LiteLLM across all backends

**5. Chunking & Seeding** - Massive parallelization
- `--num_chunks=100` splits dataset across jobs
- `--num_random_seeds=10` generates multiple samples per input
- Auto `.done` file tracking prevents re-running completed work

### Key Systems

**Inference:**
- Model backends: `nemo_skills/inference/model/` (vLLM, SGLang, Megatron, OpenAI, Azure, Gemini, etc.)
- Server scripts: `nemo_skills/inference/server/`
- Generation types: `nemo_skills/inference/factory.py` (generate, math_judge, check_contamination)
- Advanced wrappers: `CodeExecutionWrapper`, `ToolCallingWrapper`, `ParallelThinkingTask`

**Evaluation:**
- Base: `nemo_skills/evaluation/evaluator/base.py` - defines `eval_full()`, `eval_single()`, `supports_single_eval()`
- Evaluators: `evaluation/evaluator/` - math, code, SQL, instruction-following, function-calling, LLM-judge, etc.
- Metrics: `evaluation/metrics/` - supports pass@k, majority voting, variance analysis

**Datasets:**
- 80+ benchmarks in `nemo_skills/dataset/`
- Each has `__init__.py` with `DATASET_GROUP`, `METRICS_TYPE`, `GENERATION_ARGS`
- Categories: math (natural/formal), code, science, multilingual, tool-calling, long-context, VLM, audio
- Discovery: `dataset/utils.py` searches multiple locations

**Prompts:**
- YAML configs: `nemo_skills/prompt/config/` (generic/, model-specific/, judge/, vlm/, etc.)
- Few-shot examples: `prompt/few_shot_examples/` (includes BM25 retrieval)
- Code tags: `prompt/code_tags/` (defines execution markers)

**Training:**
- Data prep: `nemo_skills/training/prepare_data.py` (uses SDP pipeline)
- NeMo-RL: `training/nemo_rl/` (SFT, GRPO, reward computation)
- veRL: `training/verl/` (PPO)
- Pipeline integration: `pipeline/nemo_rl/`, `pipeline/verl/`

**Code Execution:**
- Sandbox: `nemo_skills/code_execution/sandbox.py:398` (network blocking, timeouts, safety)
- Lean4 support: `code_execution/proof_utils.py`

**Tool Calling (MCP-based):**
- Architecture: LLM → ToolManager → MCPClientTool → MCP Server
- Built-in: `nemo_skills.mcp.servers.python_tool.PythonTool`, `nemo_skills.mcp.servers.exa_tool.ExaTool`
- Usage: `++tool_modules=[...]` with `--server_args="--enable-auto-tool-choice --tool-call-parser hermes"`

**Parallel Thinking:**
- Modes: `genselect` (select best), `gensynthesis` (synthesize new)
- Online: generates N solutions then processes
- Offline: uses pre-generated solutions (enables different generator/processor models)
- Config: `++parallel_thinking.mode=genselect ++parallel_thinking.window_size=8`

### Cluster Configuration

**Config file structure** (`cluster_configs/*.yaml`):
```yaml
executor: slurm
ssh_tunnel:
  host: cluster.domain
  user: username
  job_dir: /path/on/cluster
containers:
  vllm: docker.io/vllm/vllm:latest
  nemo-skills: dockerfile:dockerfiles/Dockerfile.nemo-skills
mounts:
  - /data:/data
  - /models:/models
env_vars:
  - HF_HOME=/models/hf-cache
  - HF_TOKEN  # References local environment
required_env_vars:
  - WANDB_API_KEY  # Must be set locally, fails if not
```

**Environment variable types:**
- `env_vars` - Can have inline values or reference local env (fails silently if not set)
- `required_env_vars` - Must be set in local environment (fails if not set)

## Critical Concepts

### Code Packaging

**ALL `ns` commands package code automatically:**
- Available in containers at `/nemo_run/code`
- Stored locally: `~/.nemo_run/experiments/<expname>`
- Stored on cluster: `job_dir/<expname>` (from cluster config)

**Packaging behavior (from `nemo_skills/pipeline/utils/packager.py`):**
1. **From nemo-skills repo**: Packages nemo-skills repo only
2. **From another git repo**: Packages that repo + installed nemo_skills
3. **Outside any git repo**: Only packages installed nemo_skills (ALL files if not git-based!)

**CRITICAL CONSTRAINTS:**
- **Only git-tracked files are packaged** (exception: `.jsonl` files in `nemo_skills/dataset/`)
- Uncommitted changes will NOT be packaged, even with `NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1`
- Non-git nemo_skills installations upload ALL files (watch for large files!)
- Code copies accumulate - must manually clean `~/.nemo_run` and cluster `job_dir` periodically
- Use `--reuse_code_exp=<expname>` to avoid re-packaging

**Workflow:**
```bash
git add file.py && git commit -s -m "Add file"
ns generate ... --input_file=/nemo_run/code/file.py
```

### Container Usage

**Most commands run in containers specified in cluster config:**
- Packages installed locally are NOT available unless added to containers
- All paths must reference mounted paths (not local filesystem paths)
- Environment variables need explicit listing in cluster config's `env_vars`

### Job Dependencies

**SLURM only** (ignored for local execution):
```bash
ns run_cmd --cluster=slurm --expname=download-model --cmd="python download.py"
ns eval --cluster=slurm --benchmarks=gsm8k:4 --model=/models/my-model --run_after=download-model
```

## Code Guidelines (from CONTRIBUTING.md)

### Fail-Fast Philosophy

**This codebase prioritizes fail-fast over silent errors:**
- ❌ NO `.get()` for expected keys - use `data[key_name]` to fail with clear KeyError
- ❌ NO unnecessary exception catching - let code crash on unexpected conditions
- ❌ NO silently ignoring parameters - fail if user passes unsupported parameter
- ✅ Security not a concern (users have full system access) - `subprocess.call(..., shell=True)` is fine
- ⚠️ Exception: LLM-generated code should use sandbox API

### Keep Code Elegant

- Reuse/extend existing functionality when possible
- Avoid checking too many conditions or complicated logic
- Write simpler code even if it sacrifices rare edge-cases
- Make code self-explanatory rather than adding excessive comments
- Use simple types (dict, list, int, float, existing classes)
- Avoid complicated type unions/interfaces
- Follow existing naming conventions

**Zen of Python:** "Beautiful is better than ugly. Explicit is better than implicit. Simple is better than complex."

## Key Directories

```
nemo_skills/
├── pipeline/          # CLI & orchestration (cli.py is entry point)
│   ├── generate.py   # Generation pipeline
│   ├── eval.py       # Evaluation pipeline
│   ├── utils/        # declarative.py, scripts.py, exp.py, cluster.py
│   ├── nemo_rl/      # SFT, GRPO pipelines
│   └── verl/         # PPO pipelines
├── inference/        # Generation, model backends, servers
│   ├── generate.py   # Main generation script
│   ├── factory.py    # Generation type registry
│   ├── model/        # Backend implementations + registry
│   └── server/       # Server startup scripts
├── evaluation/       # Evaluators & metrics
├── dataset/          # 80+ benchmark datasets
├── training/         # NeMo-RL, veRL integration
├── prompt/           # YAML configs, few-shot examples, code tags
├── code_execution/   # Sandboxed execution
└── mcp/             # Tool calling (MCP protocol)

cluster_configs/      # Execution environment configs
tests/
├── gpu-tests/        # GPU-required tests
└── slurm-tests/      # End-to-end cluster tests
recipes/              # Reproducible experiments
docs/                 # MkDocs documentation
```

## Common Pitfalls

**Setup & Configuration:**
- Forgetting to run `ns setup` before using cluster features
- Not mounting paths correctly in cluster configs - all paths in jobs must reference mounted locations
- Using relative paths instead of absolute paths in cluster jobs
- Not setting `HF_HOME` in cluster config's `env_vars` (required for HuggingFace models)
- Not setting `HF_TOKEN` for gated models
- Using container paths that aren't mounted
- Expecting local packages in containers - containers only have what's in Dockerfile

**Code Management:**
- Not signing off commits - use `git commit -s` (pre-commit will catch this)
- Forgetting code is packaged from git - only committed files available
- Using uncommitted changes - won't be packaged even with env var set
- Using `git add -A` or `git add .` - stage specific files to avoid committing secrets
- Non-git nemo_skills installations package ALL files (watch for large files!)
- Not cleaning `~/.nemo_run` and cluster `job_dir` - code copies accumulate

**Generation & Evaluation:**
- Missing reasoning parser for reasoning models - must set `--server_args="--reasoning-parser ..."` or `++parse_reasoning=true`
- Wrong prompt format - check if data is already formatted (use `++prompt_format=openai`)
- Not preparing large datasets on cluster - datasets like ruler should be prepared on cluster with `--data_dir`
- Forgetting `.done` files exist - completed jobs are auto-skipped; use `++skip_filled=false` to force rerun
- Not checking first printed prompt - verify formatting before expensive jobs
- `preprocess_cmd` with `num_chunks>1` doesn't work together

**Cluster & Job Management:**
- Using local tunnel ports without customization - can conflict on shared nodes; use `--get_random_port`
- Not using `run_after` for dependencies
- Hitting SLURM time limits - use `--dependent_jobs=N` to chain jobs
- Not using `--reuse_code_exp` - unnecessarily re-packages code

**Training:**
- Not using `train_micro_batch_size=1` with sequence packing - REQUIRED
- Forgetting `context_parallel_size` for long sequences (>4k tokens)
- Not using `--disable_wandb` when `WANDB_API_KEY` not set

**Development:**
- Adding unnecessary defensive code - violates contribution guidelines
- Using `.get()` for expected keys - use direct access to fail fast
- Catching exceptions unnecessarily - let code crash

## Quick Tips

**Setup:**
- Use `ns setup` to create cluster configs interactively
- Set `HF_HOME` in cluster config's `env_vars`
- For large datasets, prepare on cluster: `ns prepare_data ruler --data_dir=/workspace/ns-data --cluster=slurm`
- Build `.sqsh` containers on cluster to avoid repeated downloads
- Clean `~/.nemo_run` and cluster `job_dir` periodically

**Debugging:**
- `ns <command> --help` - see all arguments and underlying script path
- `++log_level=debug` - verbose logging
- Check first printed prompt to verify formatting
- Inspect sbatch files in cluster `job_dir/<expname>` for manual modification
- `ns summarize_results /path/to/results` - recompute metrics

**Performance:**
- `--num_jobs=N` - parallelize evaluations across SLURM jobs
- `--num_chunks=N` - split large generation jobs
- `--num_random_seeds=N` - for pass@k or majority voting
- `--dependent_jobs=M` - chain jobs hitting time limits
- `--reuse_code_exp=<expname>` - avoid re-packaging code

**Job Management:**
- Use `expname` and `run_after` for dependencies
- Check `.done` files for completion status
- `++skip_filled=false` to force rerun
- Prefix paths with `/nemo_run/code/` to reference packaged files

**Config Files for Special Characters:**
- Use `--config-path` and `--config-name` for arguments with special characters (e.g., `</think>`)
- Avoids shell escaping issues

## Key Files Reference

**Entry Points:**
- `nemo_skills/pipeline/cli.py:47` - All CLI commands registered here
- `nemo_skills/inference/generate.py` - Core generation logic

**Registries:**
- `nemo_skills/inference/model/__init__.py` - Model backend registry
- `nemo_skills/evaluation/evaluator/__init__.py` - Evaluator registry
- `nemo_skills/inference/factory.py` - Generation type registry

**Core Systems:**
- `nemo_skills/pipeline/utils/packager.py` - Code packaging logic
- `nemo_skills/pipeline/utils/declarative.py` - Pipeline/Job/CommandGroup/Command classes
- `nemo_skills/pipeline/utils/scripts.py` - Script objects (ServerScript, etc.)
- `nemo_skills/pipeline/utils/exp.py` - Experiment management
- `nemo_skills/pipeline/utils/cluster.py` - Cluster config resolution, SSH tunneling

**Configuration:**
- `pyproject.toml` - Package metadata, scripts (`ns` command), tool configs
- `requirements/main.txt` - Production dependencies
- `requirements/common-dev.txt` - Development dependencies

## Where to Find More

**Documentation:** https://nvidia-nemo.github.io/Skills/
- **Basics:** https://nvidia-nemo.github.io/Skills/basics - Getting started, setup, core concepts
- **Pipelines:** https://nvidia-nemo.github.io/Skills/pipelines - Detailed command reference with all options
- **Tutorials:** https://nvidia-nemo.github.io/Skills/tutorials - Full workflows (GPT-OSS-120B, Nemotron evals, AIMO2 pipeline)

**In Repository:**
- **Recipes:** `recipes/` - Reproducible experiment configurations
- **Examples:** Look at existing dataset `__init__.py` files in `nemo_skills/dataset/` for benchmark setup patterns
- **Tests:** `tests/gpu-tests/`, `tests/slurm-tests/` - Real-world usage examples
- **Cluster configs:** `cluster_configs/` - Example configurations

**Command Help:**
- Run `ns <command> --help` to see wrapper arguments and underlying script path
- Look at underlying script (e.g., `nemo_skills/inference/generate.py`) for all `++` arguments

**Specific Topics:**
- **Benchmark-specific notes:** Check documentation tutorials for SWE-Bench, LiveCodeBench, IOI, Lean4, RULER, gpt-oss-120b
- **Advanced features:** Tool calling (MCP), parallel thinking (GenSelect/GenSynthesis), LLM-as-a-judge configs
- **Training details:** NeMo-RL optimization parameters, checkpoint management, MoE configuration
