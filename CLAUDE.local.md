# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Nemo-Skills is a collection of pipelines to improve "skills" of large language models (LLMs). The system supports the full lifecycle: synthetic data generation → model training → evaluation on 80+ benchmarks. It scales from local workstations to large Slurm clusters with minimal code changes.

## Development Setup

### Installation

```bash
# Development install with all dependencies
pip install -e .[dev]

# Initialize cluster configuration (required for most operations)
ns setup

# Install pre-commit hooks (required)
pre-commit install
```

### Pre-commit Requirements

All commits MUST be signed off:
```bash
git commit -s -m "Your message"
```

This adds `Signed-off-by: Your Name <your@email.com>` to comply with DCO requirements.

### Code Quality

- Run `ruff check --fix` and `ruff format` before committing (pre-commit handles this)
- Line length: 119 characters
- Target: Python 3.10+
- Do not add arbitrary defaults for configs - be explicit

### Running Tests

```bash
# Run all unit tests (CPU-only, excludes GPU and slurm tests)
pytest

# Run specific test file
pytest tests/test_declarative_pipeline.py

# Run GPU tests (requires GPU access)
pytest -m gpu
pytest tests/gpu-tests/test_generate.py

# Slurm tests are comprehensive end-to-end tests on cluster
# Located in tests/slurm-tests/ - see tests/slurm-tests/README.md for details
```

Tests marked with `@pytest.mark.gpu` require GPU access.

## Common Commands

### Basic Generation

```bash
# API-based inference
ns generate \
    --server_type=openai \
    --model=gpt-4o-mini \
    --output_dir=./output \
    --input_file=./input.jsonl \
    ++prompt_config=generic/math

# Local model with vLLM
ns generate \
    --cluster=local \
    --server_type=vllm \
    --model=Qwen/Qwen2.5-1.5B-Instruct \
    --server_gpus=1 \
    --output_dir=/workspace/output \
    --input_file=/workspace/input.jsonl \
    ++prompt_config=generic/math
```

### Evaluation

```bash
# Prepare benchmark data
ns prepare_data gsm8k math aime24

# Evaluate on benchmarks
ns eval \
    --cluster=local \
    --benchmarks=gsm8k:4,math:8 \
    --model=/models/my-model \
    --server_type=vllm \
    --output_dir=/workspace/results

# Evaluate vision-language models (VLM)
ns eval \
    --cluster=local \
    --benchmarks=mmmu-pro:4 \
    --model=/models/my-vlm \
    --server_type=vllm \
    --output_dir=/workspace/results

# Summarize results
ns summarize_results /workspace/results
```

### Training

```bash
# Prepare training data
ns prepare_data --dataset=openmathinstruct2

# SFT with NeMo-RL
ns sft_nemo_rl \
    --cluster=slurm \
    --expname=my-sft-run \
    --model=/models/base-model \
    --training_data=/data/train.jsonl

# GRPO with NeMo-RL
ns grpo_nemo_rl \
    --cluster=slurm \
    --expname=my-grpo-run \
    --model=/models/sft-model \
    --training_data=/data/train.jsonl
```

### Server Management

```bash
# Start a model server
ns start_server \
    --cluster=local \
    --server_type=vllm \
    --model=meta-llama/Llama-3.1-8B-Instruct \
    --num_gpus=1
```

### Running Arbitrary Commands

```bash
# Execute custom command in container
ns run_cmd \
    --cluster=slurm \
    --cmd="python scripts/my_script.py" \
    --expname=custom-job
```

### Documentation

```bash
# Serve documentation locally (requires mkdocs)
mkdocs serve

# Build documentation
mkdocs build
```

## Architecture

### CLI System (`ns` command)

Entry point: `nemo_skills/pipeline/cli.py:47`

Built with Typer. Main commands map to functions in `nemo_skills/pipeline/`:
- `generate` → `generation.py`
- `eval` → `eval.py`
- `prepare_data` → `data_preparation.py`
- `sft_nemo_rl`, `grpo_nemo_rl` → `nemo_rl/`
- `ppo_verl` → `verl/`
- `start_server` → `start_server.py`

### Declarative Pipeline System

**Core concept**: Layered job specification

```
Pipeline (experiment-level)
  └─ Jobs (can depend on other jobs via run_after/expname)
      └─ CommandGroup(s) (SLURM heterogeneous job groups)
          └─ Command(s) (individual processes in containers)
              └─ Script objects (typed, reusable components)
```

**Key files**:
- `nemo_skills/pipeline/utils/declarative.py:832` - Pipeline/Job/CommandGroup/Command classes
- `nemo_skills/pipeline/utils/scripts.py:428` - Script objects (ServerScript, GenerationClientScript, etc.)
- `nemo_skills/pipeline/utils/exp.py:780` - Experiment management, code packaging
- `nemo_skills/pipeline/utils/cluster.py:580` - Cluster config resolution, SSH tunneling

**Script types**:
- `ServerScript` - Model serving (vLLM, TRT-LLM, SGLang, Megatron)
- `SandboxScript` - Code execution sandbox
- `GenerationClientScript` - LLM inference client
- `BaseJobScript` - Base class with heterogeneous job support

**Key Pattern**: Cross-component references allow scripts to reference each other for coordination (e.g., client gets server hostname/port automatically).

### Inference System

**Architecture**: Pluggable backend with unified interface

**Model backends** (`nemo_skills/inference/model/`):
- `base.py` - BaseModel abstract class (uses LiteLLM for unified API)
- `vllm.py`, `sglang.py`, `megatron.py` - Self-hosted backends
- `openai.py`, `azure.py`, `gemini.py` - API providers
- `vllm.py` - Includes vision-language model (VLM) support for multimodal inputs
- `asr_nim.py`, `tts_nim.py` - Speech/audio models

**Model registry** (`inference/model/__init__.py`):
```python
models = {
    "trtllm": VLLMModel,  # TRT-LLM uses vLLM wrapper
    "vllm": VLLMModel,
    "sglang": SGLangModel,
    "openai": OpenAIModel,
    # ...
}
```

Factory: `get_model(server_type, ...)` instantiates appropriate backend.

**Advanced wrappers**:
- `CodeExecutionWrapper` - Integrates sandbox for iterative code execution
- `ToolCallingWrapper` - Function calling support (BFCL benchmarks)
- `ParallelThinkingTask` - Multi-path reasoning

**Generation types** (`inference/factory.py`):
- `generate` - Standard generation
- `math_judge` - LLM-based math judging
- `check_contamination` - Dataset contamination checking
- Custom generation modules can specify their own Python environments/dependencies

**Server management** (`inference/server/`):
- `serve_vllm.py`, `serve_sglang.py` - Server startup scripts
- Automatic port allocation
- Multi-node distributed inference support

### Evaluation System

**Architecture**: Evaluator + Metrics pattern

**Base evaluator** (`nemo_skills/evaluation/evaluator/base.py`):
```python
class BaseEvaluator:
    async def eval_full()      # Batch evaluation
    async def eval_single()    # Single sample (streaming during generation)
    def supports_single_eval() # Check if streaming supported
```

**Specialized evaluators** (`evaluation/evaluator/`):
- `math.py` - Math answer checking (symbolic, numeric, extraction)
- `code.py` - Code execution evaluation
- `bird.py` - SQL evaluation
- `ifeval.py` - Instruction following
- `bfcl.py` - Function calling benchmarks (with web search support for BFCLv4)
- `arena.py` - LLM-as-judge evaluation
- `nvembed_judge.py` - Embedding-based judging
- `scicode.py` - Scientific code evaluation
- `audio.py` - Audio input/output evaluation (ASR, TTS, audio understanding)

**Metrics system** (`evaluation/metrics/`):
- Benchmark-specific metric calculators
- `map_metrics.py` - Maps benchmark names to metric functions
- Supports aggregation: pass@k, majority voting, avg-of-N

**Evaluation pipeline**:
1. Main generation jobs (chunked/seeded for parallelization)
2. Judge jobs (if needed) - LLM or rule-based
3. Automatic metric computation and summarization
4. Group scoring for benchmark suites

### Dataset System

**Architecture**: Plugin-based dataset modules

Each benchmark is a directory in `nemo_skills/dataset/` with `__init__.py`:
```python
DATASET_GROUP = "math"           # Category
METRICS_TYPE = "math"            # Which metrics to use
GENERATION_ARGS = "++prompt_config=generic/math"  # Default args
```

**Dataset discovery** (`dataset/utils.py`):
- Searches in: `nemo_skills/dataset/`, `data_dir` parameter, `NEMO_SKILLS_EXTRA_DATASETS` env var
- Supports cluster-mounted paths with automatic download

**80+ benchmarks organized by category**:
- **Math (natural)**: gsm8k, math, aime24, aime25, hmmt_feb25, imo-answerbench, etc.
- **Math (formal)**: minif2f, proofnet, putnam-bench, imo-proofbench, imo-gradingbench, etc.
- **Code**: swe-bench, swe-bench-multilingual, livecodebench, bigcodebench, bird, etc.
- **Science**: gpqa, hle, scicode, frontierscience-olympiad, omniscience, etc.
- **Multilingual**: flores-200, wmt24pp, mmlu-prox, etc.
- **Tool-calling**: bfcl_v3, bfcl_v4
- **Long-context**: ruler, ruler2, mrcr, aalcr
- **Instruction-following**: ifbench, ifeval
- **Speech/Audio**: asr-leaderboard, mmau-pro
- **VLM**: mmmu-pro

### Prompt System

**Architecture**: YAML-based templates with dynamic construction

**Prompt configs** (`nemo_skills/prompt/config/`):
- `generic/` - Universal prompts (math, codegen, etc.)
- `llama3-instruct/`, `qwen/` - Model-specific formats
- `judge/` - Prompts for LLM judges (includes IMO and FrontierScience judges)
- `vlm/` - Vision-language model prompts (e.g., mmmu-pro)
- `multilingual/` - Language-specific prompts
- `eval/aai/` - AAI benchmark-specific prompts

**Example YAML structure** (`generic/math.yaml`):
```yaml
few_shot_examples:
  prefix: "Here are some examples..."
  template: "Problem:\n{problem}\n\nSolution:\n{solution}"
  suffix: "Here is the problem to solve:\n"

user: |-
  Solve the following math problem...
  {examples}{problem}
```

**Few-shot examples** (`prompt/few_shot_examples/`):
- `math.py`, `gsm8k.py`, `lean4.py`, etc.
- BM25-based example retrieval support
- Dynamic example selection

**Code tags** (`prompt/code_tags/`):
- Defines markers for code execution (```python, ```output)
- Format-specific output formatting

**Prompt construction flow**:
1. Load YAML config via `++prompt_config=<category>/<name>`
2. Resolve few-shot examples (if specified)
3. Fill template with data fields
4. Apply tokenizer's chat template (for chat models)

### Training System

**Architecture**: Integration with NeMo-RL and veRL

**Data preparation** (`nemo_skills/training/prepare_data.py`):
- Uses SDP (Speech Data Processor) pipeline
- Configuration in `data_preparation_utils/config/`
- Filters, processors, and merging utilities

**NeMo-RL backend** (`training/nemo_rl/`):
- `start_sft.py` - Supervised fine-tuning
- `start_grpo.py` - Group relative policy optimization
- `environments/math_environment.py` - Reward computation
- Checkpoint conversion utilities

**veRL backend** (`training/verl/`):
- `prepare_data.py` - veRL-specific data formatting
- PPO training support

**Pipeline integration** (`pipeline/nemo_rl/`, `pipeline/verl/`):
- `sft.py`, `grpo.py`, `ppo.py` - Training job creation
- Automatic resource allocation
- Checkpoint management
- Ray template support for distributed training (NeMo-RL)

### Code Execution System

**Architecture**: Sandboxed execution with safety controls

**Core sandbox** (`nemo_skills/code_execution/sandbox.py:398`):
- Network blocking support (via iptables)
- Timeout controls
- Mount restrictions for safety
- Output formatting (configurable via code_tags)

**Proof/formal math** (`code_execution/proof_utils.py`):
- Lean4 interaction utilities
- Formal proof verification

**Integration**:
- `SandboxScript` allocates sandbox alongside inference
- `CodeExecutionWrapper` coordinates LLM ↔ sandbox communication
- Supports iterative code generation and execution

## Key Architectural Patterns

### 1. Multi-Backend Support

All systems use **registry + factory pattern**:

```python
# Registry
models = {"vllm": VLLMModel, "openai": OpenAIModel, ...}

# Factory
def get_model(server_type, **kwargs):
    return models[server_type](**kwargs)
```

Enables seamless switching between:
- **Inference**: TensorRT-LLM, vLLM, SGLang, Megatron, OpenAI, Azure, Gemini
- **Training**: NeMo-RL, veRL
- **Execution**: Local, cluster, containers

### 2. Cluster Abstraction

**Executor types** (`pipeline/utils/cluster.py`):
- `none` - Direct execution (no containers)
- `local` - Docker with host networking
- `slurm` - SLURM with containers

**Cluster config** (`cluster_configs/`):
```yaml
executor: slurm
ssh_tunnel:  # Optional SSH access from local machine
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
```

**SSH tunnel support**: Submit jobs from local machine to remote SLURM cluster.

### 3. Heterogeneous SLURM Jobs

Declarative pipeline supports different resource requirements in single job:

```python
# Example: 8B model (1 GPU) + 32B model (4 GPUs) in one job
job = {
    "name": "multi_model",
    "groups": [
        CommandGroup([server_8b, client_8b], HardwareConfig(num_gpus=1)),
        CommandGroup([server_32b, client_32b], HardwareConfig(num_gpus=4))
    ]
}
```

Cross-component references use SLURM env vars: `$SLURM_MASTER_NODE_HET_GROUP_0`, etc.

### 4. Configuration Management

- **Hydra for generation**: `inference/generate.py` uses Hydra for complex configs
  - Usage: `ns generate ++prompt_config=math ++temperature=0.7`
- **Typer for pipelines**: `pipeline/` commands use Typer for type-safe CLIs
  - Usage: `ns eval --benchmarks=gsm8k:4 --model=/models/llama`
- **YAML for prompts/data**: Declarative configs in `prompt/config/` and `dataset/`

**Arguments convention**:
- `--arg_name` - Wrapper arguments (job config: cluster, GPUs, etc.)
- `++arg_name` - Main arguments (passed to underlying script)

Run `ns <command> --help` to see wrapper args and underlying script path.

### 5. Async Generation

Generation uses async/await for high concurrency (`inference/generate.py`):

```python
max_concurrent_requests = 512  # Configurable
semaphore = asyncio.Semaphore(max_concurrent_requests)

async def generate_batch(...):
    async with semaphore:
        return await model.generate(...)
```

LiteLLM provides unified async interface across all backends.

### 6. Chunking and Seeding

Supports massive parallelization (`pipeline/generate.py`):

```bash
# Split dataset into 100 chunks, run 10 seeds each
ns generate \
    --num_chunks=100 \
    --num_random_seeds=10 \
    --output_dir=/results
```

Creates: `output-rs0-chunk0.jsonl`, `output-rs0-chunk1.jsonl`, ..., `output-rs9-chunk99.jsonl`

Automatic `.done` file tracking prevents re-running completed jobs.

## Important Concepts

### Code Packaging

All `ns` commands package your code automatically:
- Packages any git repo you're running from + nemo-skills package
- Available in containers/SLURM at `/nemo_run/code`
- Stored in `~/.nemo_run/experiments/<expname>` locally
- Stored in `job_dir/<expname>` on cluster (from cluster config)

**Important**: Only committed files are packaged!

```bash
git add file.py && git commit -m "Add file"
ns generate ... --input_file=/nemo_run/code/file.py
```

### Container Usage

Most commands run in containers specified in cluster config:
- Packages installed locally are NOT available unless added to containers
- All paths must reference mounted paths (not local filesystem paths)
- Environment variables need explicit listing in cluster config's `env_vars`

### Slurm Job Scheduling

Best practice: Submit from local machine via SSH tunnel in cluster config.

**Job dependencies**:
```python
run_cmd(expname="download-model", ...)
eval(run_after="download-model", ...)  # Waits for download to complete
```

Only used with SLURM clusters, ignored for local execution.

## Data Flow Example: Evaluation

1. **User**: `ns eval --benchmarks=gsm8k:4 --model=/models/llama --server_type=vllm`
2. **CLI** (`pipeline/cli.py`) → Routes to `eval()` in `pipeline/eval.py`
3. **Eval orchestration** (`pipeline/eval.py`):
   - Loads benchmark config from `dataset/gsm8k/__init__.py`
   - Loads prompt from `prompt/config/generic/math.yaml`
   - Calls `_generate()` with 4 random seeds
4. **Generate orchestration** (`pipeline/generate.py`):
   - Creates CommandGroup with ServerScript + GenerationClientScript
   - Builds Pipeline with 4 jobs (one per seed)
   - Submits via `pipeline.run()`
5. **Execution** (`pipeline/utils/exp.py`):
   - Packages code, resolves containers
   - Creates SLURM/Docker job, submits via NeMo-Run
6. **Server starts** (`inference/server/serve_vllm.py`): Launches vLLM
7. **Generation** (`inference/generate.py`):
   - Loads dataset, constructs prompts
   - Calls `model.generate()` for each sample async
   - Writes to `output-rs{seed}.jsonl`
8. **Evaluation** (`evaluation/evaluator/math.py`):
   - Extracts answers, compares with ground truth
9. **Summarization** (`pipeline/summarize_results.py`):
   - Aggregates metrics across seeds
   - Saves to `metrics.json`, optionally logs to WandB

## Extension Points

### Adding New Model Backend

1. Implement `BaseModel` in `nemo_skills/inference/model/new_backend.py`
2. Register in `models` dict in `inference/model/__init__.py`
3. Optionally add server script in `inference/server/serve_new_backend.py`

### Adding New Benchmark

1. Create `nemo_skills/dataset/my_benchmark/__init__.py`:
   ```python
   DATASET_GROUP = "math"
   METRICS_TYPE = "math"
   GENERATION_ARGS = "++prompt_config=generic/math"
   ```
2. Add data files: `test.jsonl`, `validation.jsonl`
3. Implement evaluator in `evaluation/evaluator/` if custom logic needed
4. Add metrics in `evaluation/metrics/` if new metric type needed
5. Add to documentation with example command and expected results
6. Run `mkdocs serve` to verify documentation renders properly
7. Run GPU tests in CI (set "run GPU tests" label)
8. Consider adding to slurm tests for comprehensive validation

### Adding New Prompt Template

1. Create `nemo_skills/prompt/config/category/name.yaml`
2. Define `user`, `system`, `few_shot_examples` sections
3. Use as `++prompt_config=category/name`

### Adding New Training Backend

1. Implement integration in `nemo_skills/training/new_backend/`
2. Add pipeline command in `nemo_skills/pipeline/new_backend/`
3. Register in `nemo_skills/pipeline/cli.py`

### Custom Generation Logic

1. Implement `GenerationTask` class
2. Register in `GENERATION_MODULE_MAP` or use `--generation_module`

## Code Guidelines (from CONTRIBUTING.md)

### Don't Be Overly Defensive

This codebase prioritizes **fail-fast** over silent errors:

- **No `.get()` for expected keys**: Use `data[key_name]` instead of `data.get(key_name, "")` - let it fail with clear KeyError
- **No unnecessary exception catching**: Let code crash when unexpected things happen rather than silently misbehaving
- **No silently ignoring parameters**: If a user passes an unsupported parameter, the code should fail (use dataclasses or **kwargs)
- **Security not a concern**: Users have full system access; things like `subprocess.call(..., shell=True)` are fine
  - Exception: Code generated by LLMs should use the provided sandbox API

### Keep Code Elegant

When adding new features, try to keep the code simple and elegant:
- Reuse/extend existing functionality when possible
- Avoid checking too many conditions or complicated logic
- Write simpler code even if it sacrifices rare edge-cases
- Make the code self-explanatory rather than adding excessive comments
- Use simple types (dict, list, int, float, existing classes)
- Avoid complicated type unions/interfaces that quickly become outdated
- Follow existing naming conventions - if something is called X in one place, don't call it Y elsewhere

When in doubt, follow the Zen of Python: "Beautiful is better than ugly. Explicit is better than implicit. Simple is better than complex."

## Key Directories

```
nemo_skills/
├── pipeline/          # Pipeline orchestration and CLI
│   ├── cli.py        # Main ns CLI entry point
│   ├── generate.py   # Generation pipeline
│   ├── eval.py       # Evaluation pipeline
│   ├── utils/        # Pipeline utilities (declarative.py, scripts.py, etc.)
│   ├── megatron_lm/  # Megatron-LM training pipelines
│   ├── nemo_rl/      # NeMo-RL training pipelines (SFT, GRPO)
│   └── verl/         # VERL training pipelines (PPO)
├── inference/        # LLM inference and generation
│   ├── generate.py   # Main generation script
│   ├── factory.py    # Generation type registry
│   ├── model/        # Model provider implementations
│   └── server/       # Server startup scripts
├── evaluation/       # Evaluation and metrics
│   ├── evaluator/    # Benchmark evaluators
│   └── metrics/      # Metric implementations
├── dataset/          # 80+ benchmark datasets
├── training/         # Training pipelines (NeMo-RL, VERL)
├── prompt/           # Prompt templates
│   ├── config/       # YAML prompt configurations
│   ├── few_shot_examples/  # Few-shot example generators
│   └── code_tags/    # Code execution markers
└── code_execution/   # Code execution sandbox

cluster_configs/      # Execution environment configs (local/slurm)
tests/
├── gpu-tests/        # GPU-required tests
└── slurm-tests/      # Full cluster integration tests
recipes/              # Reproducible experiment recipes
docs/                 # Documentation (MkDocs)
```

## Common Pitfalls

- **Forgetting to run `ns setup`** before using cluster features
- **Not mounting paths correctly** in cluster configs - all paths in jobs must reference mounted locations
- **Using relative paths** instead of absolute paths in cluster jobs
- **Not signing off commits** - use `git commit -s` (pre-commit will catch this)
- **Adding unnecessary defensive code** - violates contribution guidelines (let code fail explicitly)
- **Forgetting code is packaged from git** - only committed files are available in containers/cluster
- **Not setting HF_HOME** in cluster config env_vars (required for HuggingFace models)
- **Expecting local packages in containers** - containers only have what's in their Dockerfile
- **Using container paths that aren't mounted** - will fail at runtime

## Tips

- Use `ns setup` to create cluster configs interactively
- Check `nemo_skills/dataset/` for all available benchmarks
- Use `ns <command> --help` to see available arguments and underlying script path
- Run `summarize_results` with `--cluster=slurm` if results are on cluster
- Use `expname` and `run_after` for job dependencies on SLURM
- For debugging: set `++log_level=debug` in generation commands
- Use `num_jobs` parameter to parallelize evaluation across multiple SLURM jobs
- Prefix paths with `/nemo_run/code/` to reference packaged code files
- Check `.done` files to see which jobs have completed
- The codebase uses **nemo_run** (from NVIDIA-NeMo/Run) for execution orchestration

## Key Files to Know

- `nemo_skills/pipeline/cli.py` - All CLI commands registered here
- `nemo_skills/inference/generate.py` - Core generation logic
- `nemo_skills/evaluation/evaluator/__init__.py` - Evaluator registry
- `nemo_skills/inference/model/__init__.py` - Model registry
- `pyproject.toml` - Package metadata, scripts (`ns` command), and tool configs
- `requirements/main.txt` - Production dependencies
- `requirements/common-dev.txt` - Development dependencies
