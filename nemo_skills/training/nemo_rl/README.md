# NeMo-RL Integration with NeMo-Skills

This directory contains the integration between NeMo-RL (reinforcement learning framework) and NeMo-Skills (inference and prompting framework).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                    NeMo-RL Ray Cluster                          │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                              GRPO Training Loop                            │ │
│  │                                                                            │ │
│  │   ┌─────────────────┐                           ┌────────────────────────┐ │ │
│  │   │ policy_generation│                           │    Environment        │ │ │
│  │   │ (OpenAI backend) │───generations────────────►│ (reward computation)  │ │ │
│  │   └────────┬─────────┘                           └────────────────────────┘ │ │
│  └────────────│────────────────────────────────────────────────────────────────┘ │
│               │                                                                  │
│               │  OpenAI-compatible API calls                                     │
│               │  POST /v1/chat/completions                                       │
│               ▼                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                    NeMo-Skills generate.py (Proxy Server)                  │  │
│  │                                                                            │  │
│  │   /v1/chat/completions  ──►  Prompt Formatting  ──►  Backend Model Server  │  │
│  │   /v1/completions            Code Execution          (vLLM, TRT-LLM, etc)  │  │
│  │   /generate                  Evaluation                                    │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Understanding NeMo-RL's Generation Architecture

**Important:** NeMo-RL uses **Ray-based colocated generation** by default, NOT HTTP servers.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         NeMo-RL Ray Cluster                             │
│                                                                         │
│   ┌─────────────────┐      Ray Actor Calls       ┌──────────────────┐   │
│   │  GRPO Training  │ ◄────────────────────────► │ policy_generation │  │
│   │     Loop        │     (no HTTP involved)     │  (vLLM wrapped)   │  │
│   └─────────────────┘                            └──────────────────┘   │
│                                                                         │
│   Prompt formatting happens in ns_data_processor BEFORE generation     │
└─────────────────────────────────────────────────────────────────────────┘
```

- **`generation.colocated.enabled: true`** (default) - vLLM shares training GPUs via Ray actors
- **No HTTP server** - Communication is via Ray object references
- **Prompting already applied** - `ns_data_processor` in `start_grpo.py` calls `get_prompt()` at data prep time

### Two Integration Options

#### Option 1: Data-Level Integration (Already Built-In!)

The `ns_data_processor` in `start_grpo.py` already uses NeMo-Skills' `get_prompt()` function.
Prompt formatting is applied during data preparation, before generation.

```yaml
data:
  prompt:
    prompt_config: generic/math      # NeMo-Skills prompt template
    examples_type: gsm8k_text_with_code  # Few-shot examples
```

This is sufficient for most use cases where you need prompt formatting.

#### Option 2: HTTP Proxy with NeMo-Gym Integration

NeMo-RL can expose its vLLM generation workers as an HTTP server! This enables:

1. **NeMo-Gym agents** to call the policy model via HTTP
2. **NeMo-Skills proxy** to add prompt formatting, code execution, etc.

Enable with:
```yaml
env:
  should_use_nemo_gym: true  # Automatically enables expose_http_server

policy:
  generation:
    vllm_cfg:
      async_engine: true           # Required for HTTP exposure
      expose_http_server: true     # Exposes /v1/chat/completions
```

The NeMo-Skills proxy (`generate.py`) can then:
1. **Receive requests** via OpenAI-compatible endpoints
2. **Apply NeMo-Skills logic** (prompt formatting, code execution)
3. **Forward to NeMo-RL's vLLM HTTP endpoint**
4. **Return enriched responses** to NeMo-Gym agents

### Minimal Environment Implementation

Environments only need to implement **reward computation**. Generation is handled by NeMo-RL's `policy_generation` infrastructure. See `environments/math_environment.py` and `environments/weather_environment.py` for examples.

## Quick Start

### 1. Start the NeMo-Skills Proxy Server

```bash
python -m nemo_skills.inference.generate \
    ++start_server=True \
    ++generate_port=7000 \
    ++prompt_config=generic/math \
    ++server.server_type=vllm \
    ++server.host=localhost \
    ++server.port=5000
```

This starts a server that:
- Listens on port 7000
- Uses the `generic/math` prompt template
- Forwards to a vLLM server at `localhost:5000`

### Automatic vLLM Discovery

When running alongside NeMo-RL/NeMo-Gym, the proxy can automatically discover the vLLM server:

```python
from nemo_skills.training.nemo_rl.utils import discover_vllm_server

# Auto-discover vLLM server
config = discover_vllm_server()
if config:
    print(f"Found vLLM at {config.base_url} (via {config.source})")
    # Use config.host and config.port in your server configuration
```

#### How NeMo-RL/NeMo-Gym Discovery Works

When `expose_http_server: true` is enabled in NeMo-RL's vLLM config:

1. **`VllmAsyncGenerationWorker`** starts an HTTP server on a random port and stores `self.base_url`
2. **`VllmGeneration.dp_openai_server_base_urls`** collects URLs from all workers via `report_dp_openai_server_base_url()`
3. **`NemoGym`** receives these URLs and exposes them as `policy_base_url` in the global config
4. The global config is available via the head server at `/global_config_dict_yaml`

#### Discovery Methods (in order of precedence)

1. **Environment variables** (fastest, most explicit):
   - `NEMO_RL_VLLM_URL`
   - `NEMO_SKILLS_MODEL_SERVER_URL`
   - `VLLM_BASE_URL`

2. **Ray cluster** (if connected):
   - Checks for named actors or job metadata with vLLM URL

3. **NeMo-Gym head server** (recommended when using NeMo-Gym):
   - Queries `policy_base_url` from `/global_config_dict_yaml`
   - Set `NEMO_GYM_HEAD_SERVER_URL` environment variable

#### Example: Using with NeMo-Gym Head Server

```bash
# NeMo-Gym head server is running on port 11000
export NEMO_GYM_HEAD_SERVER_URL="http://localhost:11000"

# NeMo-Skills proxy will auto-discover vLLM URL from head server
python -m nemo_skills.inference.generate \
    ++start_server=True \
    ++generate_port=7000 \
    ++prompt_config=generic/math
```

#### Example: Using Environment Variable

```bash
# Manually set the vLLM URL (useful for testing or when head server isn't available)
export NEMO_RL_VLLM_URL="http://192.168.1.10:54321/v1"

python -m nemo_skills.inference.generate \
    ++start_server=True \
    ++generate_port=7000 \
    ++prompt_config=generic/math
```

### 2. Start GRPO Training

```bash
python -m nemo_skills.training.nemo_rl.start_grpo \
    --config configs/grpo.yaml \
    ++policy.model_name=/path/to/model \
    ++data.train_data_path=/path/to/train.jsonl \
    ++data.prompt.prompt_config=generic/math
```

## Configuration Examples

### Using NeMo-Skills Proxy with OpenAI Backend

Configure NeMo-RL to use the NeMo-Skills proxy as an OpenAI-compatible server:

```yaml
# grpo_with_nemo_skills_proxy.yaml

policy:
  model_name: /path/to/your/model

  generation:
    backend: "openai"
    openai_cfg:
      base_url: "http://localhost:7000/v1"  # NeMo-Skills proxy
      api_key: "EMPTY"  # Not needed for local server
      model: "nemo-skills"
    max_new_tokens: 2048
    temperature: 0.7
    top_p: 0.95

data:
  prompt:
    prompt_config: generic/math
    examples_type: null
  train_data_path: /path/to/train.jsonl
  val_data_path: /path/to/val.jsonl

env:
  math:
    env_cls: nemo_skills.training.nemo_rl.environments.math_environment.MathEnvironment
    num_workers: 8
```

### Direct vLLM Backend (No Proxy)

For cases where you don't need NeMo-Skills prompting:

```yaml
# grpo_direct_vllm.yaml

policy:
  model_name: /path/to/your/model

  generation:
    backend: "vllm"
    max_new_tokens: 2048
    temperature: 1.0
    vllm_cfg:
      tensor_parallel_size: 1
      gpu_memory_utilization: 0.6

data:
  prompt:
    prompt_config: generic/math
  train_data_path: /path/to/train.jsonl

env:
  math:
    env_cls: nemo_skills.training.nemo_rl.environments.math_environment.MathEnvironment
    num_workers: 8
```

### Custom Environment (Weather Prediction Example)

```yaml
# grpo_weather.yaml

policy:
  model_name: /path/to/your/model
  max_total_sequence_length: 2048

  generation:
    backend: "openai"
    openai_cfg:
      base_url: "http://localhost:7000/v1"
      api_key: "EMPTY"
      model: "nemo-skills"
    max_new_tokens: 512
    temperature: 0.7

data:
  prompt:
    prompt_config: weather/prediction  # Your custom prompt config
  train_data_path: /path/to/weather_train.jsonl
  val_data_path: /path/to/weather_val.jsonl

env:
  weather:
    env_cls: nemo_skills.training.nemo_rl.environments.weather_environment.WeatherEnvironment
    num_workers: 8
```

## Server Endpoints

When running `generate.py` with `++start_server=True`, the following endpoints are available:

| Endpoint | Format | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | OpenAI Chat | Standard OpenAI chat completions format |
| `/v1/completions` | OpenAI Text | Standard OpenAI text completions format |
| `/v1/models` | OpenAI | List available models |
| `/generate` | NeMo-Skills | Native NeMo-Skills data point format |
| `/health` | JSON | Health check endpoint |

### Example: Chat Completions Request

```bash
curl -X POST http://localhost:7000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemo-skills",
    "messages": [
      {"role": "user", "content": "Solve: What is 2 + 2?"}
    ],
    "temperature": 0.7
  }'
```

### Example: Native Generate Request

```bash
curl -X POST http://localhost:7000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is 2 + 2?",
    "expected_answer": "4"
  }'
```

## Creating Custom Environments

To create a custom environment, implement the `EnvironmentInterface`:

```python
from typing import Any, TypedDict
import ray
import torch
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


class MyEnvConfig(TypedDict):
    num_workers: int


class MyEnvironmentMetadata(TypedDict):
    expected_answer: str
    # Add other fields as needed


@ray.remote(max_restarts=-1, max_task_retries=-1)
class MyEnvironment(EnvironmentInterface):
    def __init__(self, cfg: MyEnvConfig):
        self.cfg = cfg
        # Initialize workers for parallel reward computation

    def shutdown(self) -> None:
        # Cleanup workers
        pass

    def step(
        self,
        message_log_batch: list[list[dict[str, str]]],
        metadata: list[MyEnvironmentMetadata],
    ) -> EnvironmentReturn:
        # Extract assistant responses
        responses = []
        for conversation in message_log_batch:
            assistant_msgs = [m["content"] for m in conversation if m["role"] == "assistant"]
            responses.append("".join(assistant_msgs))

        # Compute rewards
        rewards = self._compute_rewards(responses, metadata)

        return EnvironmentReturn(
            observations=[{"role": "environment", "content": "feedback"} for _ in responses],
            metadata=metadata,
            next_stop_strings=[None] * len(responses),
            rewards=torch.tensor(rewards),
            terminateds=torch.ones(len(rewards)),
            answers=responses,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        # Compute and return metrics
        metrics = {
            "accuracy": batch["rewards"].mean().item(),
        }
        return batch, metrics
```

Then configure it in your YAML:

```yaml
env:
  my_task:
    env_cls: path.to.my_environment.MyEnvironment
    num_workers: 8
```

## Directory Structure

```
nemo_skills/training/nemo_rl/
├── README.md                    # This file
├── __init__.py
├── configs/
│   ├── grpo.yaml               # Default GRPO configuration
│   ├── grpo-legacy-85eeb8d.yaml
│   ├── sft.yaml                # Default SFT configuration
│   └── sft-legacy-85eeb8d.yaml
├── environments/
│   ├── __init__.py
│   ├── math_environment.py     # Math reward computation
│   └── weather_environment.py  # Weather prediction example
├── prompts/
│   ├── cot.txt
│   └── math.txt
├── utils/
│   ├── __init__.py
│   └── skills_proxy.py         # OpenAI-compatible proxy server utilities
├── start_grpo.py               # GRPO training entry point
├── start_sft.py                # SFT training entry point
├── convert_dcp_to_hf.py        # FSDP checkpoint conversion
└── convert_megatron_to_hf.py   # Megatron checkpoint conversion
```

## Using the Proxy Utilities Directly

The `skills_proxy` module can be used independently to create OpenAI-compatible
proxy servers:

```python
from nemo_skills.training.nemo_rl.utils import create_skills_proxy_app, discover_vllm_server
import uvicorn

# Option 1: Use with a GenerationTask
from nemo_skills.inference.generate import GenerationTask, GenerateSolutionsConfig

cfg = GenerateSolutionsConfig(...)
task = GenerationTask(cfg)
app = create_skills_proxy_app(generation_task=task)
uvicorn.run(app, host="0.0.0.0", port=7000)

# Option 2: Use with a custom process function
async def my_process_fn(data_point: dict, all_data: list) -> dict:
    # Your custom generation logic
    return {"generation": "Hello!", "num_generated_tokens": 1}

app = create_skills_proxy_app(
    process_fn=my_process_fn,
    prompt_format="ns",
    generation_key="generation",
)
uvicorn.run(app, host="0.0.0.0", port=7000)
```

### vLLM Server Discovery

The module provides utilities for discovering vLLM servers:

```python
from nemo_skills.training.nemo_rl.utils import (
    discover_vllm_server,
    set_vllm_server_url,
    VLLMServerConfig,
)

# Discover vLLM server (tries env vars, Ray, head server)
config = discover_vllm_server()
if config:
    print(f"Found: {config.base_url}")
    print(f"Host: {config.host}, Port: {config.port}")
    print(f"Source: {config.source}")  # e.g., "env:NEMO_RL_VLLM_URL"

# Discover with explicit Ray address
config = discover_vllm_server(ray_address="ray://head:10001")

# Discover with explicit NeMo-Gym head server
config = discover_vllm_server(head_server_url="http://localhost:11000")

# Set the vLLM URL for other processes to discover
set_vllm_server_url("http://localhost:5000")
```

The discovery order is:
1. **Environment variables** (fastest, most explicit)
   - `NEMO_RL_VLLM_URL`
   - `NEMO_SKILLS_MODEL_SERVER_URL`
   - `VLLM_BASE_URL`
2. **Ray named values** (if connected to Ray cluster)
3. **NeMo-Gym head server** (queries `/global_config_dict_yaml`)

## Troubleshooting

### Connection Refused to Proxy Server

Ensure the NeMo-Skills proxy is running before starting training:

```bash
# Check if server is up
curl http://localhost:7000/health
```

### Prompt Not Being Applied

Verify your `prompt_config` is correct:

```bash
# Test with dry_run
python -m nemo_skills.inference.generate \
    ++start_server=True \
    ++prompt_config=generic/math \
    ++dry_run=True
```

### Environment Not Found

Ensure the environment class path is correct and the module is importable:

```python
# Test import
from nemo_skills.training.nemo_rl.environments.math_environment import MathEnvironment
```

## References

- [NeMo-RL Documentation](https://github.com/NVIDIA-NeMo/RL)
- [NeMo-Skills Documentation](../../../docs/)
- [GRPO Algorithm Paper](https://arxiv.org/abs/2402.03300)
