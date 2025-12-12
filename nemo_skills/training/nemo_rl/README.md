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

### NeMo-Skills as a Proxy Server

The `nemo_skills/inference/generate.py` script can run as a server that:

1. **Receives requests** via OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`)
2. **Applies NeMo-Skills logic** (prompt formatting, code execution, few-shot examples)
3. **Forwards to the model server** (vLLM, TRT-LLM, OpenAI, etc.)
4. **Returns enriched responses** to NeMo-RL

This allows NeMo-RL/NeMo-Gym to use their existing model backends (like `SimpleResponsesAPIModel` or `VLLMModel`) while getting NeMo-Skills' rich prompting capabilities.

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
├── start_grpo.py               # GRPO training entry point
├── start_sft.py                # SFT training entry point
├── convert_dcp_to_hf.py        # FSDP checkpoint conversion
└── convert_megatron_to_hf.py   # Megatron checkpoint conversion
```

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
