# NeMo-RL Integration with NeMo-Skills

This directory contains the integration between NeMo-RL (reinforcement learning framework) and NeMo-Skills (inference and prompting framework).

---

## 🚧 TODO: Production Readiness

This is a proof-of-concept. The following items must be addressed before adding to core pipelines, **sorted by impact** (most impactful first):

---

### 1. 🔴 Create `nemo_skills.pipeline` for NeMo-RL Training (High Priority - Usability)

**Goal:** Use NeMo-Skills declarative pipeline API to orchestrate proxy + NeMo-RL job.

**Reference Implementation:** See PR 1133 for script-based pipeline pattern (please upgrade the the run.Script version of pipelines)

**Current State:** The test script (`tests/gpu-tests/test_nemo_rl_integration.sh`) manually:
1. Starts NeMo-RL in background
2. Parses vLLM URL from logs
3. Starts proxy with discovered URL
4. Waits for completion

**Files to Create/Modify:**
- `nemo_skills/pipeline/nemo_rl/grpo_gym.py` - New pipeline command
- `nemo_skills/pipeline/utils/commands.py` - Add `nemo_rl_command()` and `skills_proxy_command()` helpers


**Benefits:**
- Proper SLURM job submission
- Hostname resolution via `hostname_ref()`
- No log parsing or environment variable hacks
- Consistent with existing NeMo-Skills pipelines

---

### 2. 🔴 Replace Hardcoded Timeouts with Configuration (High Priority - Scalability)

**Goal:** Remove all hardcoded timeouts and environment variables. Configuration should happen at pipeline creation time with continuous polling.

**Current Hardcoded Timeouts/Env Vars:**

| Location | Current Value | Issue |
|----------|---------------|-------|
| `generate.py:201` | `server_wait_timeout: int = 300` | Backend server wait |
| `test_nemo_rl_integration.sh:77` | `for i in {1..360}` | 12 min vLLM wait |
| `test_nemo_rl_integration.sh:113` | `for i in {1..60}` | 2 min proxy wait |

**Environment Variables to Remove:**
```python
# From skills_proxy.py lines 76-92
VLLM_URL_ENV_VARS = ["NEMO_RL_VLLM_URL", "NEMO_SKILLS_MODEL_SERVER_URL", "VLLM_BASE_URL"]
RAY_ADDRESS_ENV_VARS = ["RAY_ADDRESS", "RAY_HEAD_ADDRESS"]
NEMO_GYM_HEAD_SERVER_ENV_VAR = "NEMO_GYM_HEAD_SERVER_URL"
# Also: NEMO_RL_MODEL_NAME (used in test script line 102)
```

**Proposed Solution:**

1. **Pipeline-level configuration:** Pass vLLM URL, model name, and timeouts as explicit config
    * some of this might not be possible (e.g., VLLM URL isn't known until NeMo-RL starts)
    * If that's the case, there needs to be a known communication protocol to discover the VLLM during runtime
2. **Health check polling:** Replace fixed timeouts with health check endpoints + backoff
3. **Failure detection:** If a component crashes (e.g., proxy, or nemo-rl), make sure the whole job fails


**Files to Modify:**
- `nemo_skills/training/nemo_rl/utils/skills_proxy.py` - Add config class, remove env var discovery

---

### 3. 🔴 Fix NEMO_GYM_PATCH - JSON vs YAML Issue (High Priority - Correctness)

**Goal:** Remove the sed patch in the test script and fix the proxy to return proper format.

**Current Hack in `test_nemo_rl_integration.sh:52`:**
```bash
NEMO_GYM_PATCH='sed -i "s/return OmegaConf.to_yaml(get_global_config_dict())/import json; return json.dumps(OmegaConf.to_container(get_global_config_dict()))/" /opt/nemo_rl_venv/lib/python3.12/site-packages/nemo_gym/server_utils.py'
```

**Root Cause:** NeMo-Gym's `global_config_dict_yaml` endpoint returns YAML, but the client expects JSON.

**Current Proxy Implementation (`skills_proxy.py:990-1014`):**
```python
@app.get("/global_config_dict_yaml")
async def global_config_dict_yaml():
    """NeMo-Gym compatibility endpoint."""
    config = {...}
    return json.dumps(config)  # Returns JSON with JSON content-type
```

**Investigation Needed:**
1. Check what NeMo-Gym's `ServerClient.load_from_global_config()` actually expects
2. Verify if the issue is content-type, content format, or both
3. May need to return YAML string with proper content-type, or JSON with different endpoint name

**Files to Modify:**
- `nemo_skills/training/nemo_rl/utils/skills_proxy.py` - Fix `/global_config_dict_yaml` response format
- `tests/gpu-tests/test_nemo_rl_integration.sh` - Remove NEMO_GYM_PATCH

---

### 4. 🔴 Verify Data/Prompt/Template Flow (High Priority - Correctness)

**Goal:** Ensure JSONL files work identically to `generate` command, with proper prompt/template handling.

**Current Data Flow:**
1. JSONL loaded by `ns_data_processor_for_nemo_gym` in `start_grpo.py:226-287`
2. Prompt formatted via `get_prompt()` → `prompt.fill()` → `user_message`
3. Wrapped in `responses_create_params.input` for NeMo-Gym
4. Proxy extracts from `/run` request, creates `data_point["messages"]`
5. `generate.py:fill_prompt()` handles openai format (lines 620-656)

**Potential Issues:**
- Double-templating if prompt is applied twice
- Loss of metadata fields during transformations
- Inconsistent handling of system messages

**Verification Steps:**
1. Compare output of standalone `generate` command vs proxy-routed generation
2. Check that `expected_answer`, `problem`, etc. fields propagate correctly
3. Verify system message handling in both paths

**Files to Audit:**
- `nemo_skills/training/nemo_rl/start_grpo.py` - `ns_data_processor_for_nemo_gym()`
- `nemo_skills/inference/generate.py` - `fill_prompt()` method
- `nemo_skills/training/nemo_rl/utils/skills_proxy.py` - `/run` endpoint message extraction

---

### 5. 🔴 Use NeMo-Skills Evaluators for Gym Reward (High Priority - Core Feature)

**Goal:** Integrate NeMo-Skills evaluators (`MathEvaluator`, `CodeExecEvaluator`, etc.) into the proxy to compute rewards during RL training.

**Current State:** Proxy returns `reward: 0.0` (hardcoded placeholder):
```python
# skills_proxy.py:1166-1167
nemo_gym_response = {
    "reward": 0.0,  # Reward will be calculated by MathEnvironment or set to 0
```

**Proposed Integration:**

Use NeMo-Skills evaluators to compute rewards directly in the proxy. This should be configured with the `++eval_type` and done during post-processing on the Skills side. We may need an additional configuration option to specify how to compute a reward on the evaluation fields.

**Files to Reference:**
- `nemo_skills/evaluation/evaluator/__init__.py:62-68` - `EVALUATOR_CLASS_MAP`
- `nemo_skills/evaluation/evaluator/math.py` - `MathEvaluator.eval_single()`
- `nemo_skills/evaluation/evaluator/base.py:91` - `supports_single_eval()`

**Benefits:**
- Reuse existing evaluation logic
- Consistent with offline NeMo-Skills evaluation

---

### 6. 🟡 Separate NeMo-Gym startup script from the math one (Medium Priority - Architecture)

**Goal:** `start_grpo.py` should remain clean and unchanged. NeMo-Gym-specific logic should move to `start_grpo_gym.py`.

**Current State:** `start_grpo.py` contains significant NeMo-Gym-specific code:
- Conditional imports with `NEMO_GYM_AVAILABLE` guard (lines 46-61)
- `create_nemo_gym_data_processor()` function (lines 214-289)
- `setup_nemo_gym_environment()` function (lines 388-478)
- `get_nemo_gym_config()` and `get_nemo_gym_agent_name()` helpers (lines 481-522)
- `_add_inline_nemo_gym_configs()` helper (lines 525-636)
- Branching logic in `main()` based on `should_use_nemo_gym` (lines 727-737)

**Files to Modify:**
- `nemo_skills/training/nemo_rl/start_grpo.py` - Extract NeMo-Gym code
- `nemo_skills/training/nemo_rl/start_grpo_gym.py` - Create new file with extracted code

**Approach:**
1. Create `start_grpo_gym.py` that imports shared utilities from `start_grpo.py`
2. Move all NeMo-Gym-specific data processors, environment setup, and config helpers
3. `start_grpo_gym.py` can call `setup()` from `start_grpo.py` then configure NeMo-Gym environment
4. Remove NeMo-Gym import guards from `start_grpo.py` (they become required imports in `start_grpo_gym.py`)

---

### 7. 🟡 Minimize Proxy Footprint / Remove Dead Code (Medium Priority - Maintainability)

**Goal:** Strip `skills_proxy.py` to minimal required functionality.

**Current Size:** ~1278 lines - likely has significant cruft from iterative development.

**Areas to Audit:**

| Section | Lines | Keep? | Notes |
|---------|-------|-------|-------|
| vLLM discovery (env, ray, head) | 321-499 | Maybe | Replace with explicit config (TODO #2) |
| Pydantic models | 559-702 | Audit | May have unused request/response models |
| `/v1/completions` endpoint | 908-968 | Audit | May not be used - NeMo-RL uses chat completions |
| `/v1/responses` endpoint | 1197-1275 | Keep | Used by NeMo-Gym |
| `/run` endpoint | 1016-1195 | Keep | Core agent endpoint |
| Generation config discovery | 149-223 | Maybe | If using explicit config, remove |

**Approach:**
1. Add logging to each endpoint, run integration test, see what's actually called
2. Remove unused discovery methods if explicit config is adopted
3. Remove Pydantic models that aren't referenced
4. Document which endpoints are required for which integration patterns

---

### 8. 🟡 Refine Temperature/Top-p Handling (Medium Priority - Usability)

**Goal:** Proxy should work out-of-the-box. Temperature/top_p config should only be needed for non-proxy case.

**Current State:** Must set `++inference.temperature=-1 ++inference.top_p=-1` to use server defaults:
```python
# generate.py:74-78
# Temperature for sampling. 0.0 = greedy decoding.
# Set to -1.0 to use the server's configured default (useful for NeMo-RL integration)
temperature: float = 0.0
top_p: float = 0.95
```

**Problem:** `-1` is a magic value. Users shouldn't need to know this.

---

### 9. 🟡 Consolidate responses_api_model Interface (Medium Priority - Maintainability)

**Goal:** Align proxy's API with NeMo-Gym's `SimpleResponsesAPIModel` expectations.

**NeMo-Gym expects certain response formats.** The proxy implements:
- `/v1/chat/completions` - Standard OpenAI format
- `/v1/responses` - OpenAI Responses API format
- `/run` - Agent endpoint with verification response

**Areas to Verify:**
1. Token ID fields match `TokenIDLogProbMixin` expectations:
   - `prompt_token_ids`
   - `generation_token_ids`
   - `generation_log_probs`

2. Response structure matches `SimpleAgentVerifyResponse`:
   - `reward` field
   - `response.output[].content[].text` path

**Files to Cross-Reference:**
- `Gym/nemo_gym/responses_api_models/simple_responses_api_model.py`
- `RL/nemo_rl/environments/nemo_gym.py` - `_postprocess_nemo_gym_to_nemo_rl_result()`

---

### 10. 🟡 Validate Tool-Calling Support (Medium Priority - Extensibility)

**Goal:** Confirm tool-calling works through proxy and returns in NeMo-Gym expected format.

**Validation Steps:**
1. Create test with `tool_modules` configured
2. Verify tool calls are executed through the proxy
3. Check response includes tool call results in NeMo-Gym format

**Files to Reference:**
- `nemo_skills/mcp/servers/python_tool.py` - Example tool
- `nemo_skills/inference/generate.py:164-183` - Tool module configuration

---

### 11. 🟡 Verify Tokenization Flow (Medium Priority - Correctness)

**Goal:** Confirm token IDs are correct throughout the stack.

**Current Flow:**
1. Proxy sets `_request_logprobs=True`, `_request_return_tokens_as_token_ids=True`
2. `generate.py:768-774` adds `top_logprobs=1` and `extra_body={"return_tokens_as_token_ids": True}`
3. vLLM returns tokens in `"token_id:12345"` format
4. Proxy calls `/tokenize` for prompt token IDs
5. All three lists returned to NeMo-Gym

**Verification:**
1. Compare token IDs from proxy vs direct vLLM call
2. Verify tokenize endpoint uses same tokenizer as generation
3. Check edge cases: special tokens, BOS/EOS handling

**Files:**
- `nemo_skills/training/nemo_rl/utils/skills_proxy.py:226-277` - tokenize_messages, extract_token_ids_from_logprobs
- `nemo_skills/inference/model/base.py` - LLM response parsing

---

### 12. 🟡 Measure Proxy Performance Overhead (Medium Priority - Validation)

**Goal:** Verify minimal latency impact from proxy layer.

**Test Approach:**
1. Standalone vLLM benchmark (direct calls)
2. Same benchmark through proxy
3. Compare latency percentiles (p50, p95, p99)
4. Profile if overhead > 5%

**Potential Overhead Sources:**
- FastAPI request parsing
- Message format conversion
- Tokenize endpoint calls (async, but adds latency)
- Logprobs extraction

**Optimization Opportunities:**
- Batch tokenize calls
- Cache model name
- Use connection pooling for vLLM requests

---

### 13. 🟢 Use Gym Environment for Reward from Skills Generation (Lower Priority - Extensibility)

**Goal:** Allow using full NeMo-Gym Resources servers for reward computation, with NeMo-Skills providing the generation.

**Use Case:** When you want advanced reward computation (LLM-as-judge, multi-step verification) that's already implemented in NeMo-Gym's resources servers.

**Approach:**
1. Stand up NeMo-Gym resources server alongside proxy
2. Proxy generates text, passes to resources server for verification
3. Resources server returns reward

**Difference from TODO #5:**
- TODO #5: NeMo-Skills evaluators compute reward in proxy (simpler, self-contained)
- This TODO: External NeMo-Gym resources server computes reward (more complex, more flexible)

**Implementation Complexity:** Requires coordinating multiple services and may need NeMo-Gym config files.

---

### 14. 🟢 Clean Up generate.py Interface (Lower Priority - Maintainability)

**Goal:** Minimize changes to `generate.py` for proxy mode. Keep core generation logic unchanged.

**Current Proxy-Specific Additions:**
- `start_server` flag and validation (lines 198-201, 242-254)
- `_configure_server_for_proxy_mode()` method
- `_discover_model_name_from_vllm()` method
- `_request_*` parameter handling in `process_single_datapoint()` (lines 752-775)
- FastAPI server startup in `generate()` method

**Proposed Cleanup:**
1. Move `_configure_server_for_proxy_mode()` to `skills_proxy.py`
2. Create `ProxyGenerationTask` subclass that adds proxy-specific behavior
3. Keep `GenerationTask` focused on batch generation

---

### 15. 🟢 Remove NeMo-Gym Import Guards (Lower Priority - Mechanical)

**Goal:** After splitting to `start_grpo_gym.py`, remove conditional imports.

**Current State (`start_grpo.py:46-61`):**
```python
_NEMO_GYM_IMPORT_ERROR = None
try:
    from nemo_rl.environments.nemo_gym import (...)
    NEMO_GYM_AVAILABLE = True
except ImportError as e:
    NEMO_GYM_AVAILABLE = False
    ...
```

**After Split:**
- `start_grpo.py` - No NeMo-Gym imports at all
- `start_grpo_gym.py` - Direct imports, fail fast if not available

---

### 16. 🟢 Clean Up Superfluous Flags (Lower Priority - Usability)

**Goal:** Reduce required configuration. The thing should just work.

**Current Flags That May Be Unnecessary:**
```bash
# From test script
++inference.temperature=-1 \        # Should be auto-detected in proxy mode
++inference.top_p=-1 \              # Should be auto-detected in proxy mode
++prompt_format=openai              # Should be default for proxy mode
```

**Proposed Defaults for Proxy Mode:**
- `prompt_format=openai` when `start_server=True`
- Temperature/top_p passthrough by default when in proxy mode
- Auto-discover model name from vLLM if not specified

---

## Priority Summary

| Priority | Count | Items |
|----------|-------|-------|
| 🔴 High (Correctness & Design) | 5 | #1 Pipeline, #2 Timeouts, #3 YAML fix, #4 Data flow, #5 Evaluators for reward |
| 🟡 Medium (Refinement) | 7 | #6 Split files, #7 Cleanup, #8 Temp/top_p, #9 API, #10 Tools, #11 Tokenization, #12 Perf |
| 🟢 Lower (Mechanical) | 4 | #13 Gym Resources, #14 generate.py, #15 Guards, #16 Flags |

---

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


1. **NeMo-RL GRPO** prepares prompts and calls NemoGym for rollout collection
2. **NemoGym** sends requests to the NeMo-Skills proxy via `/run` endpoint
3. **NeMo-Skills proxy** applies prompt formatting, then forwards to vLLM
4. **vLLM** generates text with logprobs
5. **Proxy** calls `/tokenize` to get prompt token IDs
6. **Proxy** returns token IDs + logprobs to NemoGym for training

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

## Key Files

| File | Purpose |
|------|---------|
| `nemo_skills/training/nemo_rl/start_grpo.py` | GRPO training entry point with NeMo-Gym support |
| `nemo_skills/training/nemo_rl/utils/skills_proxy.py` | FastAPI proxy server factory and vLLM discovery |
| `nemo_skills/inference/generate.py` | GenerationTask with `start_server=True` mode |
| `nemo_skills/inference/model/base.py` | LLM interface with logprobs parsing |
| `tests/gpu-tests/test_nemo_rl_integration.sh` | Integration test script |

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

## Key Changes to `start_grpo.py`

The `start_grpo.py` script is the NeMo-Skills GRPO training entry point. It was modified to support both the standard `MathEnvironment` (in-process evaluation) and `NemoGym` (HTTP-based evaluation via proxy).

### NeMo-Gym Data Processor

When using NeMo-Gym, data points need additional fields for the HTTP-based rollout collection:

```python
def create_nemo_gym_data_processor(agent_name: str = "math_with_judge_simple_agent"):
    def ns_data_processor_for_nemo_gym(...) -> DatumSpec:
        # Standard prompt formatting
        user_message = prompt.fill(datum_dict, format_as_string=True)

        # NeMo-Gym specific fields (required by rollout_collection.py)
        extra_env_info["responses_create_params"] = {
            "input": [{"role": "user", "content": user_message}],
            "tools": [],
        }
        # Agent ref tells NeMo-Gym which agent server to call
        extra_env_info["agent_ref"] = {"name": agent_name}
        # Row index for result ordering after async processing
        extra_env_info["_rowidx"] = idx
```

**Key fields:**
- `responses_create_params.input`: Messages in OpenAI format for the `/run` request
- `agent_ref.name`: Must match a server key in NeMo-Gym config (e.g., `"math_with_judge_simple_agent"`)
- `_rowidx`: Index for reordering results after parallel async processing

### Environment Setup Ordering

A critical aspect is the **ordering** of setup calls:

```python
# 1. Apply NeMo-Gym config BEFORE setup() to configure vLLM correctly
if should_use_nemo_gym:
    setup_nemo_gym_config(config, tokenizer)  # Sets expose_http_server=True, async_engine=True

# 2. Setup data (uses appropriate data processor)
dataset, val_dataset = setup_data_only(tokenizer, config["data"], use_nemo_gym=should_use_nemo_gym)

# 3. Run NeMo-RL setup() to create policy_generation
(policy, policy_generation, ...) = setup(config, tokenizer, dataset, val_dataset)

# 4. NOW set up environment - needs policy_generation.dp_openai_server_base_urls
if should_use_nemo_gym:
    task_to_env = setup_nemo_gym_environment(policy_generation, nemo_gym_config)
```

**Why this order matters:**
- `setup_nemo_gym_config()` must run before `setup()` so vLLM is configured with HTTP exposure
- `setup_nemo_gym_environment()` must run after `setup()` because it needs `policy_generation.dp_openai_server_base_urls` to know where vLLM is listening

### Inline NeMo-Gym Server Configs

When a full NeMo-Gym YAML config isn't provided, the script adds inline server configs:

```python
def _add_inline_nemo_gym_configs(initial_global_config_dict, base_urls, model_name, agent_name, proxy_url):
    # Add model server config (policy_model) - points to vLLM or proxy
    initial_global_config_dict["policy_model"] = {
        "responses_api_models": {
            "vllm_model": {
                "host": server_host,
                "port": server_port,
                "model": model_name,
                "return_token_id_information": True,  # Critical for training!
            }
        }
    }

    # Add resources server config (math_with_judge)
    initial_global_config_dict["math_with_judge"] = {...}

    # Add agent server config pointing to proxy
    initial_global_config_dict[agent_name] = {...}
```

This creates the minimal configs needed for NeMo-Gym to function without external YAML files.

### Config Discovery

The script merges NeMo-Gym config from multiple sources:

```python
def get_nemo_gym_config(config: dict[str, Any]) -> dict[str, Any]:
    # CLI overrides go to config.nemo_gym
    top_level = config.get("nemo_gym", {})
    # YAML config might have it under config.env.nemo_gym
    env_level = config.get("env", {}).get("nemo_gym", {})
    # Merge: env_level takes precedence
    return {**top_level, **env_level}
```

This allows both `nemo_gym.policy_base_url=...` CLI overrides and YAML-based configuration.

## Critical Integration Points

### 1. Temperature/Top-p Passthrough

**Problem:** NeMo-RL's vLLM worker **asserts** that request parameters match its configured values:

```python
# In RL/nemo_rl/models/generation/vllm/vllm_worker_async.py
assert request.temperature == generation_config["temperature"]
assert request.top_p == generation_config["top_p"]
```

**Solution:** Pass through sampling parameters from incoming requests:

```python
# In skills_proxy.py /run endpoint
if "temperature" in responses_create_params:
    data_point["_request_temperature"] = responses_create_params["temperature"]
if "top_p" in responses_create_params:
    data_point["_request_top_p"] = responses_create_params["top_p"]

# In generate.py process_single_datapoint
if "_request_temperature" in data_point:
    generation_params["temperature"] = data_point["_request_temperature"]
```

**Config:** Set `inference.temperature=-1` and `inference.top_p=-1` to use server defaults.

### 2. Token ID Information for Training

**Problem:** NeMo-RL needs `prompt_token_ids`, `generation_token_ids`, and `generation_log_probs` for policy gradient training.

**Solution:** The proxy:
1. Requests logprobs with `top_logprobs=1` and `return_tokens_as_token_ids=True`
2. Extracts generation token IDs from logprobs (format: `"token_id:12345"`)
3. Calls vLLM's `/tokenize` endpoint to get prompt token IDs
4. Returns all three fields in the NemoGym response format

```python
# Request logprobs
data_point["_request_logprobs"] = True
data_point["_request_return_tokens_as_token_ids"] = True

# Extract from response
generation_token_ids = [int(tok.removeprefix("token_id:")) for tok in tokens]
prompt_token_ids = await tokenize_messages(base_url, model, messages)
```

### 3. skip_tokenizer_init Configuration

**Problem:** By default, NeMo-RL sets `skip_tokenizer_init=True` during training to save GPU memory. However, this breaks HTTP endpoints because vLLM can't:
- Apply chat templates to messages
- Serve the `/tokenize` endpoint

**Solution:** When using NeMo-Gym (HTTP-based), you MUST set:

```yaml
policy:
  generation:
    vllm_cfg:
      skip_tokenizer_init: false  # Required for HTTP endpoints!
      expose_http_server: true
      async_engine: true
```

**Why:** The tokenizer is needed for:
- Converting chat messages to tokens in `/v1/chat/completions`
- The `/tokenize` endpoint that returns prompt token IDs

### 4. Response Format for NemoGym

**Problem:** NemoGym expects a specific response format from the `/run` endpoint that matches `SimpleAgentVerifyResponse`.

**Solution:** The proxy returns:

```python
{
    "reward": 0.0,  # Placeholder - MathEnvironment calculates actual reward
    "response": {
        "id": "resp-...",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "..."}],
                # Token IDs for training
                "prompt_token_ids": [...],
                "generation_token_ids": [...],
                "generation_log_probs": [...],
            }
        ],
    },
    # Pass through original request fields
    "_rowidx": ...,
    "question": ...,
    ...
}
```

### 5. vLLM Server Discovery

**Problem:** The vLLM HTTP server URL is dynamically assigned and not known in advance.

**Solution:** Multiple discovery methods in order:
1. `NEMO_RL_VLLM_URL` environment variable
2. Ray named values (if connected to Ray cluster)
3. NeMo-Gym head server query

```python
vllm_config = discover_vllm_server()
# Returns VLLMServerConfig with base_url, host, port, model_name
```

## Tricky Aspects

### 1. Timing of Model Name Discovery

The vLLM server may not be ready when the proxy initializes. Model name discovery is deferred to `_discover_model_name_from_vllm()` which is called after `wait_for_server()`:

```python
def generate(self):
    if self.cfg.start_server:
        self.wait_for_server()  # Wait for vLLM to be ready
        if hasattr(self, "_vllm_base_url"):
            self._discover_model_name_from_vllm()  # Now safe to query
```

### 2. Token Format from vLLM

vLLM returns tokens in logprobs as `"token_id:12345"` when `return_tokens_as_token_ids=True`. Without this flag, it returns the actual token string which requires the tokenizer to convert back to IDs.

```python
# Parse token_id format
if token.startswith("token_id:"):
    token_id = int(token.removeprefix("token_id:"))
```

### 3. Chat Template Application

When using `prompt_format=openai`, messages pass through without re-templating. The proxy relies on vLLM to apply the chat template. This requires:
- vLLM to have its tokenizer initialized (`skip_tokenizer_init=false`)
- The model's tokenizer to have a chat template

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

### Minimal Proxy Configuration

```bash
python -m nemo_skills.inference.generate \
    ++start_server=True \
    ++inference.temperature=-1 \
    ++inference.top_p=-1 \
    ++generate_port=7000 \
    ++prompt_format=openai
```

### NeMo-RL Configuration for NeMo-Gym

```yaml
policy:
  generation:
    vllm_cfg:
      expose_http_server: true    # Enable HTTP server
      async_engine: true          # Required for HTTP
      skip_tokenizer_init: false  # Required for chat/tokenize

env:
  should_use_nemo_gym: true

nemo_gym:
  policy_base_url: "http://localhost:7000"  # Proxy URL
```

## Server Endpoints

When running `generate.py` with `++start_server=True`, the following endpoints are available:

| Endpoint | Format | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | OpenAI Chat | Standard OpenAI chat completions format |
| `/v1/completions` | OpenAI Text | Standard OpenAI text completions format |
| `/v1/models` | OpenAI | List available models |
| `/v1/responses` | OpenAI Responses | Responses API format for NeMo-Gym |
| `/run` | NeMo-Gym Agent | Agent endpoint with verification response |
| `/generate` | NeMo-Skills | Native NeMo-Skills data point format |
| `/health` | JSON | Health check endpoint |
| `/global_config_dict_yaml` | JSON | NeMo-Gym config discovery endpoint |

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

## Risks from Upstream Changes

### NeMo-RL Changes

| Risk | Impact | Mitigation |
|------|--------|------------|
| vLLM worker API changes | `/v1/chat/completions` format | Pin NeMo-RL version, test with new versions |
| Temperature/top_p assertion changes | Request rejection | Monitor for assertion changes |
| Token ID response format | Training data corruption | Verify `return_tokens_as_token_ids` behavior |
| `skip_tokenizer_init` default | HTTP endpoints break | Document requirement, add validation |
| `setup_nemo_gym_config()` API changes | vLLM not configured correctly | Monitor function signature |
| `policy_generation.dp_openai_server_base_urls` removal | Can't discover vLLM URLs | Critical - core integration point |
| `grpo_train()` / `setup()` signature changes | Training script breaks | Pin version, test with updates |

### NeMo-Gym Changes

| Risk | Impact | Mitigation |
|------|--------|------------|
| `/run` request format | Request parsing failures | Version pin, test with new versions |
| Response format expectations | Training failures | Monitor `_postprocess_nemo_gym_to_nemo_rl_result` |
| Token ID field names | Training data missing | Keep aligned with `TokenIDLogProbMixin` |
| `NemoGymConfig` TypedDict changes | Environment init fails | Monitor required fields |
| `agent_ref` format changes | Agent routing fails | Verify agent lookup logic |
| `responses_create_params.input` format | Rollout requests malformed | Keep aligned with OpenAI Responses API |
| Inline server config structure | Server discovery fails | Test with NeMo-Gym updates |

### vLLM Changes

| Risk | Impact | Mitigation |
|------|--------|------------|
| `/tokenize` endpoint changes | Prompt token IDs missing | Version pin, test tokenize response format |
| Logprobs format changes | Generation token IDs wrong | Monitor `choice.logprobs.content` structure |
| Chat completion response format | Parsing failures | Test with new vLLM versions |

### start_grpo.py Specific Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| `AllTaskProcessedDataset` API changes | Data loading fails | Test with NeMo-RL updates |
| `DatumSpec` field requirements | Training data invalid | Monitor expected fields |
| Environment registration changes | Env not found | Verify `ACTOR_ENVIRONMENT_REGISTRY` |
| Config resolution order | Wrong values applied | Keep get_nemo_gym_config() in sync |

## Testing

Run the integration test:

```bash
cd tests/gpu-tests
./test_nemo_rl_integration.sh
```

The test verifies:
1. NeMo-RL starts vLLM with HTTP exposed
2. Proxy discovers and connects to vLLM
3. NemoGym uses proxy for rollouts
4. Token IDs are correctly passed through
5. Training step completes

## Debugging

Enable debug logging:

```python
import logging
logging.getLogger("nemo_skills").setLevel(logging.DEBUG)
```

Key log messages:
- `Token ID information enabled. vLLM URL: ..., model: ...`
- `Requesting logprobs: top_logprobs=1, extra_body=...`
- `Extracted N tokens from logprobs, first few: ...`
- `Tokenize returned N prompt tokens`
- `Returning token info: prompt_tokens=X, gen_tokens=Y, logprobs=Z`

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
