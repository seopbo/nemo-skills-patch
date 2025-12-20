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
NeMo-Skills Proxy Server Utilities for NeMo-Gym Integration.

This module provides:
1. A factory function to create proxy FastAPI applications for NeMo-Gym
2. Discovery utilities for finding vLLM servers in NeMo-RL environments

Usage:
    from nemo_skills.training.nemo_rl.utils.skills_proxy import (
        create_skills_proxy_app,
        discover_vllm_server,
    )

    # Discover the vLLM server address
    server_url = discover_vllm_server()

    # In your generation task
    app = create_skills_proxy_app(generation_task)
    uvicorn.run(app, host="0.0.0.0", port=7000)

The proxy server exposes:
    - /run: NeMo-Gym agent endpoint for rollout collection
    - /global_config_dict_yaml: NeMo-Gym configuration endpoint
    - /health: Health check
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Protocol

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


# =============================================================================
# vLLM Server Discovery
# =============================================================================
#
# When NeMo-RL exposes its vLLM generation workers as HTTP servers
# (via expose_http_server: true), this utility helps discover the server URL.
#
# Discovery methods (in order of precedence):
# 1. Environment variables
# 2. Ray named values (if connected to Ray cluster)
# 3. NeMo-Gym head server query (if head server address is known)


# Environment variables for vLLM server discovery
VLLM_URL_ENV_VARS = [
    "NEMO_RL_VLLM_URL",  # Set by NeMo-RL when expose_http_server=true
    "NEMO_SKILLS_MODEL_SERVER_URL",  # General-purpose model server URL
    "VLLM_BASE_URL",  # Common vLLM URL variable
]

# Environment variables for Ray cluster address
RAY_ADDRESS_ENV_VARS = [
    "RAY_ADDRESS",  # Standard Ray address variable
    "RAY_HEAD_ADDRESS",  # Alternative Ray head address
]

# Environment variable for NeMo-Gym head server
NEMO_GYM_HEAD_SERVER_ENV_VAR = "NEMO_GYM_HEAD_SERVER_URL"

# Ray named value key for vLLM server URL
RAY_VLLM_URL_KEY = "nemo_rl_vllm_http_url"


@dataclass
class VLLMGenerationConfig:
    """Generation configuration from a vLLM server."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None


@dataclass
class VLLMServerConfig:
    """Configuration for a discovered vLLM server."""

    base_url: str  # Full base URL, e.g., "http://localhost:5000"
    host: str
    port: int
    source: str  # How the server was discovered
    model_name: str | None = None  # The model name served by vLLM
    generation_config: VLLMGenerationConfig | None = None  # Generation parameters from server


def get_vllm_model_name(base_url: str, timeout: float = 5.0) -> str | None:
    """Query the vLLM server to get the model name being served.

    Args:
        base_url: The vLLM server base URL (e.g., "http://localhost:5000/v1")
        timeout: Request timeout in seconds

    Returns:
        The model name if found, None otherwise
    """
    try:
        # Remove /v1 suffix if present, then add /v1/models
        url = base_url.rstrip("/")
        if url.endswith("/v1"):
            models_url = f"{url}/models"
        else:
            models_url = f"{url}/v1/models"

        response = requests.get(models_url, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                model_name = data["data"][0].get("id")
                if model_name:
                    LOG.info(f"Discovered model name from vLLM: {model_name}")
                    return model_name
    except Exception as e:
        LOG.debug(f"Failed to get model name from vLLM: {e}")

    return None


def get_vllm_generation_config(
    head_server_url: str | None = None,
    timeout: float = 5.0,
) -> VLLMGenerationConfig | None:
    """Query the NeMo-Gym head server for generation config.

    The NeMo-RL vLLM server expects requests to have matching temperature/top_p values.
    This function discovers those values from the head server's global config.

    Args:
        head_server_url: Optional NeMo-Gym head server URL (e.g., "http://localhost:11000").
                        If None, will try to get from NEMO_GYM_HEAD_SERVER_URL.
        timeout: Request timeout in seconds

    Returns:
        VLLMGenerationConfig if found, None otherwise
    """
    # Get head server URL from environment if not provided
    if not head_server_url:
        head_server_url = os.environ.get(NEMO_GYM_HEAD_SERVER_ENV_VAR)

    if not head_server_url:
        LOG.debug("No NeMo-Gym head server URL available for generation config discovery")
        return None

    try:
        # Query the head server for global config
        response = requests.get(
            f"{head_server_url.rstrip('/')}/global_config_dict_yaml",
            timeout=timeout,
        )
        response.raise_for_status()

        # Parse the config (it's JSON-encoded YAML)
        config_yaml = response.content.decode()
        config = json.loads(config_yaml)

        # Look for generation parameters in the config
        # NeMo-RL may store these in various locations
        gen_config = VLLMGenerationConfig()

        # Check for sampling parameters at various locations
        for key_path in [
            "generation",
            "policy_generation",
            "model_config",
            "sampling",
            "responses_api_models",
        ]:
            if key_path in config:
                sub_config = config[key_path]
                if isinstance(sub_config, dict):
                    if "temperature" in sub_config:
                        gen_config.temperature = sub_config["temperature"]
                    if "top_p" in sub_config:
                        gen_config.top_p = sub_config["top_p"]
                    if "top_k" in sub_config:
                        gen_config.top_k = sub_config["top_k"]
                    if "max_tokens" in sub_config:
                        gen_config.max_tokens = sub_config["max_tokens"]

        # Also check top-level keys
        if "temperature" in config:
            gen_config.temperature = config["temperature"]
        if "top_p" in config:
            gen_config.top_p = config["top_p"]

        if gen_config.temperature is not None or gen_config.top_p is not None:
            LOG.info(f"Discovered generation config from head server: {gen_config}")
            return gen_config

    except Exception as e:
        LOG.debug(f"Failed to get generation config from head server: {e}")

    return None


async def tokenize_messages(
    base_url: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    timeout: float = 30.0,
) -> list[int] | None:
    """Call the vLLM /tokenize endpoint to get prompt token IDs.

    This is needed for NeMo-RL training which requires the exact token IDs
    used to generate the prompt.

    Args:
        base_url: The vLLM server base URL (e.g., "http://localhost:5000/v1")
        model: The model name
        messages: List of chat messages
        tools: Optional list of tools
        timeout: Request timeout in seconds

    Returns:
        List of token IDs if successful, None otherwise
    """
    import aiohttp

    try:
        # The tokenize endpoint is at /tokenize (not /v1/tokenize)
        url = base_url.rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]  # Remove /v1
        tokenize_url = f"{url}/tokenize"

        body = {
            "model": model,
            "messages": messages,
        }
        if tools:
            body["tools"] = tools

        async with aiohttp.ClientSession() as session:
            async with session.post(tokenize_url, json=body, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                if response.status == 200:
                    data = await response.json()
                    tokens = data.get("tokens", [])
                    LOG.debug(f"Tokenize returned {len(tokens)} tokens")
                    return tokens
                else:
                    LOG.warning(f"Tokenize endpoint returned status {response.status}")
                    return None
    except Exception as e:
        LOG.warning(f"Failed to call tokenize endpoint: {e}")
        return None


def extract_token_ids_from_logprobs(logprobs_content: list[dict]) -> tuple[list[int], list[float]]:
    """Extract generation token IDs and log probs from logprobs content.

    vLLM returns tokens in logprobs content either as:
    - Direct token strings (need tokenizer to convert)
    - Token IDs prefixed with "token_id:" when return_tokens_as_token_ids=True

    Args:
        logprobs_content: The logprobs.content from a chat completion response

    Returns:
        Tuple of (generation_token_ids, generation_log_probs)
    """
    generation_token_ids = []
    generation_log_probs = []

    for entry in logprobs_content:
        # Get the log probability
        logprob = entry.get("logprob", 0.0)
        generation_log_probs.append(logprob)

        # Get the token - may be "token_id:12345" format or actual token string
        token = entry.get("token", "")
        if isinstance(token, str) and token.startswith("token_id:"):
            # vLLM with return_tokens_as_token_ids=True
            try:
                token_id = int(token.removeprefix("token_id:"))
                generation_token_ids.append(token_id)
            except ValueError:
                LOG.warning(f"Failed to parse token_id from: {token}")
                generation_token_ids.append(0)
        elif isinstance(token, int):
            generation_token_ids.append(token)
        else:
            # Token is a string - we'd need the tokenizer to get the ID
            # For now, use -1 as a placeholder
            LOG.debug(f"Token is string, not ID: {token[:20] if token else 'empty'}...")
            generation_token_ids.append(-1)

    return generation_token_ids, generation_log_probs


def discover_vllm_server(
    ray_address: str | None = None,
    head_server_url: str | None = None,
    timeout: float = 5.0,
    skip_model_discovery: bool = False,
) -> VLLMServerConfig | None:
    """
    Discover the vLLM HTTP server address using multiple methods.

    This function attempts to find the vLLM server that NeMo-RL exposes when
    running with `expose_http_server: true`. It tries the following methods
    in order:

    1. **Environment variables**: Checks NEMO_RL_VLLM_URL, NEMO_SKILLS_MODEL_SERVER_URL,
       and VLLM_BASE_URL for a configured URL.

    2. **Ray named values**: If connected to a Ray cluster, checks for the
       server URL stored under a known key.

    3. **NeMo-Gym head server**: If the head server URL is known, queries it
       for the model server configuration.

    Args:
        ray_address: Optional Ray cluster address. If None, will try to get from
                    environment or connect to an existing cluster.
        head_server_url: Optional NeMo-Gym head server URL (e.g., "http://localhost:11000").
                        If None, will try to get from NEMO_GYM_HEAD_SERVER_URL.
        timeout: Timeout for HTTP requests in seconds.
        skip_model_discovery: If True, skip querying for model name and generation config.
                             Useful when the server might not be ready yet.

    Returns:
        VLLMServerConfig if found, None otherwise.

    Example:
        # Simple discovery
        config = discover_vllm_server()
        if config:
            print(f"Found vLLM server at {config.base_url} (via {config.source})")

        # With explicit Ray address
        config = discover_vllm_server(ray_address="ray://head-node:10001")

        # With explicit head server URL
        config = discover_vllm_server(head_server_url="http://localhost:11000")

        # URL discovery only (when server might not be ready)
        config = discover_vllm_server(skip_model_discovery=True)
    """
    # Method 1: Environment variables
    config = _discover_from_env()
    if config:
        if not skip_model_discovery:
            config.model_name = get_vllm_model_name(config.base_url, timeout)
            config.generation_config = get_vllm_generation_config(head_server_url, timeout)
        return config

    # Method 2: Ray named values
    config = _discover_from_ray(ray_address)
    if config:
        if not skip_model_discovery:
            config.model_name = get_vllm_model_name(config.base_url, timeout)
            config.generation_config = get_vllm_generation_config(head_server_url, timeout)
        return config

    # Method 3: NeMo-Gym head server
    config = _discover_from_head_server(head_server_url, timeout)
    if config:
        if not skip_model_discovery:
            config.model_name = get_vllm_model_name(config.base_url, timeout)
            config.generation_config = get_vllm_generation_config(head_server_url, timeout)
        return config

    return None


def _parse_url(url: str) -> tuple[str, int] | None:
    """Parse a URL into (host, port)."""
    try:
        # Handle URLs with and without scheme
        if "://" not in url:
            url = f"http://{url}"

        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5000  # Default vLLM port
        return host, port
    except Exception:
        return None


def _discover_from_env() -> VLLMServerConfig | None:
    """Try to discover vLLM server from environment variables."""
    for env_var in VLLM_URL_ENV_VARS:
        url = os.environ.get(env_var)
        if url:
            parsed = _parse_url(url)
            if parsed:
                host, port = parsed
                base_url = f"http://{host}:{port}"
                LOG.info(f"Discovered vLLM server from {env_var}: {base_url}")
                return VLLMServerConfig(
                    base_url=base_url,
                    host=host,
                    port=port,
                    source=f"env:{env_var}",
                )
    return None


def _discover_from_ray(ray_address: str | None = None) -> VLLMServerConfig | None:
    """Try to discover vLLM server from Ray named values."""
    try:
        import ray
    except ImportError:
        LOG.debug("Ray not available, skipping Ray discovery")
        return None

    # Try to connect to Ray if not already connected
    try:
        if not ray.is_initialized():
            # Try to get Ray address from environment or use provided
            address = ray_address
            if not address:
                for env_var in RAY_ADDRESS_ENV_VARS:
                    address = os.environ.get(env_var)
                    if address:
                        break

            if address:
                ray.init(address=address, ignore_reinit_error=True)
            else:
                # Try to connect to local Ray cluster
                ray.init(address="auto", ignore_reinit_error=True)

        # Try to get the vLLM URL from Ray named values
        try:
            url = ray.get_actor(RAY_VLLM_URL_KEY).get_url.remote()
            url = ray.get(url, timeout=5)
            if url:
                parsed = _parse_url(url)
                if parsed:
                    host, port = parsed
                    base_url = f"http://{host}:{port}"
                    LOG.info(f"Discovered vLLM server from Ray: {base_url}")
                    return VLLMServerConfig(
                        base_url=base_url,
                        host=host,
                        port=port,
                        source="ray:named_actor",
                    )
        except Exception as e:
            LOG.debug(f"Could not get vLLM URL from Ray named actor: {e}")

        # Alternative: Check Ray runtime context for vLLM worker info
        try:
            ctx = ray.get_runtime_context()
            # Check if we have vLLM HTTP info in job config
            job_config = ctx.get_job_config()
            if job_config and hasattr(job_config, "metadata"):
                vllm_url = job_config.metadata.get("vllm_http_url")
                if vllm_url:
                    parsed = _parse_url(vllm_url)
                    if parsed:
                        host, port = parsed
                        base_url = f"http://{host}:{port}"
                        LOG.info(f"Discovered vLLM server from Ray job metadata: {base_url}")
                        return VLLMServerConfig(
                            base_url=base_url,
                            host=host,
                            port=port,
                            source="ray:job_metadata",
                        )
        except Exception as e:
            LOG.debug(f"Could not get vLLM URL from Ray context: {e}")

    except Exception as e:
        LOG.debug(f"Ray discovery failed: {e}")

    return None


def _discover_from_head_server(head_server_url: str | None = None, timeout: float = 5.0) -> VLLMServerConfig | None:
    """Try to discover vLLM server from NeMo-Gym head server.

    NeMo-Gym stores the vLLM server URL(s) in its global config as `policy_base_url`.
    This is set by NeMo-RL when it passes `VllmGeneration.dp_openai_server_base_urls`
    to the NemoGym config.
    """
    try:
        import requests
    except ImportError:
        LOG.debug("requests not available, skipping head server discovery")
        return None

    # Get head server URL from environment if not provided
    if not head_server_url:
        head_server_url = os.environ.get(NEMO_GYM_HEAD_SERVER_ENV_VAR)

    if not head_server_url:
        LOG.debug("No NeMo-Gym head server URL available")
        return None

    try:
        # Query the head server for global config
        response = requests.get(
            f"{head_server_url.rstrip('/')}/global_config_dict_yaml",
            timeout=timeout,
        )
        response.raise_for_status()

        # Parse the config (it's JSON-encoded YAML)
        config_yaml = response.content.decode()
        config = json.loads(config_yaml)

        # NeMo-Gym stores the vLLM URL as "policy_base_url"
        # This comes from VllmGeneration.dp_openai_server_base_urls
        policy_base_url = config.get("policy_base_url")
        if policy_base_url:
            # Can be a single URL or a list of URLs (one per DP rank)
            if isinstance(policy_base_url, list):
                # Take the first one - they're all equivalent for our purposes
                url = policy_base_url[0]
            else:
                url = policy_base_url

            parsed = _parse_url(url)
            if parsed:
                host, port = parsed
                # The URL already includes /v1 from NeMo-RL
                base_url = url if url.startswith("http") else f"http://{url}"
                LOG.info(f"Discovered vLLM server from head server (policy_base_url): {base_url}")
                return VLLMServerConfig(
                    base_url=base_url,
                    host=host,
                    port=port,
                    source="head_server:policy_base_url",
                )

        # Fallback: Look for other server configurations
        for key, value in config.items():
            if isinstance(value, dict) and "host" in value and "port" in value:
                # Check if this looks like a model/policy server
                if any(term in key.lower() for term in ["model", "policy", "vllm", "generation"]):
                    host = value["host"]
                    port = value["port"]
                    base_url = f"http://{host}:{port}"
                    LOG.info(f"Discovered vLLM server from head server ({key}): {base_url}")
                    return VLLMServerConfig(
                        base_url=base_url,
                        host=host,
                        port=port,
                        source=f"head_server:{key}",
                    )

    except Exception as e:
        LOG.debug(f"Head server discovery failed: {e}")

    return None


def set_vllm_server_url(url: str, env_var: str = "NEMO_RL_VLLM_URL") -> None:
    """
    Set the vLLM server URL in the environment for discovery.

    This is useful when starting NeMo-RL training to make the vLLM URL
    discoverable by NeMo-Skills proxy servers.

    Args:
        url: The vLLM server URL (e.g., "http://localhost:5000")
        env_var: The environment variable to set (default: NEMO_RL_VLLM_URL)

    Example:
        # In NeMo-RL training script
        vllm_port = start_vllm_http_server()
        set_vllm_server_url(f"http://localhost:{vllm_port}")
    """
    os.environ[env_var] = url
    LOG.info(f"Set {env_var}={url}")


# =============================================================================
# Generation Task Protocol
# =============================================================================


class GenerationTaskProtocol(Protocol):
    """Protocol defining the interface required from a generation task."""

    @property
    def cfg(self) -> Any:
        """Configuration object with generation_key and other settings."""
        ...

    async def _process_single_datapoint_core(self, data_point: dict[str, Any], all_data: list) -> dict[str, Any]:
        """Process a single data point through the NeMo-Skills pipeline."""
        ...


# Type alias for the process function
ProcessFn = Callable[[dict[str, Any], list], Coroutine[Any, Any, dict[str, Any]]]


# =============================================================================
# FastAPI App Factory
# =============================================================================


def create_skills_proxy_app(
    generation_task: GenerationTaskProtocol | None = None,
    process_fn: ProcessFn | None = None,
    generation_key: str = "generation",
    title: str = "NeMo Skills Generation Server",
    return_token_id_information: bool = True,
    vllm_base_url: str | None = None,
    model_name: str | None = None,
) -> FastAPI:
    """
    Create a FastAPI application for NeMo-Gym integration.

    This proxy passes messages through to the NeMo-Skills generation pipeline,
    which can handle code execution loops, tool calling, and sandbox integration.

    Endpoints:
        - /run: NeMo-Gym agent endpoint for rollout collection
        - /global_config_dict_yaml: NeMo-Gym configuration endpoint
        - /health: Health check

    Args:
        generation_task: A GenerationTask instance (has cfg and _process_single_datapoint_core).
                        Either this or process_fn must be provided.
        process_fn: Alternative to generation_task - a coroutine function that processes
                   data points. Signature: async (data_point: dict, all_data: list) -> dict
        generation_key: The key in the output dict that contains the generation.
                       Only used if process_fn is provided.
        title: Title for the FastAPI application.
        return_token_id_information: If True (default), request logprobs from vLLM and
                                    return prompt_token_ids, generation_token_ids, and
                                    generation_log_probs in responses. Required for NeMo-RL training.
        vllm_base_url: The vLLM server base URL for tokenize calls. If None, will be
                      auto-discovered from generation_task or environment.
        model_name: The model name for tokenize calls. If None, will be auto-discovered.

    Returns:
        FastAPI application with NeMo-Gym compatible endpoints.

    Example:
        # Using with GenerationTask
        task = GenerationTask(cfg)
        app = create_skills_proxy_app(generation_task=task)
        uvicorn.run(app, host="0.0.0.0", port=7000)

        # Using with custom process function
        async def my_process(data_point, all_data):
            return {"generation": "Hello!", "num_generated_tokens": 1}

        app = create_skills_proxy_app(process_fn=my_process)
    """
    if generation_task is None and process_fn is None:
        raise ValueError("Either generation_task or process_fn must be provided")

    # Get configuration from task or use defaults
    if generation_task is not None:
        _generation_key = getattr(generation_task.cfg, "generation_key", "generation")
        _process_fn = generation_task._process_single_datapoint_core
    else:
        _generation_key = generation_key
        _process_fn = process_fn

    # Get vLLM base URL and model name for tokenize calls
    _vllm_base_url = vllm_base_url
    _model_name = model_name
    _return_token_id_information = return_token_id_information

    if _vllm_base_url is None and generation_task is not None:
        # Try to get from generation_task
        if hasattr(generation_task, "_vllm_base_url"):
            _vllm_base_url = generation_task._vllm_base_url
        elif hasattr(generation_task.cfg, "server"):
            server_cfg = generation_task.cfg.server
            host = server_cfg.get("host")
            port = server_cfg.get("port")
            if host and port:
                _vllm_base_url = f"http://{host}:{port}/v1"

    if _model_name is None and generation_task is not None:
        if hasattr(generation_task.cfg, "server"):
            _model_name = generation_task.cfg.server.get("model")

    # Also check environment
    if _model_name is None:
        _model_name = os.environ.get("NEMO_RL_MODEL_NAME")

    if _return_token_id_information:
        LOG.info(f"Token ID information enabled. vLLM URL: {_vllm_base_url}, model: {_model_name}")

    app = FastAPI(
        title=title,
        description="Proxy server that adds NeMo-Skills prompting logic to model generations. "
        "Compatible with OpenAI API for seamless integration with NeMo-Gym/NeMo-RL.",
    )

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.get("/global_config_dict_yaml")
    async def global_config_dict_yaml():
        """NeMo-Gym compatibility endpoint.

        NeMo-Gym's ServerClient.load_from_global_config() calls this endpoint
        to get the global configuration. It expects a JSON-encoded config dict.

        For the NeMo-Skills proxy, we return a minimal config that tells NeMo-Gym
        to use this proxy as the model server.
        """
        # Return a minimal config that NeMo-Gym can use
        # The key fields are:
        # - responses_api_models: tells NeMo-Gym how to create API models
        # - Any other fields NeMo-Gym needs for rollouts
        config = {
            "responses_api_models": {
                "nemo-skills": {
                    "type": "openai",
                    "model": "nemo-skills",
                }
            },
            # Tell NeMo-Gym this is the policy server
            "is_skills_proxy": True,
        }
        return json.dumps(config)

    @app.post("/run")
    async def run_agent(request_body: dict):
        """NeMo-Gym agent /run endpoint for rollout collection.

        This endpoint handles NeMo-Gym's agent rollout requests. It acts as a
        minimal agent that:
        1. Extracts the prompt from responses_create_params
        2. Calls the generation pipeline
        3. Returns a minimal verification response with reward=0.0

        For proper reward calculation, use MathEnvironment or configure
        a full NeMo-Gym resources server.

        Request format (from NeMo-Gym rollout_collection.py):
            {
                "responses_create_params": {
                    "input": [{"role": "user", "content": "..."}],
                    "tools": []
                },
                "agent_ref": {"name": "..."},
                "_rowidx": 0,
                # ... other fields from data
            }

        Response format (SimpleAgentVerifyResponse):
            {
                "reward": 0.0,
                "response": {
                    "output": [...],
                    ...
                },
                # ... other fields
            }
        """
        try:
            # Extract the prompt from responses_create_params
            responses_create_params = request_body.get("responses_create_params", {})
            input_messages = responses_create_params.get("input", [])

            # Convert to our message format
            messages = []
            for msg in input_messages:
                if isinstance(msg, dict):
                    messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
                elif isinstance(msg, str):
                    messages.append({"role": "user", "content": msg})

            if not messages:
                # Fallback: try to get question from the request body directly
                question = request_body.get("question", "")
                if question:
                    messages = [{"role": "user", "content": question}]
                else:
                    messages = [{"role": "user", "content": ""}]

            # Process through NeMo-Skills pipeline
            data_point = {"messages": messages}
            # Pass through extra fields that might be needed (like expected_answer)
            for key in ["question", "expected_answer", "problem"]:
                if key in request_body:
                    data_point[key] = request_body[key]

            # Pass through sampling parameters from responses_create_params
            # This is critical for NeMo-RL integration where vLLM asserts temperature matches
            if "temperature" in responses_create_params:
                data_point["_request_temperature"] = responses_create_params["temperature"]
            if "top_p" in responses_create_params:
                data_point["_request_top_p"] = responses_create_params["top_p"]
            if "max_tokens" in responses_create_params:
                data_point["_request_max_tokens"] = responses_create_params["max_tokens"]
            if "max_output_tokens" in responses_create_params:
                data_point["_request_max_tokens"] = responses_create_params["max_output_tokens"]
            if "stop" in responses_create_params:
                data_point["_request_stop"] = responses_create_params["stop"]

            # Request logprobs for token ID extraction if enabled
            if _return_token_id_information:
                data_point["_request_logprobs"] = True
                data_point["_request_return_tokens_as_token_ids"] = True

            LOG.debug(
                f"/run request params: temperature={data_point.get('_request_temperature')}, "
                f"top_p={data_point.get('_request_top_p')}, "
                f"max_tokens={data_point.get('_request_max_tokens')}, "
                f"logprobs={data_point.get('_request_logprobs')}"
            )

            output = await _process_fn(data_point, [])

            generation_text = output.get(_generation_key, output.get("generation", ""))

            # Extract token ID information if enabled
            prompt_token_ids = []
            generation_token_ids = []
            generation_log_probs = []

            if _return_token_id_information:
                # Extract generation token IDs from logprobs in the output
                LOG.info(
                    f"Token ID extraction: output keys={list(output.keys())}, "
                    f"has_logprobs={'logprobs' in output}, has_tokens={'tokens' in output}"
                )
                if "logprobs" in output and "tokens" in output:
                    # Logprobs were returned - extract token IDs
                    tokens = output.get("tokens", [])
                    LOG.info(f"Extracting from {len(tokens)} tokens")
                    logprobs = output.get("logprobs", [])

                    for i, token in enumerate(tokens):
                        # Token may be "token_id:12345" format or actual token string
                        if isinstance(token, str) and token.startswith("token_id:"):
                            try:
                                token_id = int(token.removeprefix("token_id:"))
                                generation_token_ids.append(token_id)
                            except ValueError:
                                generation_token_ids.append(0)
                        elif isinstance(token, int):
                            generation_token_ids.append(token)
                        else:
                            # Token is a string - use -1 as placeholder
                            generation_token_ids.append(-1)

                        if i < len(logprobs):
                            generation_log_probs.append(logprobs[i])
                        else:
                            generation_log_probs.append(0.0)

                # Get prompt token IDs via /tokenize endpoint
                if _vllm_base_url and _model_name:
                    LOG.info(f"Calling tokenize: url={_vllm_base_url}, model={_model_name}")
                    prompt_token_ids = (
                        await tokenize_messages(
                            base_url=_vllm_base_url,
                            model=_model_name,
                            messages=messages,
                            tools=responses_create_params.get("tools"),
                        )
                        or []
                    )
                    LOG.info(f"Tokenize returned {len(prompt_token_ids)} prompt tokens")
                else:
                    LOG.warning(f"Cannot tokenize: vllm_url={_vllm_base_url}, model={_model_name}")

            # Build NeMo-Gym compatible response
            # This matches the format expected by NemoGym._postprocess_nemo_gym_to_nemo_rl_result
            LOG.info(
                f"Returning token info: prompt_tokens={len(prompt_token_ids)}, "
                f"gen_tokens={len(generation_token_ids)}, logprobs={len(generation_log_probs)}"
            )

            nemo_gym_response = {
                "reward": 0.0,  # Reward will be calculated by MathEnvironment or set to 0
                "response": {
                    "id": f"resp-{uuid.uuid4().hex[:8]}",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": generation_text,
                                }
                            ],
                            # Token IDs for NeMo-RL training
                            "prompt_token_ids": prompt_token_ids,
                            "generation_token_ids": generation_token_ids,
                            "generation_log_probs": generation_log_probs,
                        }
                    ],
                },
                # Pass through the original request fields
                **{k: v for k, v in request_body.items() if k not in ["responses_create_params"]},
            }

            return JSONResponse(content=nemo_gym_response)

        except Exception as e:
            LOG.exception("Error processing /run request")
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

    return app
