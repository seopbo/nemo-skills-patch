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
NeMo-Skills Proxy Server Utilities for NeMo-RL/NeMo-Gym Integration.

This module provides:
1. OpenAI-compatible API models for FastAPI endpoints
2. A factory function to create proxy FastAPI applications
3. Discovery utilities for finding vLLM servers in NeMo-RL environments

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
    - /v1/chat/completions: OpenAI-compatible chat completions
    - /v1/completions: OpenAI-compatible text completions
    - /v1/models: Model listing
    - /generate: NeMo-Skills native endpoint
    - /health: Health check
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Protocol

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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
class VLLMServerConfig:
    """Configuration for a discovered vLLM server."""

    base_url: str  # Full base URL, e.g., "http://localhost:5000"
    host: str
    port: int
    source: str  # How the server was discovered
    model_name: str | None = None  # The model name served by vLLM


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


def discover_vllm_server(
    ray_address: str | None = None,
    head_server_url: str | None = None,
    timeout: float = 5.0,
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
    """
    # Method 1: Environment variables
    config = _discover_from_env()
    if config:
        config.model_name = get_vllm_model_name(config.base_url, timeout)
        return config

    # Method 2: Ray named values
    config = _discover_from_ray(ray_address)
    if config:
        config.model_name = get_vllm_model_name(config.base_url, timeout)
        return config

    # Method 3: NeMo-Gym head server
    config = _discover_from_head_server(head_server_url, timeout)
    if config:
        config.model_name = get_vllm_model_name(config.base_url, timeout)
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
# OpenAI-Compatible API Models
# =============================================================================
#
# Note: The OpenAI Python library exposes TypedDicts for requests (not Pydantic
# models) and Pydantic models for responses (designed for parsing, not defining
# endpoints). We define explicit Pydantic models here for FastAPI compatibility
# and clear API documentation.


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""

    role: str
    content: str
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = "nemo-skills"
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: str | None = None
    # NeMo-Skills extension: extra data for prompt filling
    extra_data: dict | None = Field(default=None, description="Extra data for NeMo-Skills prompt filling")


class CompletionRequest(BaseModel):
    """OpenAI-compatible text completion request."""

    model: str = "nemo-skills"
    prompt: str | list[str]
    temperature: float | None = None
    top_p: float | None = None
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: str | None = None
    # NeMo-Skills extension: extra data for prompt filling
    extra_data: dict | None = Field(default=None, description="Extra data for NeMo-Skills prompt filling")


class ChatCompletionChoice(BaseModel):
    """OpenAI-compatible chat completion choice."""

    index: int
    message: ChatMessage
    finish_reason: str | None = "stop"


class CompletionChoice(BaseModel):
    """OpenAI-compatible text completion choice."""

    index: int
    text: str
    finish_reason: str | None = "stop"


class Usage(BaseModel):
    """OpenAI-compatible usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class CompletionResponse(BaseModel):
    """OpenAI-compatible text completion response."""

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage


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
) -> FastAPI:
    """
    Create a FastAPI application that serves as an OpenAI-compatible proxy.

    This proxy passes OpenAI-format messages through to the NeMo-Skills generation
    pipeline, which can handle code execution loops, tool calling, and sandbox
    integration.

    When used with NeMo-RL (which already templates prompts via ns_data_processor),
    configure with prompt_format=openai to pass messages through without re-templating.

    Args:
        generation_task: A GenerationTask instance (has cfg and _process_single_datapoint_core).
                        Either this or process_fn must be provided.
        process_fn: Alternative to generation_task - a coroutine function that processes
                   data points. Signature: async (data_point: dict, all_data: list) -> dict
        generation_key: The key in the output dict that contains the generation.
                       Only used if process_fn is provided.
        title: Title for the FastAPI application.

    Returns:
        FastAPI application with OpenAI-compatible endpoints.

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

    app = FastAPI(
        title=title,
        description="Proxy server that adds NeMo-Skills prompting logic to model generations. "
        "Compatible with OpenAI API for seamless integration with NeMo-Gym/NeMo-RL.",
    )

    @app.post("/generate")
    async def generate_endpoint(data_point: dict):
        """Generate output for a single data point (NeMo-Skills native format).

        Request body should be the data point dictionary directly.
        The body is parsed as JSON and passed directly to the generation logic.
        """
        try:
            output = await _process_fn(data_point, [])
            return JSONResponse(content=output)
        except Exception as e:
            LOG.exception("Error processing generation request")
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(request: ChatCompletionRequest):
        """OpenAI-compatible chat completions endpoint.

        This endpoint passes messages through to the NeMo-Skills generation pipeline.
        Messages are passed through directly (no re-templating), making it compatible
        with NeMo-RL which already templates prompts via ns_data_processor.

        The generation pipeline handles:
        - Code execution loops (if code_execution=True)
        - Tool calling (if configured)
        - Sandbox integration

        Usage with NeMo-RL:
            Configure policy.generation to point to this server as an OpenAI endpoint.

        Usage with NeMo-Gym SimpleResponsesAPIModel:
            Set the base_url to point to this server.
        """
        try:
            # Build data point - pass messages through directly
            messages = [{"role": m.role, "content": m.content} for m in request.messages]
            data_point = {"messages": messages}

            # If extra_data is provided, merge it (for metadata like expected_answer)
            if request.extra_data:
                data_point.update(request.extra_data)

            # Process through NeMo-Skills pipeline
            output = await _process_fn(data_point, [])

            # Build OpenAI-compatible response
            generation_text = output.get(_generation_key, output.get("generation", ""))
            num_tokens = output.get("num_generated_tokens", 0)

            response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=generation_text),
                        finish_reason=output.get("finish_reason", "stop"),
                    )
                ],
                usage=Usage(
                    prompt_tokens=output.get("num_input_tokens", 0),
                    completion_tokens=num_tokens,
                    total_tokens=output.get("num_input_tokens", 0) + num_tokens,
                ),
            )

            return response

        except Exception as e:
            LOG.exception("Error processing chat completion request")
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions(request: CompletionRequest):
        """OpenAI-compatible text completions endpoint.

        Converts text prompts to message format and passes through to the pipeline.
        """
        try:
            # Handle single prompt or list of prompts
            prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt

            choices = []
            total_completion_tokens = 0

            for idx, prompt in enumerate(prompts):
                # Convert text prompt to message format for consistent handling
                data_point = {"messages": [{"role": "user", "content": prompt}]}
                if request.extra_data:
                    data_point.update(request.extra_data)

                # Process through NeMo-Skills pipeline
                output = await _process_fn(data_point, [])

                generation_text = output.get(_generation_key, output.get("generation", ""))
                num_tokens = output.get("num_generated_tokens", 0)
                total_completion_tokens += num_tokens

                choices.append(
                    CompletionChoice(
                        index=idx,
                        text=generation_text,
                        finish_reason=output.get("finish_reason", "stop"),
                    )
                )

            response = CompletionResponse(
                id=f"cmpl-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=request.model,
                choices=choices,
                usage=Usage(
                    prompt_tokens=0,  # Not tracked for text completions
                    completion_tokens=total_completion_tokens,
                    total_tokens=total_completion_tokens,
                ),
            )

            return response

        except Exception as e:
            LOG.exception("Error processing completion request")
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

    @app.get("/v1/models")
    async def list_models():
        """List available models (OpenAI-compatible)."""
        return {
            "object": "list",
            "data": [
                {
                    "id": "nemo-skills",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "nvidia",
                }
            ],
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    return app
