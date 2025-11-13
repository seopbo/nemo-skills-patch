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
Script classes for NeMo-Skills pipeline components.

These classes wrap NeMo-Run's run.Script interface to provide typed, reusable
job components (servers, clients, sandboxes) with explicit fields and
cross-component reference support for heterogeneous jobs.

Example:
    # Create a server script with automatic port allocation
    server = ServerScript(
        server_type="vllm",
        model_path="/models/llama-8b",
        cluster_config=cluster_config,
        num_gpus=8,
    )

    # Create a client that references the server
    client = GenerationClientScript(
        output_dir="/results",
        input_file="/data/input.jsonl",
        server=server,  # Cross-component reference
    )

    # Use in Command objects
    Command(script=server, container="vllm", ...)
    Command(script=client, container="nemo-skills", ...)
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import nemo_run as run

from nemo_skills.pipeline.utils.commands import sandbox_command
from nemo_skills.pipeline.utils.generation import get_generation_cmd
from nemo_skills.pipeline.utils.server import get_free_port, get_server_command
from nemo_skills.utils import get_logger_name

if TYPE_CHECKING:
    # Avoid circular imports for type hints
    pass

LOG = logging.getLogger(get_logger_name(__file__))


@dataclass
class BaseJobScript(run.Script):
    """Base class for job component scripts with heterogeneous job support.

    This class provides:
    - het_group_index tracking for cross-component references in heterogeneous SLURM jobs
    - hostname_ref() method for getting hostnames in het jobs
    - Common pattern for Script initialization

    Attributes:
        het_group_index: Index in heterogeneous job group (set by Pipeline at runtime)
    """

    het_group_index: Optional[int] = field(default=None, init=False, repr=False)

    def hostname_ref(self) -> str:
        """Get hostname reference for hetjob cross-component communication.

        Returns a shell variable reference that resolves to the master node hostname
        for this het group. Uses environment variables automatically exported by nemo-run:
            SLURM_MASTER_NODE_HET_GROUP_0, SLURM_MASTER_NODE_HET_GROUP_1, etc.

        These are set via:
            export SLURM_MASTER_NODE_HET_GROUP_N=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_N | head -n1)
        """
        if self.het_group_index is None:
            return "127.0.0.1"  # Local fallback for non-heterogeneous jobs

        # Use the environment variable exported by nemo-run
        return f"${{SLURM_MASTER_NODE_HET_GROUP_{self.het_group_index}:-localhost}}"


@dataclass(kw_only=True)
class ServerScript(BaseJobScript):
    """Script for model inference servers (vLLM, TRT-LLM, SGLang, etc.).

    This script wraps server command builders and provides:
    - Automatic port allocation if not specified
    - Type-safe server configuration
    - Cross-component address sharing (get_address())
    - Resource requirement tracking (num_gpus, num_nodes, num_tasks)

    Attributes:
        server_type: Type of server (vllm, trtllm, sglang, megatron, openai, etc.)
        model_path: Path to model weights or model name for API services
        cluster_config: Cluster configuration dictionary
        num_gpus: Number of GPUs required (default: 8)
        num_nodes: Number of nodes required (default: 1)
        server_args: Additional server-specific arguments
        server_entrypoint: Custom server entrypoint script (optional)
        port: Server port (allocated automatically if None)
        allocate_port: Whether to allocate port automatically (default: True)
        num_tasks: Number of MPI tasks (computed in __post_init__)
        log_prefix: Prefix for log files (default: "server")

    Example:
        # Basic usage
        server = ServerScript(
            server_type="vllm",
            model_path="/models/llama-3-8b",
            cluster_config=cluster_config,
            num_gpus=8,
        )

        # Access allocated port
        print(f"Server will run on port {server.port}")

        # Get full address for client connection
        address = server.get_address()  # Returns "hostname:port"
    """

    server_type: str
    model_path: str
    cluster_config: Dict
    num_gpus: int = 8
    num_nodes: int = 1
    server_args: str = ""
    server_entrypoint: Optional[str] = None  # Custom server entrypoint script
    port: Optional[int] = None
    allocate_port: bool = True

    # Computed fields (set in __post_init__)
    num_tasks: int = field(init=False, repr=False)
    log_prefix: str = field(default="server", init=False)

    def __post_init__(self):
        """Initialize server script.

        - Allocates port if not provided
        - Builds server command using get_server_command()
        - Sets self.inline to the command string
        - Computes num_tasks from server command builder
        """
        # Allocate port if not provided
        if self.port is None and self.allocate_port:
            self.port = get_free_port(strategy="random")
            LOG.debug(f"Allocated port {self.port} for {self.server_type} server")

        # Build server command
        cmd, self.num_tasks = get_server_command(
            server_type=self.server_type,
            num_gpus=self.num_gpus,
            num_nodes=self.num_nodes,
            model_path=self.model_path,
            cluster_config=self.cluster_config,
            server_port=self.port,
            server_args=self.server_args,
            server_entrypoint=self.server_entrypoint,
        )

        self.inline = cmd
        # Set entrypoint for run.Script (required by parent class)
        # Note: This is different from server_entrypoint which is for custom server scripts
        object.__setattr__(self, "entrypoint", "bash")
        super().__post_init__()

    def get_address(self) -> str:
        """Get server address for client connections.

        Returns hostname:port string that clients can use to connect.
        In heterogeneous jobs, hostname_ref() returns a bash expression
        that resolves at runtime.

        Returns:
            Server address in format "hostname:port"

        Example:
            # Use in client command
            client_cmd = f"python client.py --server-url http://{server.get_address()}"
        """
        return f"{self.hostname_ref()}:{self.port}"


@dataclass(kw_only=True)
class SandboxScript(BaseJobScript):
    """Script for code execution sandbox container.

    The sandbox provides a secure environment for executing LLM-generated code.
    This script wraps sandbox command builders and provides:
    - Automatic port allocation
    - Mount configuration (can optionally keep mounts, though risky)
    - Type-safe sandbox configuration

    Attributes:
        cluster_config: Cluster configuration dictionary
        port: Sandbox port (allocated automatically if None)
        keep_mounts: Whether to keep filesystem mounts (default: False, risky if True).
                     Note: This is stored for documentation but actually handled at
                     the executor level, not in the sandbox command itself.
        allocate_port: Whether to allocate port automatically (default: True)
        log_prefix: Prefix for log files (default: "sandbox")

    Example:
        sandbox = SandboxScript(
            cluster_config=cluster_config,
            keep_mounts=False,  # Safer: sandbox has no access to mounted paths
        )

        # Client can reference sandbox port
        client = GenerationClientScript(..., sandbox=sandbox)
    """

    cluster_config: Dict
    port: Optional[int] = None
    keep_mounts: bool = False
    allocate_port: bool = True
    log_prefix: str = field(default="sandbox", init=False)

    def __post_init__(self):
        """Initialize sandbox script.

        - Allocates port if not provided
        - Builds sandbox command using sandbox_command()
        - Sets self.inline to the command string
        """
        # Allocate port if not provided
        if self.port is None and self.allocate_port:
            self.port = get_free_port(strategy="random")
            LOG.debug(f"Allocated port {self.port} for sandbox")

        # Build sandbox command
        # Note: keep_mounts is handled at the executor level, not in the command itself
        cmd, _ = sandbox_command(
            cluster_config=self.cluster_config,
            port=self.port,
        )

        self.inline = cmd
        # Set entrypoint for run.Script (required by parent class)
        object.__setattr__(self, "entrypoint", "bash")
        super().__post_init__()


@dataclass(kw_only=True)
class GenerationClientScript(BaseJobScript):
    """Script for LLM generation/inference client.

    This script wraps generation command builders and provides:
    - Cross-component references to server and sandbox
    - Lazy command building when cross-refs are present
    - Type-safe generation configuration
    - Environment variable handling for sandbox/server communication

    Attributes:
        output_dir: Directory for output files
        input_file: Input JSONL file (mutually exclusive with input_dir)
        input_dir: Input directory (mutually exclusive with input_file)
        extra_arguments: Additional arguments for generation script
        random_seed: Random seed for sampling (optional)
        chunk_id: Chunk ID for parallel processing (optional)
        num_chunks: Total number of chunks (required if chunk_id set)
        preprocess_cmd: Command to run before generation (optional)
        postprocess_cmd: Command to run after generation (optional)
        wandb_parameters: WandB logging configuration (optional)
        with_sandbox: Whether sandbox is enabled
        script: Module or file path for generation script (default: nemo_skills.inference.generate)
        server: Reference to ServerScript for cross-component communication (optional)
        sandbox: Reference to SandboxScript for cross-component communication (optional)
        log_prefix: Prefix for log files (default: "main")

    Example:
        # Without cross-component references
        client = GenerationClientScript(
            output_dir="/results",
            input_file="/data/input.jsonl",
            extra_arguments="++inference.temperature=0.7",
        )

        # With server and sandbox references (lazy command building)
        client = GenerationClientScript(
            output_dir="/results",
            input_file="/data/input.jsonl",
            server=server_script,  # Will resolve server address at runtime
            sandbox=sandbox_script,  # Will set NEMO_SKILLS_SANDBOX_PORT env var
            with_sandbox=True,
        )
    """

    output_dir: str
    input_file: Optional[str] = None
    input_dir: Optional[str] = None
    extra_arguments: str = ""
    random_seed: Optional[int] = None
    chunk_id: Optional[int] = None
    num_chunks: Optional[int] = None
    preprocess_cmd: Optional[str] = None
    postprocess_cmd: Optional[str] = None
    wandb_parameters: Optional[Dict] = None
    with_sandbox: bool = False
    script: str = "nemo_skills.inference.generate"

    # Cross-component references (optional)
    server: Optional["ServerScript"] = None
    sandbox: Optional["SandboxScript"] = None

    log_prefix: str = field(default="main", init=False)

    def __post_init__(self):
        """Initialize generation client script.

        If cross-component references (server/sandbox) are present, builds
        command lazily via a callable. The callable is evaluated later when
        het_group_index is assigned, allowing hostname_ref() to work correctly.

        Otherwise, builds command immediately.
        """
        # Check if we need lazy command building (has cross-component refs)
        has_cross_refs = self.sandbox is not None

        if has_cross_refs:
            # Lazy command building - will be evaluated when het_group_index is set
            def build_cmd() -> Tuple[str, Dict]:
                """Build command at runtime when cross-refs are resolved."""
                env_vars = {}

                # Add sandbox port to environment if sandbox is referenced
                if self.sandbox:
                    env_vars["NEMO_SKILLS_SANDBOX_PORT"] = str(self.sandbox.port)

                # Build generation command
                cmd = get_generation_cmd(
                    output_dir=self.output_dir,
                    input_file=self.input_file,
                    input_dir=self.input_dir,
                    extra_arguments=self.extra_arguments,
                    random_seed=self.random_seed,
                    chunk_id=self.chunk_id,
                    num_chunks=self.num_chunks,
                    preprocess_cmd=self.preprocess_cmd,
                    postprocess_cmd=self.postprocess_cmd,
                    wandb_parameters=self.wandb_parameters,
                    with_sandbox=self.with_sandbox,
                    script=self.script,
                )

                # Return command and runtime metadata (environment vars)
                return cmd, {"environment": env_vars}

            self.inline = build_cmd
        else:
            # No cross-refs, build immediately
            cmd = get_generation_cmd(
                output_dir=self.output_dir,
                input_file=self.input_file,
                input_dir=self.input_dir,
                extra_arguments=self.extra_arguments,
                random_seed=self.random_seed,
                chunk_id=self.chunk_id,
                num_chunks=self.num_chunks,
                preprocess_cmd=self.preprocess_cmd,
                postprocess_cmd=self.postprocess_cmd,
                wandb_parameters=self.wandb_parameters,
                with_sandbox=self.with_sandbox,
                script=self.script,
            )
            self.inline = cmd

        # Set entrypoint for run.Script (required by parent class)
        object.__setattr__(self, "entrypoint", "bash")
        super().__post_init__()
