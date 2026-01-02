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

import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from utils import require_env_var

from tests.conftest import docker_rm


@dataclass
class NemoRLTestConfig:
    """Configuration for NeMo-RL integration test.

    All parameters can be configured at creation time:
    - vllm_discovery_timeout: vLLM server discovery timeout in seconds (default: 720)
    - proxy_ready_timeout: Proxy server ready timeout in seconds (default: 120)
    - config_dir: Path to cluster config directory (default: tests/gpu-tests/)
                  Can be overridden via NEMO_SKILLS_TEST_CONFIG_DIR environment variable

    Example:
        # Use defaults
        config = NemoRLTestConfig(model="...", log_dir=Path("..."), proxy_port=7000)

        # Override timeouts
        config = NemoRLTestConfig(
            model="...",
            log_dir=Path("..."),
            proxy_port=7000,
            vllm_discovery_timeout=1800,  # 30 minutes
            proxy_ready_timeout=300,       # 5 minutes
        )

        # Use external config (e.g., /home/wedu/test-local.yaml)
        export NEMO_SKILLS_TEST_CONFIG_DIR=/home/wedu
        pytest tests/gpu-tests/test_nemo_rl_integration.py
    """

    model: str
    log_dir: Path
    proxy_port: int
    vllm_discovery_timeout: int = 720  # 12 minutes - can be overridden at creation
    proxy_ready_timeout: int = 120  # 2 minutes - can be overridden at creation
    cluster: str = "test-local"
    config_dir: Path | None = None  # Will be set from env or default

    def __post_init__(self):
        """Set config_dir from environment variable if not provided."""
        if self.config_dir is None:
            # Check for external config directory (e.g., /home/wedu/)
            env_config_dir = os.environ.get("NEMO_SKILLS_TEST_CONFIG_DIR")
            if env_config_dir:
                self.config_dir = Path(env_config_dir)
            else:
                # Default to test directory
                self.config_dir = Path(__file__).absolute().parent


def wait_for_url_in_log(
    log_file: Path, pattern: str = r"vLLM Base URLs: \['(http://[\d.]+:\d+/v1)'\]", timeout: int = 720
) -> str | None:
    """Wait for a URL to appear in a log file.

    Default pattern matches NeMo-RL/NeMo-Gym vLLM Base URLs format:
        vLLM Base URLs: ['http://10.110.40.219:50651/v1']
    """
    regex = re.compile(pattern)
    start_time = time.time()

    while time.time() - start_time < timeout:
        if log_file.exists():
            try:
                content = log_file.read_text()
                match = regex.search(content)
                if match:
                    return match.group(1)
            except Exception:
                pass
        time.sleep(2)

    return None


def wait_for_ready_message(log_file: Path, message: str = "Server ready at", timeout: int = 120) -> bool:
    """Wait for a ready message in a log file."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        if log_file.exists():
            try:
                content = log_file.read_text()
                if message in content:
                    return True
            except Exception:
                pass
        time.sleep(2)

    return False


@pytest.mark.gpu
def test_nemo_rl_integration():
    """Test NeMo-RL integration with NeMo-Skills proxy and NeMo-Gym.

    Architecture:
      1. NeMo-RL starts with NeMo-Gym, vLLM exposes HTTP
      2. Detect vLLM URL from logs
      3. Start proxy with discovered vLLM URL (explicit configuration)
      4. NemoGym environment calls proxy for rollouts
      5. Proxy forwards to vLLM, returns responses
    """
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")

    # Create test configuration
    log_dir = Path(tempfile.mkdtemp(prefix="nemo-rl-test-"))
    config = NemoRLTestConfig(
        model=model_path,
        log_dir=log_dir,
        proxy_port=0,  # Auto-assign
    )

    print("Test configuration:")
    print(f"  Model: {config.model}")
    print(f"  Log directory: {config.log_dir}")
    print(f"  vLLM timeout: {config.vllm_discovery_timeout}s")
    print(f"  Proxy timeout: {config.proxy_ready_timeout}s")

    # Setup log files and data
    nemo_rl_log = config.log_dir / "nemo-rl.log"
    proxy_log = config.log_dir / "nemo-skills.log"
    data_file = config.log_dir / "test_data.jsonl"

    # Auto-assign proxy port
    if config.proxy_port == 0:
        import socket

        s = socket.socket()
        s.bind(("", 0))
        config.proxy_port = s.getsockname()[1]
        s.close()
        print(f"  Auto-assigned proxy port: {config.proxy_port}")

    proxy_url = f"http://localhost:{config.proxy_port}/v1"

    # Create test data
    data_file.write_text(
        '{"problem": "What is 1 + 1?", "expected_answer": "2", "difficulty": "easy", "category": "arithmetic", "task_name": "math"}\n'
        '{"problem": "What is 3 + 2?", "expected_answer": "5", "difficulty": "easy", "category": "arithmetic", "task_name": "math"}\n'
        '{"problem": "What is 10 - 3?", "expected_answer": "8", "difficulty": "easy", "category": "arithmetic", "task_name": "math"}\n'
    )

    # NemoGym bug patch - server returns YAML but client expects JSON
    nemo_gym_patch = (
        "sed -i 's/return OmegaConf\\.to_yaml(get_global_config_dict())"
        "/import json; return json.dumps(OmegaConf.to_container(get_global_config_dict()))/' "
        "/opt/nemo_rl_venv/lib/python3.12/site-packages/nemo_gym/server_utils.py"
    )

    try:
        # Clean up any existing containers
        docker_rm([])

        # ================================================================
        # Step 1: Start NeMo-RL with NeMo-Gym (vLLM will expose HTTP)
        # ================================================================
        print("\nStarting NeMo-RL with NeMo-Gym integration...")

        nemo_rl_cmd = (
            f"{nemo_gym_patch} && "
            f"python -u -m nemo_skills.training.nemo_rl.start_grpo "
            f"  ++policy.model_name={config.model} "
            f"  ++data.train_data_path={data_file} "
            f"  ++data.prompt.prompt_config=generic/math "
            f"  ++policy.generation.vllm_cfg.expose_http_server=true "
            f"  ++policy.generation.vllm_cfg.async_engine=true "
            f"  ++policy.generation.vllm_cfg.skip_tokenizer_init=false "
            f"  ++policy.megatron_cfg.enabled=false "
            f"  ++env.should_use_nemo_gym=true "
            f"  ++nemo_gym.policy_base_url={proxy_url} "
            f"  ++grpo.max_num_epochs=5 "
            f"  ++grpo.num_iterations=1 "
            f"  ++grpo.num_generations_per_prompt=2 "
            f"  ++grpo.num_prompts_per_step=1 "
            f"  ++policy.train_global_batch_size=2"
        )

        ns_run_cmd = (
            f"ns run_cmd --cluster {config.cluster} "
            f"--config_dir {config.config_dir} "
            f"--container nemo-rl --num_gpus 1 "
            f'"{nemo_rl_cmd}"'
        )

        # Start NeMo-RL in background, redirect output to log
        with open(nemo_rl_log, "w") as log:
            nemo_rl_proc = subprocess.Popen(ns_run_cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, text=True)

        # ================================================================
        # Step 2: Wait for vLLM to start and get its URL
        # ================================================================
        print(f"Waiting for vLLM to start (timeout: {config.vllm_discovery_timeout}s)...")

        vllm_url = wait_for_url_in_log(nemo_rl_log, timeout=config.vllm_discovery_timeout)

        if not vllm_url:
            print("ERROR: Failed to discover vLLM URL")
            print("=== NeMo-RL Log (last 100 lines) ===")
            if nemo_rl_log.exists():
                lines = nemo_rl_log.read_text().splitlines()
                print("\n".join(lines[-100:]))
            pytest.fail("vLLM URL not found in logs")

        print(f"Found vLLM at: {vllm_url}")

        # ================================================================
        # Step 3: Start NeMo-Skills proxy with discovered vLLM URL
        # ================================================================
        print(f"Starting NeMo-Skills proxy (connecting to {vllm_url})...")

        # Use explicit configuration (no environment variables!)
        proxy_cmd = (
            f"python -m nemo_skills.inference.generate "
            f"  ++start_server=True "
            f"  ++inference.temperature=-1 "
            f"  ++inference.top_p=-1 "
            f"  ++generate_port={config.proxy_port} "
            f"  ++prompt_format=openai "
            f"  ++evaluator.type=math "
            f"  ++server.base_url={vllm_url} "
            f"  ++server.model={config.model}"
        )

        ns_proxy_cmd = (
            f"ns run_cmd --cluster {config.cluster} "
            f"--config_dir {config.config_dir} "
            f"--container nemo-skills --num_gpus 0 "
            f'"{proxy_cmd}"'
        )

        # Start proxy in background, redirect output to log
        with open(proxy_log, "w") as log:
            proxy_proc = subprocess.Popen(ns_proxy_cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, text=True)

        # Wait for proxy to be ready
        print(f"Waiting for proxy to be ready (timeout: {config.proxy_ready_timeout}s)...")

        if not wait_for_ready_message(proxy_log, timeout=config.proxy_ready_timeout):
            print("ERROR: Proxy did not become ready")
            print("=== Proxy Log (last 100 lines) ===")
            if proxy_log.exists():
                lines = proxy_log.read_text().splitlines()
                print("\n".join(lines[-100:]))
            pytest.fail("Proxy not ready")

        print("Proxy is ready!")

        # ================================================================
        # Step 4: Wait for NeMo-RL training to complete
        # ================================================================
        print("\nWaiting for NeMo-RL training to complete...")

        nemo_rl_proc.wait()

        if nemo_rl_proc.returncode != 0:
            print(f"ERROR: NeMo-RL training failed with exit code {nemo_rl_proc.returncode}")
            print("=== NeMo-RL Log (last 200 lines) ===")
            if nemo_rl_log.exists():
                lines = nemo_rl_log.read_text().splitlines()
                print("\n".join(lines[-200:]))
            pytest.fail(f"NeMo-RL training failed (exit code {nemo_rl_proc.returncode})")

        print("NeMo-RL training completed successfully!")

        # ================================================================
        # Step 5: Verify proxy was used
        # ================================================================
        print("\nChecking results...")

        # Check if NeMo-Gym was actually used
        log_content = nemo_rl_log.read_text()
        if "NeMo-Gym environment created" in log_content:
            print("✓ NeMo-Gym environment was created")
        elif "NeMo-Gym is not available" in log_content:
            print("✗ NeMo-Gym import failed - check logs for details")
            pytest.fail("NeMo-Gym not available")
        else:
            print("⚠️ Couldn't determine if NeMo-Gym was used")

        # Count requests to the proxy
        proxy_content = proxy_log.read_text() if proxy_log.exists() else ""
        proxy_requests = proxy_content.count("/v1/chat/completions")

        print(f"Proxy received approximately {proxy_requests} requests")

        if proxy_requests > 0:
            print("\n✅ Integration test PASSED")
            print("  - NeMo-RL started vLLM with HTTP exposed")
            print("  - Proxy connected to vLLM via explicit configuration")
            print(f"  - NeMo-Gym used proxy for rollouts ({proxy_requests} requests)")
        else:
            print("\n⚠️ Integration test completed but proxy may not have been used")
            print("  Possible reasons:")
            print("  - NeMo-Gym wasn't available in container")
            print("  - Proxy started too late (race condition)")
            print("  - Training finished before doing rollouts")
            pytest.fail("Proxy not used")

    finally:
        # Clean up processes
        try:
            nemo_rl_proc.terminate()
            nemo_rl_proc.wait(timeout=10)
        except Exception:
            pass

        try:
            proxy_proc.terminate()
            proxy_proc.wait(timeout=10)
        except Exception:
            pass

        # Clean up files (keep logs on failure)
        # pytest will handle this via its cleanup mechanism
        print(f"\nTest logs preserved at: {config.log_dir}")
