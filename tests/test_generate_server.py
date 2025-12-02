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
Pytest tests for the FastAPI generation server.

These tests verify that the /generate endpoint works correctly with NVIDIA API endpoints.

To run these tests:
    pytest tests/test_generate_server.py -v

To run with a server already running:
    pytest tests/test_generate_server.py -v --server-url http://localhost:7000

To skip tests if server is not available:
    pytest tests/test_generate_server.py -v --skip-if-no-server
"""

import os

import pytest
import requests

# Default configuration
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 7000
DEFAULT_SERVER_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"

# NVIDIA API Configuration (can be overridden via environment variables)
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "YOUR_NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "nvidia/nvidia-nemotron-nano-9b-v2")

# Sample data points for testing
SAMPLE_DATA_POINT_OPENAI = {
    "messages": [{"role": "system", "content": "/nothink"}, {"role": "user", "content": "What is 2 + 2?"}]
}

SAMPLE_DATA_POINT_NS = {"problem": "What is 2 + 2?", "expected_answer": "4"}


def check_server_health(server_url: str, timeout: int = 5) -> bool:
    """
    Check if the server is running by attempting to connect.

    Args:
        server_url: Base URL of the generation server
        timeout: Timeout in seconds

    Returns:
        True if server is reachable, False otherwise
    """
    try:
        # FastAPI automatically creates a /docs endpoint
        response = requests.get(f"{server_url}/docs", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def create_server_config(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    model: str = NVIDIA_MODEL,
    base_url: str = NVIDIA_BASE_URL,
    api_key: str = NVIDIA_API_KEY,
    prompt_format: str = "openai",
    prompt_config: str | None = None,
) -> dict:
    """
    Create a server configuration dictionary.

    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        model: Model name/identifier
        base_url: NVIDIA API base URL
        api_key: NVIDIA API key
        prompt_format: Prompt format ("ns" or "openai")
        prompt_config: Prompt config file path (required if prompt_format=ns)

    Returns:
        Configuration dictionary
    """
    return {
        "start_server": True,
        "generate_host": host,
        "generate_port": port,
        "input_file": None,
        "output_file": None,
        "server": {
            "server_type": "openai",
            "base_url": base_url,
            "model": model,
            "api_key": api_key,
        },
        "prompt_format": prompt_format,
        "prompt_config": prompt_config,
        "inference": {
            "endpoint_type": "chat",
            "temperature": 0.0,
        },
    }


@pytest.fixture(scope="session")
def server_url(pytestconfig) -> str:
    """Fixture providing the server URL (can be overridden via --server-url)."""
    return pytestconfig.getoption("--server-url", default=DEFAULT_SERVER_URL)


@pytest.fixture(scope="session")
def skip_if_no_server(pytestconfig) -> bool:
    """Fixture indicating whether to skip tests if server is not available."""
    return pytestconfig.getoption("--skip-if-no-server", default=False)


@pytest.fixture
def server_config() -> dict:
    """Fixture providing default server configuration."""
    return create_server_config()


@pytest.fixture(scope="function")
def server_available(server_url, skip_if_no_server):
    """
    Fixture that checks if server is available and skips tests if not.

    Note: This fixture does not start the server. The server should be started
    manually or via a separate process before running tests.
    """
    if not check_server_health(server_url):
        if skip_if_no_server:
            pytest.skip("Server not available and --skip-if-no-server is set")
        pytest.fail(
            f"Server not available at {server_url}. "
            "Start the server manually or use --skip-if-no-server to skip tests."
        )
    return True


def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--server-url",
        action="store",
        default=DEFAULT_SERVER_URL,
        help="URL of the generation server (default: http://0.0.0.0:7000)",
    )
    parser.addoption(
        "--skip-if-no-server",
        action="store_true",
        default=False,
        help="Skip tests if server is not available",
    )


class TestGenerateServer:
    """Test class for the FastAPI generation server."""

    def test_server_health(self, server_url, skip_if_no_server, server_available):
        """Test that the server is responding to health checks."""
        # server_available fixture will handle skipping/failing if server is not available
        assert check_server_health(server_url), f"Server health check failed at {server_url}"

    def test_generate_endpoint_openai_format(self, server_url, server_available):
        """Test the /generate endpoint with OpenAI format data point."""

        endpoint_url = f"{server_url}/generate"

        response = requests.post(
            endpoint_url,
            json=SAMPLE_DATA_POINT_OPENAI,
            headers={"Content-Type": "application/json"},
            timeout=300,  # 5 minute timeout for generation
        )

        assert response.status_code == 200, f"Request failed with status {response.status_code}: {response.text}"

        result = response.json()
        assert isinstance(result, dict), "Response should be a dictionary"
        assert "generation" in result, f"'generation' key not found in response. Available keys: {list(result.keys())}"
        assert isinstance(result["generation"], str), "Generation should be a string"
        assert len(result["generation"]) > 0, "Generation should not be empty"

    @pytest.mark.parametrize(
        "data_point",
        [
            SAMPLE_DATA_POINT_OPENAI,
            {"messages": [{"role": "user", "content": "Say hello"}]},
        ],
    )
    def test_generate_endpoint_various_inputs(self, server_url, server_available, data_point):
        """Test the /generate endpoint with various input formats."""

        endpoint_url = f"{server_url}/generate"

        response = requests.post(
            endpoint_url,
            json=data_point,
            headers={"Content-Type": "application/json"},
            timeout=300,
        )

        assert response.status_code == 200, f"Request failed with status {response.status_code}: {response.text}"
        result = response.json()
        assert "generation" in result

    def test_generate_endpoint_error_handling(self, server_url, server_available):
        """Test error handling for invalid requests."""

        endpoint_url = f"{server_url}/generate"

        # Test with invalid JSON
        response = requests.post(
            endpoint_url,
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        # Should return 422 (Unprocessable Entity) for invalid JSON
        assert response.status_code in [400, 422], f"Expected 400 or 422, got {response.status_code}"

    def test_generate_endpoint_response_structure(self, server_url, server_available):
        """Test that the response has the expected structure."""

        endpoint_url = f"{server_url}/generate"

        response = requests.post(
            endpoint_url,
            json=SAMPLE_DATA_POINT_OPENAI,
            headers={"Content-Type": "application/json"},
            timeout=300,
        )

        assert response.status_code == 200
        result = response.json()

        # Check for expected keys
        assert "generation" in result

        # Check for optional generation stats (if enabled)
        # These may or may not be present depending on config
        if "generation_time" in result:
            assert isinstance(result["generation_time"], (int, float))
            assert result["generation_time"] >= 0


class TestServerConfiguration:
    """Test class for server configuration utilities."""

    def test_create_server_config_defaults(self):
        """Test creating server config with default values."""
        config = create_server_config()

        assert config["start_server"] is True
        assert config["generate_host"] == DEFAULT_HOST
        assert config["generate_port"] == DEFAULT_PORT
        assert config["input_file"] is None
        assert config["output_file"] is None
        assert config["server"]["server_type"] == "openai"
        assert config["prompt_format"] == "openai"

    def test_create_server_config_custom(self):
        """Test creating server config with custom values."""
        config = create_server_config(
            host="127.0.0.1",
            port=8000,
            model="custom-model",
            prompt_format="ns",
        )

        assert config["generate_host"] == "127.0.0.1"
        assert config["generate_port"] == 8000
        assert config["server"]["model"] == "custom-model"
        assert config["prompt_format"] == "ns"


if __name__ == "__main__":
    # Allow running as a script for convenience
    pytest.main([__file__, "-v"])
