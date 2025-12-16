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

import base64
import os
import tempfile
from unittest.mock import patch

import pytest

from nemo_skills.inference.model.vllm import VLLMModel, audio_file_to_base64


def test_audio_file_to_base64():
    """Test basic audio file encoding to base64."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f:
        test_content = b"RIFF" + b"\x00" * 100
        f.write(test_content)
        temp_path = f.name

    try:
        result = audio_file_to_base64(temp_path)
        assert isinstance(result, str)
        assert len(result) > 0
        decoded = base64.b64decode(result)
        assert decoded == test_content
    finally:
        os.unlink(temp_path)


@pytest.fixture
def mock_vllm_model():
    """Create a mock VLLMModel for testing audio preprocessing."""
    with patch.object(VLLMModel, "__init__", lambda self, **kwargs: None):
        model = VLLMModel()
        model.data_dir = ""
        return model


def test_content_text_to_list_with_audio(mock_vllm_model, tmp_path):
    """Test converting string content with audio to list format.

    CRITICAL: Audio must come BEFORE text for Qwen Audio to transcribe correctly.
    """
    audio_path = tmp_path / "test.wav"
    with open(audio_path, "wb") as f:
        f.write(b"RIFF" + b"\x00" * 100)

    # Set data_dir to tmp_path parent so path resolution works
    mock_vllm_model.data_dir = str(tmp_path)

    message = {"role": "user", "content": "Describe this audio", "audio": {"path": "test.wav"}}

    result = mock_vllm_model.content_text_to_list(message)

    assert isinstance(result["content"], list)
    assert len(result["content"]) == 2
    assert result["content"][0]["type"] == "audio_url"
    assert result["content"][0]["audio_url"]["url"].startswith("data:audio/wav;base64,")
    assert result["content"][1]["type"] == "text"


def test_content_text_to_list_with_multiple_audios(mock_vllm_model, tmp_path):
    """Test handling message with multiple audio files.

    CRITICAL: Audio must come BEFORE text for Qwen Audio to transcribe correctly.
    """
    audio_paths = []
    for i in range(2):
        audio_path = tmp_path / f"test_{i}.wav"
        with open(audio_path, "wb") as f:
            f.write(b"RIFF" + b"\x00" * 100)
        audio_paths.append(f"test_{i}.wav")

    mock_vllm_model.data_dir = str(tmp_path)

    message = {
        "role": "user",
        "content": "Compare these",
        "audios": [{"path": audio_paths[0]}, {"path": audio_paths[1]}],
    }

    result = mock_vllm_model.content_text_to_list(message)

    assert isinstance(result["content"], list)
    assert len(result["content"]) == 3
    # Audio MUST come before text for Qwen Audio
    assert result["content"][0]["type"] == "audio_url"
    assert result["content"][1]["type"] == "audio_url"
    assert result["content"][2]["type"] == "text"
