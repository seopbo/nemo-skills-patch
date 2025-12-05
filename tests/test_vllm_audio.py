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
from unittest.mock import MagicMock

import pytest

from nemo_skills.inference.generate import GenerationTask


@pytest.fixture
def mock_generation_task():
    """Create a mock GenerationTask for testing audio preprocessing."""
    mock_cfg = MagicMock()
    mock_cfg.drop_content_types = ["audio_url"]

    task = MagicMock(spec=GenerationTask)
    task.cfg = mock_cfg
    # Use the real methods
    task._audio_file_to_base64 = GenerationTask._audio_file_to_base64.__get__(task)
    task._convert_audio_to_base64 = GenerationTask._convert_audio_to_base64.__get__(task)
    return task


def test_audio_file_to_base64(mock_generation_task):
    """Test basic audio file encoding to base64."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f:
        test_content = b"RIFF" + b"\x00" * 100
        f.write(test_content)
        temp_path = f.name

    try:
        result = mock_generation_task._audio_file_to_base64(temp_path)
        assert isinstance(result, str)
        assert len(result) > 0
        decoded = base64.b64decode(result)
        assert decoded == test_content
    finally:
        os.unlink(temp_path)


def test_convert_audio_to_base64_with_audio(mock_generation_task, tmp_path):
    """Test converting string content with audio to list format.

    CRITICAL: Audio must come BEFORE text for Qwen Audio to transcribe correctly.
    """
    audio_path = tmp_path / "audio" / "test.wav"
    audio_path.parent.mkdir(exist_ok=True)
    with open(audio_path, "wb") as f:
        f.write(b"RIFF" + b"\x00" * 100)

    message = {"role": "user", "content": "Describe this audio", "audio": {"path": str(audio_path)}}

    result = mock_generation_task._convert_audio_to_base64(message)

    assert isinstance(result["content"], list)
    assert len(result["content"]) == 2
    assert result["content"][0]["type"] == "audio_url"
    assert result["content"][0]["audio_url"]["url"].startswith("data:audio/wav;base64,")
    assert result["content"][1]["type"] == "text"
    assert "audio" not in result


def test_convert_audio_to_base64_with_multiple_audios(mock_generation_task, tmp_path):
    """Test handling message with multiple audio files.

    CRITICAL: Audio must come BEFORE text for Qwen Audio to transcribe correctly.
    """
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(exist_ok=True)

    audio_paths = []
    for i in range(2):
        audio_path = audio_dir / f"test_{i}.wav"
        with open(audio_path, "wb") as f:
            f.write(b"RIFF" + b"\x00" * 100)
        audio_paths.append(str(audio_path))

    message = {
        "role": "user",
        "content": "Compare these",
        "audios": [{"path": audio_paths[0]}, {"path": audio_paths[1]}],
    }

    result = mock_generation_task._convert_audio_to_base64(message)

    assert isinstance(result["content"], list)
    assert len(result["content"]) == 3
    # Audio MUST come before text for Qwen Audio
    assert result["content"][0]["type"] == "audio_url"
    assert result["content"][1]["type"] == "audio_url"
    assert result["content"][2]["type"] == "text"
    # Original audios key should be removed
    assert "audios" not in result
