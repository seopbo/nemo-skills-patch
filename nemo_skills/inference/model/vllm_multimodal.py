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
import logging
import os

from nemo_skills.utils import get_logger_name

from .vllm import VLLMModel

LOG = logging.getLogger(get_logger_name(__file__))


class VLLMMultimodalModel(VLLMModel):
    """VLLMModel with support for saving audio responses to disk.

    When the server returns audio in the response, this model will:
    1. Save the audio bytes to a file in output_dir/audio/
    2. Replace the base64 data with the file path in the result
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_audio_dir = None
        if self.output_dir:
            self.output_audio_dir = os.path.join(self.output_dir, "audio")
            os.makedirs(self.output_audio_dir, exist_ok=True)
            LOG.info(f"Audio responses will be saved to: {self.output_audio_dir}")

    def _parse_chat_completion_response(self, response, include_response: bool = False, **kwargs) -> dict:
        """Parse chat completion response and save any audio to disk."""
        result = super()._parse_chat_completion_response(response, include_response=include_response, **kwargs)

        choice = response.choices[0]
        if hasattr(choice.message, "audio") and choice.message.audio:
            result["audio"] = self._process_audio_response(choice.message.audio, response.id)

        return result

    def _process_audio_response(self, audio_data, response_id: str) -> dict:
        """Process audio data: save to file and return metadata with path."""
        audio_info = {
            "format": getattr(audio_data, "format", "wav"),
            "sample_rate": getattr(audio_data, "sample_rate", 22050),
            "transcript": getattr(audio_data, "transcript", None),
        }

        audio_base64 = getattr(audio_data, "data", None)
        if not audio_base64:
            return audio_info

        if self.output_audio_dir:
            try:
                audio_bytes = base64.b64decode(audio_base64)
                filename = f"{response_id}.wav"
                filepath = os.path.join(self.output_audio_dir, filename)

                with open(filepath, "wb") as f:
                    f.write(audio_bytes)

                audio_info["path"] = filepath
                audio_info["size_bytes"] = len(audio_bytes)
                LOG.info(f"Saved audio: {filepath} ({len(audio_bytes)} bytes)")
            except Exception as e:
                LOG.warning(f"Failed to save audio: {e}")
                audio_info["data"] = audio_base64
        else:
            audio_info["data"] = audio_base64

        return audio_info
