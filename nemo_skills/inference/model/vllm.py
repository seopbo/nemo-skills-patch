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

import requests

from nemo_skills.utils import get_logger_name

from .base import BaseModel
from .utils import ServerTokenizer

LOG = logging.getLogger(get_logger_name(__file__))


def audio_file_to_base64(audio_file_path: str) -> str:
    """Encodes an audio file into a base64 string."""
    with open(audio_file_path, "rb") as audio_file:
        audio_content = audio_file.read()
        return base64.b64encode(audio_content).decode("utf-8")


def load_audio_file(audio_file_path: str):
    """Load audio file and return array and sampling rate."""
    import soundfile as sf

    audio_array, sampling_rate = sf.read(audio_file_path)
    return audio_array, sampling_rate


def chunk_audio(audio_array, sampling_rate, chunk_duration_sec=30):
    """Chunk audio array into segments of specified duration.

    Args:
        audio_array: Audio data as numpy array
        sampling_rate: Sampling rate in Hz
        chunk_duration_sec: Duration of each chunk in seconds

    Returns:
        List of audio chunks
    """
    import numpy as np

    chunk_samples = int(chunk_duration_sec * sampling_rate)
    num_chunks = int(np.ceil(len(audio_array) / chunk_samples))

    chunks = []
    for i in range(num_chunks):
        start = i * chunk_samples
        end = min((i + 1) * chunk_samples, len(audio_array))
        chunks.append(audio_array[start:end])

    return chunks


def save_audio_chunk_to_base64(audio_chunk, sampling_rate):
    """Save audio chunk to temporary file and convert to base64.

    Args:
        audio_chunk: Audio data as numpy array
        sampling_rate: Sampling rate in Hz

    Returns:
        Base64 encoded audio string
    """
    import tempfile

    import soundfile as sf

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        sf.write(tmp_path, audio_chunk, sampling_rate)

    try:
        # Read and encode
        with open(tmp_path, "rb") as f:
            audio_content = f.read()
            encoded = base64.b64encode(audio_content).decode("utf-8")
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return encoded


class VLLMModel(BaseModel):
    def __init__(
        self,
        data_dir: str = "",
        enable_audio_chunking: bool = True,
        audio_chunk_task_types: list[str] | None = None,
        chunk_audio_threshold_sec: int = 30,
        **kwargs,
    ):
        """Initialize VLLMModel with audio chunking support.

        Args:
            data_dir: Base directory for audio files
            enable_audio_chunking: Master switch for audio chunking
            audio_chunk_task_types: If None, chunk all task types; if specified, only chunk these
            chunk_audio_threshold_sec: Audio duration threshold for chunking
            **kwargs: Other parameters passed to BaseModel
        """
        self.data_dir = data_dir
        self.enable_audio_chunking = enable_audio_chunking
        self.audio_chunk_task_types = audio_chunk_task_types
        self.chunk_audio_threshold_sec = chunk_audio_threshold_sec
        super().__init__(**kwargs)

    def _get_tokenizer_endpoint(self):
        """
        Returns the tokenizer endpoint if available, otherwise returns None.
        """
        tokenize_url = self.base_url.replace("/v1", "/tokenize")
        payload = {"prompt": "Test prompt"}

        try:
            response = requests.post(tokenize_url, json=payload)
            if response.status_code == 200:
                LOG.info(f"Tokenize endpoint is available! - {tokenize_url}")
                return ServerTokenizer(tokenize_url)
            else:
                return None
        except requests.exceptions.RequestException:
            return None

    def content_text_to_list(self, message):
        """Convert message content with audio to proper list format.

        Handles 'audio' or 'audios' keys in messages and converts them to
        base64-encoded audio_url content items.

        CRITICAL: Audio must come BEFORE text for Qwen models to transcribe correctly.
        """
        if "audio" in message or "audios" in message:
            content = message["content"]
            if isinstance(content, str):
                message["content"] = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                message["content"] = content
            else:
                raise TypeError(str(content))

        audio_items = []

        if "audio" in message:
            audio = message["audio"]
            base64_audio = audio_file_to_base64(os.path.join(self.data_dir, audio["path"]))
            audio_message = {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{base64_audio}"}}
            audio_items.append(audio_message)
        elif "audios" in message:
            for audio in message["audios"]:
                base64_audio = audio_file_to_base64(os.path.join(self.data_dir, audio["path"]))
                audio_message = {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{base64_audio}"}}
                audio_items.append(audio_message)

        # Insert audio items at the BEGINNING of content list (before text)
        if audio_items:
            message["content"] = audio_items + message["content"]

        return message

    def _preprocess_messages_for_model(self, messages: list[dict]) -> list[dict]:
        """Preprocess messages based on model-specific requirements.

        Remove /no_think suffix from system message as many models don't
        recognize it and it degrades performance (especially Qwen).
        """
        processed_messages = []
        for msg in messages:
            msg_copy = msg.copy()

            if msg_copy.get("role") == "system" and isinstance(msg_copy.get("content"), str):
                content = msg_copy["content"]
                if "/no_think" in content:
                    LOG.info(f"[PREPROCESS] BEFORE: '{content}'")
                    content = content.replace(" /no_think", "").replace("/no_think", "")
                    msg_copy["content"] = content.strip()
                    LOG.info(f"[PREPROCESS] AFTER: '{msg_copy['content']}'")

            processed_messages.append(msg_copy)

        return processed_messages

    def _build_request_body(self, top_k, min_p, repetition_penalty, extra_body: dict = None):
        full_extra_body = {
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "spaces_between_special_tokens": False,
        }

        if top_k > 0:
            full_extra_body["top_k"] = top_k

        if extra_body:
            full_extra_body.update(extra_body)

        return full_extra_body

    def _build_completion_request_params(
        self,
        prompt: str,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = None,
        top_logprobs: int | None = None,
        timeout: int | None = None,
        stop_phrases: list[str] | None = None,
        stream: bool = False,
        reasoning_effort: str | None = None,
        extra_body: dict = None,
        tools: list[dict] | None = None,
    ) -> dict:
        assert reasoning_effort is None, "reasoning_effort is not supported for text completion requests"
        assert tools is None, "tools are not supported for text completion requests"
        return {
            "prompt": prompt,
            "max_tokens": tokens_to_generate,
            "temperature": temperature,
            "top_p": top_p,
            "seed": random_seed,
            "stop": stop_phrases or None,
            "logprobs": top_logprobs,
            "stream": stream,
            "echo": False,
            "skip_special_tokens": False,
            "n": 1,
            "logit_bias": None,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "timeout": timeout,
            "extra_body": self._build_request_body(top_k, min_p, repetition_penalty, extra_body=extra_body),
        }

    async def generate_async(
        self,
        prompt: str | list[dict] | None = None,
        tokens_to_generate: int | None = None,
        task_type: str = None,
        **kwargs,
    ) -> dict:
        """Generate with automatic audio chunking for long audio files.

        This override checks if the prompt (messages) contains long audio.
        If so, it chunks the audio, processes each chunk separately, and aggregates results.
        """
        if isinstance(prompt, list):
            messages = prompt
            needs_chunking, audio_path, duration = self._needs_audio_chunking(messages, task_type)

            if needs_chunking:
                audio_array, sampling_rate = load_audio_file(audio_path)
                chunks = chunk_audio(audio_array, sampling_rate, self.chunk_audio_threshold_sec)

                chunk_results = []
                for chunk_idx, audio_chunk in enumerate(chunks):
                    chunk_messages = []
                    for msg in messages:
                        msg_copy = msg.copy()
                        if msg_copy.get("role") == "user" and ("audio" in msg_copy or "audios" in msg_copy):
                            chunk_base64 = save_audio_chunk_to_base64(audio_chunk, sampling_rate)

                            content = msg_copy.get("content", "")
                            if isinstance(content, str):
                                msg_copy["content"] = [{"type": "text", "text": content}]

                            # Add audio chunk at the beginning (before text)
                            msg_copy["content"] = [
                                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{chunk_base64}"}}
                            ] + msg_copy["content"]

                            # Remove original audio fields to avoid double processing
                            msg_copy.pop("audio", None)
                            msg_copy.pop("audios", None)

                        chunk_messages.append(msg_copy)

                    chunk_messages = self._preprocess_messages_for_model(chunk_messages)

                    result = await super().generate_async(
                        prompt=chunk_messages, tokens_to_generate=tokens_to_generate, **kwargs
                    )

                    generation = result.get("generation", "")
                    chunk_results.append(generation.strip())

                aggregated_text = " ".join(chunk_results)

                # Return result with aggregated generation
                # Use the last chunk's result structure but replace generation
                if result:
                    final_result = result.copy()
                    final_result["generation"] = aggregated_text
                    final_result["num_audio_chunks"] = len(chunks)
                    final_result["audio_duration"] = duration
                else:
                    final_result = {
                        "generation": aggregated_text,
                        "num_audio_chunks": len(chunks),
                        "audio_duration": duration,
                    }

                return final_result

        # Default behavior for non-chunked audio or non-list prompts
        return await super().generate_async(prompt=prompt, tokens_to_generate=tokens_to_generate, **kwargs)

    def _needs_audio_chunking(self, messages: list[dict], task_type: str = None) -> tuple[bool, str, float]:
        """Check if audio in messages needs chunking.

        Modified to support all task types by default, with optional filtering.

        Returns:
            Tuple of (needs_chunking, audio_path, duration)
        """
        if not self.enable_audio_chunking:
            return False, None, 0.0

        # Check if task type should be chunked (if filter is specified)
        if self.audio_chunk_task_types is not None:
            if task_type not in self.audio_chunk_task_types:
                return False, None, 0.0

        # Find audio in messages
        for msg in messages:
            if msg.get("role") == "user":
                audio_info = msg.get("audio") or (msg.get("audios", [{}])[0] if msg.get("audios") else {})
                if audio_info and "path" in audio_info:
                    audio_path = os.path.join(self.data_dir, audio_info["path"])

                    if not os.path.exists(audio_path):
                        return False, None, 0.0

                    # Load audio to check duration
                    try:
                        audio_array, sampling_rate = load_audio_file(audio_path)
                        duration = len(audio_array) / sampling_rate

                        if duration > self.chunk_audio_threshold_sec:
                            return True, audio_path, duration
                    except Exception:
                        pass

        return False, None, 0.0

    def _build_chat_request_params(
        self,
        messages: list[dict],
        stream: bool,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
        timeout: int | None = None,
        top_logprobs: int | None = None,
        reasoning_effort: str | None = None,
        tools: list[dict] | None = None,
        extra_body: dict = None,
    ) -> dict:
        # Preprocess messages for model-specific requirements (e.g., remove /no_think for Qwen)
        messages = self._preprocess_messages_for_model(messages)
        messages = [self.content_text_to_list(message) for message in messages]
        request = {
            "messages": messages,
            "max_tokens": tokens_to_generate,
            "temperature": temperature,
            "top_p": top_p,
            "seed": random_seed,
            "stop": stop_phrases or None,
            "logprobs": top_logprobs is not None,
            "top_logprobs": top_logprobs,
            "n": 1,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": stream,
            "timeout": timeout,
            "extra_body": self._build_request_body(top_k, min_p, repetition_penalty, extra_body=extra_body),
            "tools": tools,
        }
        if reasoning_effort:
            request["allowed_openai_params"] = ["reasoning_effort"]
            request["reasoning_effort"] = reasoning_effort
        return request

    def _build_responses_request_params(self, input, **kwargs) -> dict:
        # Parameters are the same as chat completion request params
        # For now, we hack this by renaming messages to input
        # Until we need more parameters for responses API
        responses_params = self._build_chat_request_params(messages=input, **kwargs)
        responses_params["input"] = responses_params.pop("messages")
        return responses_params
