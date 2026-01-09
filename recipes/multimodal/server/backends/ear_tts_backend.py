# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""EAR TTS backend using NemotronVoiceChat's TTS model for text-to-speech synthesis."""

import io
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from omegaconf import DictConfig, OmegaConf

from .base import BackendConfig, GenerationRequest, GenerationResult
from .magpie_tts_backend import MagpieTTSBackend, MagpieTTSConfig

# TTS constants
TTS_SAMPLE_RATE = 22050
FRAME_SIZE_SEC = 0.08  # 80ms per frame
DEFAULT_CODEC_TOKEN_HISTORY_SIZE = 60
SILENCE_THRESHOLD = 0.1  # Max magnitude threshold for silence detection
SILENCE_DURATION_SEC = 2.0  # Stop if last N seconds are silent


@dataclass
class EarTTSConfig(MagpieTTSConfig):
    """Configuration for EAR TTS backend - extends MagpieTTSConfig."""

    # EAR TTS specific paths
    tts_checkpoint_path: Optional[str] = None  # Path to TTS checkpoint (safetensors)
    speaker_reference: Optional[str] = None  # Speaker reference audio for voice cloning
    config_path: Optional[str] = None  # Optional YAML config path

    # TTS parameters
    codec_token_history_size: int = DEFAULT_CODEC_TOKEN_HISTORY_SIZE
    guidance_enabled: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EarTTSConfig":
        known_fields = {
            "model_path",
            "device",
            "dtype",
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "codec_model_path",
            "use_cfg",
            "cfg_scale",
            "max_decoder_steps",
            "use_local_transformer",
            "output_sample_rate",
            "hparams_file",
            "checkpoint_file",
            "legacy_codebooks",
            "legacy_text_conditioning",
            "hparams_from_wandb",
            # EAR TTS specific
            "tts_checkpoint_path",
            "speaker_reference",
            "config_path",
            "codec_token_history_size",
            "guidance_enabled",
        }
        known = {k: v for k, v in d.items() if k in known_fields}
        extra = {k: v for k, v in d.items() if k not in known_fields}
        return cls(**known, extra_config=extra)


class EarTTSBackend(MagpieTTSBackend):
    """
    EAR TTS backend using NemotronVoiceChat's TTS model.

    Inherits from MagpieTTSBackend and overrides load_model() and generate()
    to use the EAR TTS model instead of MagpieTTS.
    """

    @property
    def name(self) -> str:
        return "ear_tts"

    def __init__(self, config: BackendConfig):
        # Convert to EarTTSConfig
        if isinstance(config, EarTTSConfig):
            self.ear_config = config
        else:
            self.ear_config = EarTTSConfig.from_dict(
                {
                    **{
                        k: getattr(config, k)
                        for k in ["model_path", "device", "dtype", "max_new_tokens", "temperature", "top_p", "top_k"]
                        if hasattr(config, k)
                    },
                    **config.extra_config,
                }
            )

        # Call grandparent __init__ to skip MagpieTTSBackend's init
        from .base import InferenceBackend

        InferenceBackend.__init__(self, self.ear_config)
        self.tts_config = self.ear_config

        self._model = None
        self._model_cfg = None
        self._tokenizer = None

        # TTS state
        self.first_context_subword_id = None
        self.generation_config = None
        self.first_tts_code_input = None
        self.first_tts_past_key_values_input = None
        self.target_sample_rate = TTS_SAMPLE_RATE
        self.target_fps = None

    def _clone_cache(self, cache):
        """Deep clone cache structures."""
        if cache is None:
            return None
        if isinstance(cache, torch.Tensor):
            return cache.detach().clone()
        if isinstance(cache, (list, tuple)):
            return type(cache)(self._clone_cache(x) for x in cache)
        if isinstance(cache, dict):
            return {k: self._clone_cache(v) for k, v in cache.items()}
        if hasattr(cache, "__dict__"):
            import copy

            return copy.deepcopy(cache)
        return cache

    def load_model(self) -> None:
        """Load the EAR TTS model from NemotronVoiceChat."""
        import sys

        from safetensors.torch import load_file

        print(f"[EarTTS] Loading model from {self.config.model_path}...")

        # Clear cached nemo modules FIRST to force reimport from our paths
        nemo_modules = [k for k in sys.modules.keys() if k.startswith("nemo")]
        for mod in nemo_modules:
            del sys.modules[mod]
        print(f"[EarTTS] Cleared {len(nemo_modules)} cached nemo modules")

        # Ensure code path is FIRST in sys.path (for speechlm2 module)
        code_path = os.environ.get("UNIFIED_SERVER_CODE_PATH", "")
        print(f"[EarTTS] UNIFIED_SERVER_CODE_PATH = '{code_path}'")
        if code_path:
            paths = [p for p in code_path.split(":") if p]
            # Remove existing entries and re-add at front
            for path in paths:
                while path in sys.path:
                    sys.path.remove(path)
            for path in reversed(paths):
                sys.path.insert(0, path)
                print(f"[EarTTS] Added to sys.path: {path}")
        else:
            print("[EarTTS] WARNING: No code path found in env!")

        # Debug: show current path
        print(f"[EarTTS] sys.path (first 5): {sys.path[:5]}")

        try:
            from nemo.collections.speechlm2.models.nemotron_voicechat import NemotronVoiceChat
            from nemo.collections.speechlm2.parts.pretrained import set_model_dict_for_partial_init
        except ImportError as e:
            raise RuntimeError(f"Failed to import NemotronVoiceChat. Error: {e}")

        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        # Load config
        tts_path = self.ear_config.tts_checkpoint_path or self.config.model_path
        config_file = os.path.join(tts_path, "config.json")
        print(f"[EarTTS] Loading config: {config_file}")

        with open(config_file, "r") as f:
            cfg = DictConfig(json.load(f))

        # Set speaker reference
        speaker_ref = self.ear_config.speaker_reference
        if not speaker_ref and self.ear_config.config_path:
            yaml_cfg = OmegaConf.load(self.ear_config.config_path)
            speaker_ref = yaml_cfg.get("model", {}).get("inference_speaker_reference")
        if speaker_ref:
            if "model" not in cfg:
                cfg.model = {}
            cfg.model.inference_speaker_reference = speaker_ref

        self._model_cfg = cfg

        # Disable pretrained model loading
        if hasattr(cfg.model, "speech_generation") and hasattr(cfg.model.speech_generation, "model"):
            cfg.model.speech_generation.model.pretrained_model = None
        if hasattr(cfg.model, "stt") and hasattr(cfg.model.stt, "model"):
            cfg.model.stt.model.pretrained_s2s_model = None

        # Initialize and load model
        print("[EarTTS] Initializing model structure...")
        self._model = NemotronVoiceChat(OmegaConf.to_container(cfg, resolve=True))

        safetensors_path = os.path.join(tts_path, "model.safetensors")
        if os.path.exists(safetensors_path):
            print(f"[EarTTS] Loading TTS weights from: {tts_path}")
            state_dict = load_file(safetensors_path)
            tts_only = {k: v for k, v in state_dict.items() if k.startswith("tts_model.")}
            print(f"[EarTTS] Loading {len(tts_only)} TTS parameters")
            tts_only = set_model_dict_for_partial_init(tts_only, self._model.state_dict())
            self._model.load_state_dict(tts_only, strict=False)
        else:
            raise ValueError(f"TTS weights not found at {safetensors_path}")

        self._model.to(self.config.device)
        self._model.eval()
        self._tokenizer = self._model.stt_model.tokenizer

        if hasattr(self._model, "tts_model"):
            self.target_fps = self._model.tts_model.target_fps
            self.target_sample_rate = self._model.tts_model.target_sample_rate
            print(f"[EarTTS] TTS: fps={self.target_fps}, sample_rate={self.target_sample_rate}")
            self._prepare_tts_initial_state()

        self._is_loaded = True
        print("[EarTTS] Model loaded successfully")

    def _prepare_tts_initial_state(self):
        """Prepare TTS warmup state with speaker reference."""
        from nemo.collections.audio.parts.utils.resampling import resample
        from nemo.collections.speechlm2.parts.precision import fp32_precision

        if not hasattr(self._model, "tts_model"):
            return

        speaker_ref = self._model_cfg.model.get("inference_speaker_reference") if self._model_cfg else None
        if not speaker_ref:
            speaker_ref = self.ear_config.speaker_reference
        if not speaker_ref:
            print("[EarTTS] Warning: No speaker reference")
            return

        print(f"[EarTTS] Preparing TTS with speaker: {speaker_ref}")

        with fp32_precision():
            speaker_audio, speaker_sr = torchaudio.load(speaker_ref)
            speaker_audio = resample(speaker_audio, speaker_sr, self._model.tts_model.target_sample_rate)

        speaker_audio = speaker_audio.to(self.config.device)
        speaker_audio_lens = torch.tensor([speaker_audio.size(1)], device=self.config.device).long()

        self._model.tts_model.set_init_inputs(speaker_audio=speaker_audio, speaker_audio_lens=speaker_audio_lens)
        init_inputs = self._model.tts_model.get_init_inputs(B=1)
        self.generation_config = self._model.tts_model._get_generation_config(
            guidance_enabled=self.ear_config.guidance_enabled
        )
        init_inputs.update({"use_cache": True, "past_key_values": None, "guidance_enabled": True})

        with torch.no_grad():
            outputs = self._model.tts_model.tts_model(**init_inputs)
            code = init_inputs["code"][:, -1:]

        self.first_context_subword_id = init_inputs["subword_ids"][:, -1].unsqueeze(-1)
        self.first_tts_code_input = code.detach().clone()
        self.first_tts_past_key_values_input = self._clone_cache(outputs.past_key_values)
        print("[EarTTS] TTS warmup state prepared")

    @torch.no_grad()
    def _synthesize_text(self, text: str) -> Optional[np.ndarray]:
        """Synthesize audio from text using EAR TTS.

        Generates audio frames until the last 2 seconds have max magnitude below threshold,
        indicating the model has finished speaking.
        """
        from nemo.collections.speechlm2.parts.precision import fp32_precision

        if not text or not self.generation_config:
            return None

        device = self.config.device
        token_ids = self._tokenizer.text_to_ids(text)
        if not token_ids:
            return None

        num_tokens = len(token_ids)
        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

        # Max frames: generous upper bound (10x tokens should be plenty)
        max_frames = num_tokens * 10

        # Samples needed to check for silence (2 seconds)
        sample_rate = self.target_sample_rate or TTS_SAMPLE_RATE
        samples_for_silence_check = int(sample_rate * SILENCE_DURATION_SEC)

        # Initialize TTS state
        past_key_values = self._clone_cache(self.first_tts_past_key_values_input)
        code = self.first_tts_code_input.detach().clone()
        codec_history_size = self.ear_config.codec_token_history_size
        audio_toks_buffer = (
            self._model.tts_model.codec_silence_tokens.view(1, 1, -1).expand(-1, codec_history_size, -1).to(device)
        )

        audio_segments = []
        samples_per_frame = int(float(sample_rate) * FRAME_SIZE_SEC)
        total_samples = 0

        for frame_idx in range(max_frames):
            # Cycle through tokens, repeating the last token after we've used all
            token_idx = min(frame_idx, num_tokens - 1)
            current_subword_id = token_tensor[:, token_idx].unsqueeze(-1)

            if frame_idx == 0:
                prev_subword_id = self.first_context_subword_id
            else:
                prev_token_idx = min(frame_idx - 1, num_tokens - 1)
                prev_subword_id = token_tensor[:, prev_token_idx].unsqueeze(-1)

            code, past_key_values = self._model.tts_model.infer_codes_one_step(
                current_subword_id=current_subword_id,
                prev_subword_id=prev_subword_id,
                current_subword_mask=torch.ones(1, 1, device=device, dtype=torch.bool),
                prev_audio_tokens=code,
                past_key_values=past_key_values,
                guidance_enabled=self.ear_config.guidance_enabled,
                generation_config=self.generation_config,
                ignore_eos_flag_stop=True,
            )

            audio_toks_buffer = torch.cat([audio_toks_buffer[:, 1:], code], dim=1)

            with fp32_precision():
                decoded_audio, _ = self._model.tts_model.audio_codec.decode(
                    audio_toks_buffer, torch.tensor([codec_history_size], dtype=torch.long, device=device)
                )
            frame_audio = decoded_audio[:, :, -samples_per_frame:]
            audio_segments.append(frame_audio)
            total_samples += samples_per_frame

            # Check for silence after we have enough samples
            if total_samples >= samples_for_silence_check:
                # Get last 2 seconds of audio
                recent_audio = torch.cat(audio_segments, dim=-1)[:, :, -samples_for_silence_check:]
                max_magnitude = recent_audio.abs().max().item()

                if max_magnitude < SILENCE_THRESHOLD:
                    # Silence detected - stop generating
                    break

        if audio_segments:
            audio_tensor = torch.cat(audio_segments, dim=-1)

            # Trim trailing silence
            audio_np = audio_tensor.float().cpu().numpy().squeeze()

            # Find where audio becomes silent (from the end)
            window_size = int(sample_rate * 0.1)  # 100ms window
            for trim_point in range(len(audio_np) - window_size, 0, -window_size):
                window_max = np.abs(audio_np[trim_point : trim_point + window_size]).max()
                if window_max >= SILENCE_THRESHOLD:
                    # Found non-silent audio, trim after this point + small buffer
                    audio_np = audio_np[: trim_point + window_size * 2]
                    break

            max_val = np.abs(audio_np).max()
            if max_val > 0:
                audio_np = audio_np / max_val * 0.95
            return audio_np
        return None

    def generate(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        """Generate audio from text requests."""
        if not self._is_loaded:
            return [GenerationResult(error="Model not loaded", request_id=r.request_id) for r in requests]
        if not requests:
            return []

        results = []
        for req in requests:
            start_time = time.time()
            try:
                parsed = self._extract_json(req.text)
                text = parsed.get("text", "")

                if not text:
                    results.append(GenerationResult(error="No text provided", request_id=req.request_id))
                    continue

                audio_np = self._synthesize_text(text)
                if audio_np is None:
                    results.append(GenerationResult(error="Failed to synthesize audio", request_id=req.request_id))
                    continue

                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, audio_np, self.target_sample_rate, format="WAV")
                elapsed_ms = (time.time() - start_time) * 1000
                audio_duration = len(audio_np) / self.target_sample_rate

                results.append(
                    GenerationResult(
                        text=text,
                        audio_bytes=wav_buffer.getvalue(),
                        audio_sample_rate=self.target_sample_rate,
                        audio_format="wav",
                        request_id=req.request_id,
                        generation_time_ms=elapsed_ms,
                        debug_info={
                            "audio_duration_sec": audio_duration,
                            "rtf": elapsed_ms / 1000 / audio_duration if audio_duration > 0 else 0,
                        },
                    )
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                results.append(GenerationResult(error=str(e), request_id=req.request_id))
        return results

    def health_check(self) -> Dict[str, Any]:
        """Return health status."""
        h = super().health_check()
        if self._is_loaded:
            h.update(
                {
                    "sample_rate": self.target_sample_rate,
                    "fps": self.target_fps,
                    "tts_enabled": self.generation_config is not None,
                    "speaker_reference": self.ear_config.speaker_reference,
                }
            )
        return h


class EarTTSBatchBackend(EarTTSBackend):
    """
    Optimized EAR TTS backend that decodes audio only once at the end.

    This version generates all codes first (token-by-token), then decodes
    the entire sequence in one batch operation - significantly faster than
    decoding after every token.
    """

    @property
    def name(self) -> str:
        return "ear_tts_batch"

    @torch.no_grad()
    def _synthesize_text(self, text: str) -> Optional[np.ndarray]:
        """Synthesize audio from text - optimized batch decoding version.

        Generates 10x tokens worth of frames, decodes all at once, then trims trailing silence.
        """
        from nemo.collections.speechlm2.parts.precision import fp32_precision

        if not text or not self.generation_config:
            return None

        device = self.config.device
        token_ids = self._tokenizer.text_to_ids(text)
        if not token_ids:
            return None

        num_tokens = len(token_ids)
        max_frames = num_tokens * 10  # Generate more frames than tokens
        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

        # Initialize TTS state
        past_key_values = self._clone_cache(self.first_tts_past_key_values_input)
        code = self.first_tts_code_input.detach().clone()

        # Generate ALL codes first (no decoding in the loop)
        all_codes = []

        for frame_idx in range(max_frames):
            # Cycle through tokens, repeating the last token after we've used all
            token_idx = min(frame_idx, num_tokens - 1)
            current_subword_id = token_tensor[:, token_idx].unsqueeze(-1)

            if frame_idx == 0:
                prev_subword_id = self.first_context_subword_id
            else:
                prev_token_idx = min(frame_idx - 1, num_tokens - 1)
                prev_subword_id = token_tensor[:, prev_token_idx].unsqueeze(-1)

            code, past_key_values = self._model.tts_model.infer_codes_one_step(
                current_subword_id=current_subword_id,
                prev_subword_id=prev_subword_id,
                current_subword_mask=torch.ones(1, 1, device=device, dtype=torch.bool),
                prev_audio_tokens=code,
                past_key_values=past_key_values,
                guidance_enabled=self.ear_config.guidance_enabled,
                generation_config=self.generation_config,
                ignore_eos_flag_stop=True,
            )
            all_codes.append(code)

        # Decode ALL codes at once at the end
        if all_codes:
            all_codes_tensor = torch.cat(all_codes, dim=1)  # [B, max_frames, codebook_dim]
            len_codes = torch.tensor([max_frames], dtype=torch.long, device=device)

            with fp32_precision():
                decoded_audio, _ = self._model.tts_model.audio_codec.decode(all_codes_tensor, len_codes)

            audio_np = decoded_audio.float().cpu().numpy().squeeze()

            # Trim trailing silence
            sample_rate = self.target_sample_rate or TTS_SAMPLE_RATE
            window_size = int(sample_rate * 0.1)  # 100ms window
            for trim_point in range(len(audio_np) - window_size, 0, -window_size):
                window_max = np.abs(audio_np[trim_point : trim_point + window_size]).max()
                if window_max >= SILENCE_THRESHOLD:
                    # Found non-silent audio, trim after this point + small buffer
                    audio_np = audio_np[: trim_point + window_size * 2]
                    break

            max_val = np.abs(audio_np).max()
            if max_val > 0:
                audio_np = audio_np / max_val * 0.95
            return audio_np

        return None
