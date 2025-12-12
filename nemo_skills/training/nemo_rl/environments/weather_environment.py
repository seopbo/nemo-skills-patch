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
Weather Prediction Environment for NeMo-RL.

This is a minimal environment example that demonstrates how to create custom
reward functions for NeMo-RL training. The environment follows the same pattern
as MathEnvironment - it only computes rewards based on model generations.

Generation is handled by NeMo-RL's policy_generation, which can be configured
to use the NeMo-Skills proxy server via OpenAI-compatible endpoints:

    policy:
      generation:
        backend: "openai"  # or use vllm pointing to NeMo-Skills proxy
        openai_cfg:
          base_url: "http://localhost:7000/v1"  # NeMo-Skills proxy
          model: "nemo-skills"

This way:
1. NeMo-RL handles all generation using its existing infrastructure
2. Generations go through NeMo-Skills proxy for prompt formatting
3. This environment only computes rewards from the generated responses
"""

import logging
from typing import Any, Optional, TypedDict

import ray
import torch
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.utils import chunk_list_to_workers


class WeatherEnvConfig(TypedDict):
    num_workers: int
    stop_strings: Optional[list[str]]


class WeatherEnvironmentMetadata(TypedDict):
    expected_weather: str  # Ground truth weather prediction
    location: str
    date: str


@ray.remote
class WeatherVerifyWorker:
    """Worker that evaluates weather predictions."""

    def __init__(self) -> None:
        logging.getLogger("weather_verify").setLevel(logging.INFO)

    def verify(
        self,
        pred_responses: list[str],
        ground_truths: list[str],
    ) -> list[tuple[float, str]]:
        """Verify the correctness of weather predictions.

        Args:
            pred_responses: Predicted weather responses from the LLM
            ground_truths: Expected weather conditions

        Returns:
            List of (score, extracted_prediction) tuples
        """
        results = []
        for response, ground_truth in zip(pred_responses, ground_truths):
            try:
                extracted_prediction = self._extract_weather_prediction(response)
                score = self._compute_weather_score(extracted_prediction, ground_truth)
            except Exception as e:
                logging.warning(f"Error verifying weather prediction: {e}")
                score = 0.0
                extracted_prediction = None

            results.append((score, extracted_prediction))
        return results

    def _extract_weather_prediction(self, response: str) -> str:
        """Extract the weather prediction from the model response."""
        response_lower = response.lower()

        weather_conditions = [
            "sunny",
            "cloudy",
            "rainy",
            "stormy",
            "snowy",
            "partly cloudy",
            "overcast",
            "clear",
            "foggy",
            "windy",
        ]

        for condition in weather_conditions:
            if condition in response_lower:
                return condition

        # If no condition found, return the last sentence as the prediction
        sentences = response.strip().split(".")
        return sentences[-1].strip() if sentences else response

    def _compute_weather_score(self, prediction: str, ground_truth: str) -> float:
        """Compute similarity score between prediction and ground truth."""
        if prediction is None:
            return 0.0

        prediction_lower = prediction.lower().strip()
        ground_truth_lower = ground_truth.lower().strip()

        # Exact match
        if prediction_lower == ground_truth_lower:
            return 1.0

        # Partial match (ground truth contained in prediction)
        if ground_truth_lower in prediction_lower:
            return 0.8

        # Check for semantic similarity (simplified)
        similar_pairs = [
            ({"sunny", "clear"}, 0.7),
            ({"cloudy", "overcast", "partly cloudy"}, 0.6),
            ({"rainy", "stormy"}, 0.5),
        ]

        for similar_set, score in similar_pairs:
            if prediction_lower in similar_set and ground_truth_lower in similar_set:
                return score

        return 0.0


@ray.remote(max_restarts=-1, max_task_retries=-1)
class WeatherEnvironment(EnvironmentInterface):
    """
    Custom environment for weather prediction RL training.

    This environment follows the same pattern as MathEnvironment:
    - Receives model generations from NeMo-RL's policy_generation
    - Computes rewards based on prediction accuracy
    - Returns observations, rewards, and termination flags

    To use NeMo-Skills proxy for generation enrichment, configure NeMo-RL's
    generation backend to point to the NeMo-Skills server's OpenAI-compatible
    endpoints (/v1/chat/completions or /v1/completions).
    """

    def __init__(self, cfg: WeatherEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]

        # Create worker pool for parallel verification
        self.workers = [
            WeatherVerifyWorker.options(runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}).remote()
            for _ in range(self.num_workers)
        ]

    def shutdown(self) -> None:
        """Shutdown all workers."""
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: list[list[dict[str, str]]],
        metadata: list[WeatherEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """Run a step in the weather environment.

        Args:
            message_log_batch: Batch of OpenAI-API-like message logs from policy_generation
            metadata: Metadata containing expected_weather for evaluation

        Returns:
            EnvironmentReturn with observations, rewards, and termination flags
        """
        # Extract assistant responses from conversations
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [msg["content"] for msg in conversation if msg["role"] == "assistant"]
            assistant_response_batch.append("".join(assistant_responses))

        # Get ground truths from metadata
        ground_truths = [m["expected_weather"] for m in metadata]

        # Chunk work across verification workers
        chunked_responses = chunk_list_to_workers(assistant_response_batch, self.num_workers)
        chunked_ground_truths = chunk_list_to_workers(ground_truths, self.num_workers)

        # Process in parallel
        futures = [
            self.workers[i].verify.remote(chunk, gt_chunk)
            for i, (chunk, gt_chunk) in enumerate(zip(chunked_responses, chunked_ground_truths))
        ]

        results = ray.get(futures)

        # Flatten results
        results = [item for sublist in results for item in sublist]
        scores, extracted_predictions = zip(*results) if results else ([], [])

        # Build observations
        observations = [
            {
                "role": "environment",
                "content": f"Weather prediction: {'correct' if score >= 0.8 else 'incorrect'}",
            }
            for score in scores
        ]

        # Build reward and done tensors
        rewards = torch.tensor(list(scores)).cpu()
        done = torch.ones_like(rewards).cpu()

        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
            answers=list(extracted_predictions),
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        """Compute metrics for the environment."""
        rewards = batch["rewards"]

        # Calculate accuracy at different thresholds
        accuracy_exact = (rewards == 1.0).float().mean().item()
        accuracy_relaxed = (rewards >= 0.5).float().mean().item()

        # Calculate average reward for correct predictions
        correct_mask = rewards >= 0.8
        if correct_mask.sum() > 0:
            avg_correct_gen_length = (
                (batch["generation_lengths"] - batch["prompt_lengths"])[correct_mask].float().mean().item()
            )
        else:
            avg_correct_gen_length = 0

        metrics = {
            "accuracy_exact": accuracy_exact,
            "accuracy_relaxed": accuracy_relaxed,
            "mean_reward": rewards.mean().item(),
            "fraction_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": avg_correct_gen_length,
        }

        return batch, metrics
