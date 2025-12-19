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

# copied and edited from https://github.com/NVIDIA/NeMo-RL/blob/ab1b638a499308caea022648daaf6994d390cbde/examples/run_grpo_math.py

import argparse
import copy
import importlib
import json
import os
import pprint
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import ray
from datasets import Dataset, load_dataset
from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec, TaskDataProcessFnCallable, TaskDataSpec
from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY, get_actor_python_env
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES, init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import setup_make_sequence_length_divisible_by

# NeMo-Gym imports (optional - only needed when using NeMo-Gym)
_NEMO_GYM_IMPORT_ERROR = None
try:
    from nemo_rl.environments.nemo_gym import (
        NemoGym,
        NemoGymConfig,
        setup_nemo_gym_config,
    )

    NEMO_GYM_AVAILABLE = True
except ImportError as e:
    NEMO_GYM_AVAILABLE = False
    _NEMO_GYM_IMPORT_ERROR = str(e)
    NemoGym = None  # type: ignore
    NemoGymConfig = None  # type: ignore
    setup_nemo_gym_config = None  # type: ignore


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


# ===============================================================================
#                             Custom Math Dataset (@nemo-skills)
# ===============================================================================


def load_jsonl_as_dataset(
    filepath: str,
    force_string: bool = False,
    keep_fields: Optional[list[str]] = None,
) -> Dataset:
    """
    Load a JSONL file and convert it to a Hugging Face Dataset.

    Args:
        filepath (str): Path to the .jsonl file.

    Returns:
        Dataset: Hugging Face Dataset object.
    """
    records: list[dict[str, Any]] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(obj)

    return Dataset.from_list(records)


def extract_dataset(split, dataset_path):
    if dataset_path is None:
        return None
    if not dataset_path.startswith("/"):
        original_ds = load_dataset(dataset_path, split=split)
    else:
        original_ds = load_jsonl_as_dataset(dataset_path)
    return original_ds


def format_passthrough(data):
    return {
        **data,
        # For v0.1 release, nemo rl datasets require a task_name key such that user can map a task processor per unique task.
        "task_name": "math",
    }


def prepare_math_dataset(split_ds):
    # Format the examples, removing original columns
    train_formatted = split_ds["train"].map(format_passthrough)
    val_raw = split_ds.get("validation", None)
    val_formatted = None if val_raw is None else val_raw.map(format_passthrough)

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class NeMoSkillsDataset:
    """Custom dataset class for NeMo Skills Math Environment."""

    def __init__(self, training_data, validation_data):
        """Initialize the dataset with training and validation data."""
        self.training_data = training_data
        self.validation_data = validation_data

        # Load the datasets
        self.formatted_ds = prepare_math_dataset(
            {
                "train": extract_dataset("train", training_data),
                "validation": extract_dataset("validation", validation_data),
            }
        )


# ===============================================================================
#                             Math Data Processor
# ===============================================================================
TokenizerType = PreTrainedTokenizerBase


@dataclass
class NSTaskDataSpec(TaskDataSpec):
    prompt_spec: dict[str, Any] | None = None


def ns_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: NSTaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Standard NeMo-Skills data processor for MathEnvironment."""
    prompt_spec = task_data_spec.prompt_spec
    extra_env_info = copy.deepcopy(datum_dict)

    prompt = get_prompt(
        prompt_config=prompt_spec["prompt_config"],
        tokenizer=tokenizer,
        examples_type=prompt_spec["examples_type"],
        config_dir=prompt_spec["config_dir"],
    )
    # we need to include system message here as roles are only used for masking
    # so prompt.fill can return a combined system + user message
    # if we use separate, it will have double BOS in the tokens!
    user_message = prompt.fill(datum_dict, format_as_string=True)
    message_log = [
        {
            "role": "user",
            "content": user_message,
            "token_ids": tokenizer([user_message], return_tensors="pt", add_special_tokens=False)["input_ids"][0],
        }
    ]

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # make smaller and mask out
        for chat_message in message_log:
            chat_message["token_ids"] = chat_message["token_ids"][: min(4, max_seq_length // len(message_log))]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict["task_name"],
    }
    return output


def create_nemo_gym_data_processor(agent_name: str = "math_with_judge_simple_agent"):
    """
    Create a NeMo-Skills data processor for NeMo-Gym integration.

    Args:
        agent_name: The name of the agent server to call (must match a top-level key
                    in the NeMo-Gym config, e.g., "math_with_judge_simple_agent")

    Returns:
        A data processor function configured for the specified agent.
    """

    def ns_data_processor_for_nemo_gym(
        datum_dict: dict[str, Any],
        task_data_spec: NSTaskDataSpec,
        tokenizer: TokenizerType,
        max_seq_length: int,
        idx: int,
    ) -> DatumSpec:
        """
        NeMo-Skills data processor for NeMo-Gym integration.

        This differs from the standard processor by:
        1. Adding responses_create_params for OpenAI Responses API format
        2. Adding agent_ref to specify which agent server to call
        3. Adding _rowidx for result ordering
        """
        prompt_spec = task_data_spec.prompt_spec
        extra_env_info = copy.deepcopy(datum_dict)

        prompt = get_prompt(
            prompt_config=prompt_spec["prompt_config"],
            tokenizer=tokenizer,
            examples_type=prompt_spec["examples_type"],
            config_dir=prompt_spec["config_dir"],
        )
        user_message = prompt.fill(datum_dict, format_as_string=True)
        message_log = [
            {
                "role": "user",
                "content": user_message,
                "token_ids": tokenizer([user_message], return_tensors="pt", add_special_tokens=False)["input_ids"][0],
            }
        ]

        length = sum(len(m["token_ids"]) for m in message_log)

        loss_multiplier = 1.0
        if length > max_seq_length:
            for chat_message in message_log:
                chat_message["token_ids"] = chat_message["token_ids"][: min(4, max_seq_length // len(message_log))]
            loss_multiplier = 0.0

        # NeMo-Gym specific fields (required by rollout_collection.py)
        # These fields are used by NemoGym.run_rollouts() to make agent calls
        extra_env_info["responses_create_params"] = {
            "input": [{"role": "user", "content": user_message}],
            "tools": [],  # No tools for basic math
        }
        # Agent ref tells NeMo-Gym which agent server to call
        # Must match a top-level key in the nemo_gym config (e.g., "math_with_judge_simple_agent")
        extra_env_info["agent_ref"] = {"name": agent_name}
        # Row index for result ordering after async processing
        extra_env_info["_rowidx"] = idx

        output: DatumSpec = {
            "message_log": message_log,
            "length": length,
            "extra_env_info": extra_env_info,
            "loss_multiplier": loss_multiplier,
            "idx": idx,
            "task_name": datum_dict["task_name"],  # Keep original task_name ("math")
        }
        return output

    return ns_data_processor_for_nemo_gym


def setup_data_only(
    tokenizer: TokenizerType,
    data_config: DataConfig,
    use_nemo_gym: bool = False,
    nemo_gym_agent_name: str = "math_with_judge_simple_agent",
) -> tuple[AllTaskProcessedDataset, Optional[AllTaskProcessedDataset]]:
    """
    Set up datasets without creating environments.
    Environments are created separately after policy_generation is available.

    Args:
        tokenizer: The tokenizer to use for encoding.
        data_config: Data configuration dict.
        use_nemo_gym: Whether to use NeMo-Gym for rollouts.
        nemo_gym_agent_name: The agent name to use in NeMo-Gym (must match a
            top-level key in the nemo_gym config, like "math_with_judge_simple_agent").
    """
    print("\n▶ Setting up data...")
    prompt_config = data_config["prompt"]

    # The data comes with task_name="math" from format_passthrough
    # We always use "math" as the task name key, but choose the processor based on env type
    task_name = "math"

    math_task_spec = NSTaskDataSpec(
        task_name=task_name,
        prompt_spec=prompt_config,
    )

    data = NeMoSkillsDataset(
        data_config["train_data_path"],
        data_config["val_data_path"],
    )

    # Choose processor based on environment type
    # Both processors handle "math" task name, but nemo_gym processor adds extra fields
    if use_nemo_gym:
        processor = create_nemo_gym_data_processor(agent_name=nemo_gym_agent_name)
        print(f"  Using NeMo-Gym data processor with agent: {nemo_gym_agent_name}")
    else:
        processor = ns_data_processor
        print("  Using standard MathEnvironment data processor")

    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = defaultdict(
        lambda: (math_task_spec, processor)
    )
    # Register for "math" task name since that's what the data has
    task_data_processors["math"] = (math_task_spec, processor)

    dataset = AllTaskProcessedDataset(
        data.formatted_ds["train"],
        tokenizer,
        math_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset: Optional[AllTaskProcessedDataset] = None
    if data.formatted_ds["validation"]:
        val_dataset = AllTaskProcessedDataset(
            data.formatted_ds["validation"],
            tokenizer,
            math_task_spec,
            task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )

    return dataset, val_dataset


def setup_math_environment(
    env_configs: dict[str, Any],
) -> dict[str, EnvironmentInterface]:
    """Set up the standard MathEnvironment."""
    env_cls_path = env_configs["math"].get(
        "env_cls",
        "nemo_skills.training.nemo_rl.environments.math_environment.MathEnvironment",
    )
    ACTOR_ENVIRONMENT_REGISTRY[env_cls_path] = PY_EXECUTABLES.SYSTEM

    module_name, class_name = env_cls_path.rsplit(".", 1)
    env_module = importlib.import_module(module_name)
    env_cls = getattr(env_module, class_name)

    math_env = env_cls.options(
        runtime_env={
            "py_executable": get_actor_python_env(env_cls_path),
            "env_vars": dict(os.environ),
        }
    ).remote(env_configs["math"])

    task_to_env: dict[str, EnvironmentInterface] = defaultdict(lambda: math_env)
    task_to_env["math"] = math_env
    return task_to_env


def setup_nemo_gym_environment(
    policy_generation,
    nemo_gym_config: dict[str, Any],
) -> dict[str, EnvironmentInterface]:
    """
    Set up NeMo-Gym environment using policy_generation's vLLM server URLs.

    This follows the canonical approach from run_grpo_nemo_gym.py:
    1. Use policy_generation.dp_openai_server_base_urls for the vLLM server URLs
    2. Pass the nemo_gym config as initial_global_config_dict
    3. The NemoGym class adds flat keys (policy_model_name, policy_base_url, etc.)

    IMPORTANT: The nemo_gym config should contain:
    - config_paths: List of paths to Gym server config files (relative to Gym repo)
      Example: ["responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
                "resources_servers/math_with_judge/configs/math_with_judge.yaml"]
    - Any inline overrides for server configs

    The agent_ref in the data must match a top-level key in the resolved config
    (e.g., "math_with_judge_simple_agent" from math_with_judge.yaml).
    """
    if not NEMO_GYM_AVAILABLE:
        raise ImportError(f"NeMo-Gym is not available: {_NEMO_GYM_IMPORT_ERROR}")

    # Get the vLLM server URLs from policy_generation
    # These are the actual OpenAI-compatible endpoints exposed by vLLM
    base_urls = policy_generation.dp_openai_server_base_urls
    model_name = policy_generation.cfg["model_name"]

    print(f"  ⚙️  Setting up NeMo-Gym with {len(base_urls)} vLLM server(s)")
    print(f"      Model: {model_name}")
    print(f"      Base URLs: {base_urls}")

    # The initial_global_config_dict is passed to NemoGym
    # NemoGym.__init__ will add: policy_model_name, policy_api_key, policy_base_url
    # See nemo_rl/environments/nemo_gym.py lines 51-59
    initial_global_config_dict = nemo_gym_config.copy() if nemo_gym_config else {}

    # Validate that config_paths are provided for NeMo-Gym server startup
    config_paths = initial_global_config_dict.get("config_paths", [])
    if not config_paths:
        print("  ⚠️  Warning: No config_paths provided in nemo_gym config.")
        print("     NeMo-Gym requires config_paths pointing to server config files.")
        print("     Example config_paths:")
        print("       - responses_api_models/vllm_model/configs/vllm_model_for_training.yaml")
        print("       - resources_servers/math_with_judge/configs/math_with_judge.yaml")
        print("     Without these, no agent/resources servers will be started.")
    else:
        print(f"      Config paths: {config_paths}")

    # Create NemoGymConfig (TypedDict with model_name, base_urls, initial_global_config_dict)
    nemo_gym_env_config: NemoGymConfig = {
        "model_name": model_name,
        "base_urls": base_urls,
        "initial_global_config_dict": initial_global_config_dict,
    }

    # Register NeMo-Gym environment
    nemo_gym_cls_path = "nemo_rl.environments.nemo_gym.NemoGym"
    ACTOR_ENVIRONMENT_REGISTRY[nemo_gym_cls_path] = PY_EXECUTABLES.SYSTEM

    nemo_gym_env = NemoGym.options(
        runtime_env={
            "py_executable": get_actor_python_env(nemo_gym_cls_path),
        }
    ).remote(nemo_gym_env_config)

    # Blocking wait for NeMo-Gym to spin up (like in run_grpo_nemo_gym.py)
    print("  ⏳ Waiting for NeMo-Gym environment to initialize...")
    ray.get(nemo_gym_env.health_check.remote())
    print("  ✓ NeMo-Gym environment ready")

    # Map "math" task to NeMo-Gym environment since data has task_name="math"
    task_to_env: dict[str, EnvironmentInterface] = defaultdict(lambda: nemo_gym_env)
    task_to_env["math"] = nemo_gym_env
    return task_to_env


def get_nemo_gym_agent_name(nemo_gym_config: dict[str, Any]) -> str:
    """
    Get the agent name to use for NeMo-Gym rollouts.

    This should match a top-level key in the NeMo-Gym config that defines
    an agent server (e.g., "math_with_judge_simple_agent").

    Can be configured via nemo_gym.agent_name in the config.
    """
    # Default to math_with_judge_simple_agent which is defined in
    # resources_servers/math_with_judge/configs/math_with_judge.yaml
    return nemo_gym_config.get("agent_name", "math_with_judge_simple_agent")


def main() -> None:
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "grpo.yaml")

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
    if config["policy"]["make_sequence_length_divisible_by"] is None:
        tp = config["policy"]["tensor_model_parallel_size"]
        cp = config["policy"]["context_parallel_size"]
        config["policy"]["make_sequence_length_divisible_by"] = setup_make_sequence_length_divisible_by(tp, cp)
    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Check if we should use NeMo-Gym
    should_use_nemo_gym = config["env"].get("should_use_nemo_gym", False)

    if should_use_nemo_gym and not NEMO_GYM_AVAILABLE:
        print(f"  ⚠️  Warning: NeMo-Gym requested but import failed: {_NEMO_GYM_IMPORT_ERROR}")
        print("     Falling back to MathEnvironment")
        should_use_nemo_gym = False

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}")

    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, "A generation config is required for GRPO"
    config["policy"]["generation"] = configure_generation_config(config["policy"]["generation"], tokenizer)

    # Apply NeMo-Gym specific config if enabled
    # This is done BEFORE setup() so vLLM is configured to expose HTTP server
    nemo_gym_agent_name = None
    if should_use_nemo_gym:
        # Use the canonical setup_nemo_gym_config function from nemo_rl
        # This sets: vllm_cfg.async_engine=True, vllm_cfg.expose_http_server=True
        # And clears stop_strings/stop_token_ids
        setup_nemo_gym_config(config, tokenizer)
        print("  ✓ NeMo-Gym config applied (vLLM HTTP server exposed)")

        # Get the agent name to use from the config
        nemo_gym_config = config["env"].get("nemo_gym", {})
        nemo_gym_agent_name = get_nemo_gym_agent_name(nemo_gym_config)
        print(f"  Agent name for rollouts: {nemo_gym_agent_name}")

    # Setup data (without environments - those come after setup())
    dataset, val_dataset = setup_data_only(
        tokenizer,
        config["data"],
        use_nemo_gym=should_use_nemo_gym,
        nemo_gym_agent_name=nemo_gym_agent_name or "math_with_judge_simple_agent",
    )

    # Run setup() to create policy, policy_generation, etc.
    # policy_generation.dp_openai_server_base_urls is only available after this
    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    # NOW set up the environment - after setup() so we have policy_generation
    if should_use_nemo_gym:
        # Get nemo_gym config from env.nemo_gym
        nemo_gym_config = config["env"].get("nemo_gym", {})
        task_to_env = setup_nemo_gym_environment(policy_generation, nemo_gym_config)
        val_task_to_env = task_to_env
    else:
        task_to_env = setup_math_environment(config["env"])
        val_task_to_env = task_to_env

    # Check if async mode is enabled
    if "async_grpo" in config["grpo"] and config["grpo"]["async_grpo"]["enabled"]:
        # Async GRPO does not support dynamic sampling, reward scaling, or reward shaping (DAPO features)
        unsupported_features = [
            "use_dynamic_sampling",
            "reward_scaling",
            "reward_shaping",
        ]

        for feature in unsupported_features:
            if feature not in config["grpo"]:
                continue

            if feature == "use_dynamic_sampling":
                if config["grpo"][feature]:
                    raise NotImplementedError(f"{feature} is not supported with async GRPO")
            else:
                if config["grpo"][feature]["enabled"]:
                    raise NotImplementedError(f"{feature} is not supported with async GRPO")

        from nemo_rl.algorithms.grpo import async_grpo_train

        print("🚀 Running async GRPO training")

        async_config = config["grpo"]["async_grpo"]
        # Run async GRPO training
        async_grpo_train(
            policy=policy,
            policy_generation=policy_generation,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            task_to_env=task_to_env,
            val_task_to_env=val_task_to_env,
            logger=logger,
            checkpointer=checkpointer,
            grpo_save_state=grpo_state,
            master_config=master_config,
            max_trajectory_age_steps=async_config["max_trajectory_age_steps"],
        )
    else:
        print("🚀 Running synchronous GRPO training")

        # Run standard GRPO training
        grpo_train(
            policy,
            policy_generation,
            dataloader,
            val_dataloader,
            tokenizer,
            loss_fn,
            task_to_env,
            val_task_to_env,
            logger,
            checkpointer,
            grpo_state,
            master_config,
        )


if __name__ == "__main__":
    main()
