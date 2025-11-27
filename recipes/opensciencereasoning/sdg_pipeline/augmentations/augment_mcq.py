import json
import logging
import shlex
import sys
from dataclasses import field
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import OmegaConf

from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments
from nemo_skills.utils import nested_dataclass, setup_logging

LOG = logging.getLogger(__name__)


@nested_dataclass(kw_only=True)
class AugmentMcqConfig:
    """Configuration for MCQ augmentation task."""
    # Optional fields that can be set via command line
    cluster: Optional[str] = None
    output_dir: Optional[str] = None
    input_file: Optional[str] = None
    
    # Generation kwargs for the augmentation process
    generation_kwargs: dict = field(default_factory=dict)


def augment_mcq(cluster, expname, stage_config, **kwargs):
    """Run difficulty estimation generation, judge correctness, and postprocess metrics.

    This stage:
      - Generates multiple solutions per problem using the provided model/prompt.
      - Runs LLM-based judging (math_judge) over those generations to get Yes/No per sample.
      - Postprocesses the judgements to append three keys to the final results file:
        - difficulty_model: the model used for generation
        - difficulty_model_pass_rate: decimal fraction of correct judgements (e.g., 0.5)
        - difficulty_model_pass_at_n: formatted fraction "correct/total" (e.g., 2/4)

    Note: The judging step extracts predicted answers using the \\boxed{...} convention.
    It will only work out-of-the-box when generations include a final answer in boxed format.
    """
    output_dir = stage_config.output_dir
    input_file = stage_config.input_file

    generation_kwargs = stage_config.generation_kwargs

    generation_args = generation_kwargs.get("args", {})
    generation_ctx_args = generation_kwargs.get("ctx_args", "")


    generate(
        ctx=wrap_arguments(generation_ctx_args),
        cluster=cluster,
        input_file=input_file,
        output_dir=f"{output_dir}/generation",
        expname=f"{expname}-generation",
        postprocess_cmd=f"python /nemo_run/code/recipes/opensciencereasoning/sdg_pipeline/augmentations/parse_new_options.py {output_dir}/generation/output.jsonl", 
        **generation_args,
    )

   
    # run_cmd(
    #     ctx=wrap_arguments(
    #         f"python /nemo_run/code/recipes/opensciencereasoning/sdg_pipeline/scripts/aggregate_difficulty.py "
    #         f"    --judgement_dir '{output_dir}/judgement' "
    #         f"    --output_file '{output_dir}/{OUTPUT_FILE}' "
    #         f"    --difficulty_model '{generation_args['model'].split('/')[-1]}' "
    #     ),
    #     cluster=cluster,
    #     log_dir=f"{output_dir}/logs",
    #     run_after=f"{expname}-judgement",
    #     expname=expname,
    # )


# Register the configuration with Hydra's ConfigStore
cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_augment_mcq_config", node=AugmentMcqConfig)


@hydra.main(version_base=None, config_path=".", config_name="augmentation_args")
def main(cfg: AugmentMcqConfig):
    """Main entry point for MCQ augmentation."""
    cfg = AugmentMcqConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    # Validate required fields
    if cfg.input_file is None:
        raise ValueError("input_file must be specified")
    if cfg.output_dir is None:
        raise ValueError("output_dir must be specified")

    augment_mcq(
        cluster=cfg.cluster or "local",
        expname="augment_mcq",
        stage_config=cfg,
    )


if __name__ == "__main__":
    setup_logging()
    main()