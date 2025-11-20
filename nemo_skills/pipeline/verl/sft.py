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

import logging
import os
from dataclasses import dataclass
from typing import List

import typer

from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils import (
    add_task,
    check_if_mounted,
    get_cluster_config,
    get_exp,
    get_timeout_str,
    run_exp,
)
from nemo_skills.pipeline.verl import verl_app
from nemo_skills.utils import get_logger_name, setup_logging, validate_wandb_project_name

LOG = logging.getLogger(get_logger_name(__file__))


@dataclass
class TrainingParams:
    model: str
    output_dir: str
    training_data: str
    validation_data: str
    num_gpus: int
    num_nodes: int
    expname: str
    disable_wandb: bool
    wandb_project: str
    timeout: str
    verl_config_dir: str
    verl_config_name: str
    extra_arguments: str = ""


def _get_torchrun_cmd(cluster_config, params: TrainingParams) -> str:
    format_dict = {}
    if cluster_config["executor"] != "slurm":
        if params.num_nodes != 1:
            raise AssertionError("Local executor only supports single node training")
        format_dict["nnodes"] = 1
        format_dict["nproc_per_node"] = params.num_gpus
        format_dict["node_rank"] = 0
        format_dict["master_addr"] = "localhost"
    else:
        format_dict["nnodes"] = params.num_nodes
        format_dict["nproc_per_node"] = params.num_gpus
        format_dict["node_rank"] = "$SLURM_PROCID"
        format_dict["master_addr"] = "$SLURM_MASTER_NODE"

    format_dict["master_port"] = 9901

    base_cmd = (
        "torchrun --nproc_per_node {nproc_per_node} --nnodes {nnodes} --node-rank {node_rank} "
        "--master_addr {master_addr} --master_port {master_port} "
    )
    return base_cmd.format(**format_dict)


def _build_hydra_overrides(params: TrainingParams) -> List[str]:
    overrides = [
        f"trainer.default_local_dir={os.path.join(params.output_dir, 'checkpoints')}",
        f"++trainer.timeout={params.timeout}",
        f"trainer.n_gpus_per_node={params.num_gpus}",
        f"trainer.nnodes={params.num_nodes}",
        f"trainer.project_name={params.wandb_project}",
        f"trainer.experiment_name={params.expname}",
        f"model.partial_pretrain={params.model}",
        f"data.train_files='{params.training_data}'",
    ]

    if params.validation_data:
        overrides.append(f"data.val_files='{params.validation_data}'")

    if params.disable_wandb:
        overrides.append("trainer.logger=['console']")
    else:
        if os.getenv("WANDB_API_KEY") is None:
            raise ValueError("WANDB_API_KEY is not set. Use --disable_wandb to disable wandb logging")
        validate_wandb_project_name(
            wandb_project=params.wandb_project,
            wandb_name=params.expname,
            wandb_id=params.expname,
        )
        overrides.append("trainer.logger=['console','wandb']")
        overrides.append(f"++trainer.wandb_id={params.expname}")
        overrides.append("++trainer.wandb_resume=allow")
        overrides.append(f"++trainer.wandb_group={params.expname}")

    return overrides


def _get_config_flags(params: TrainingParams) -> str:
    flags: List[str] = []
    if params.verl_config_dir:
        flags.append(f"--config-path {params.verl_config_dir}")
    if params.verl_config_name:
        flags.append(f"--config-name {params.verl_config_name}")
    return " ".join(flags)


def get_training_cmd(cluster_config, params: TrainingParams) -> str:
    torchrun_cmd = _get_torchrun_cmd(cluster_config, params)
    hydra_overrides = _build_hydra_overrides(params)
    hydra_override_str = " ".join(hydra_overrides)
    config_flags = _get_config_flags(params)
    if config_flags:
        config_flags = f"{config_flags} "

    cmd = (
        "export HYDRA_FULL_ERROR=1 && "
        "export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        "cd /nemo_run/code && "
        "echo 'Starting Verl SFT training' && "
        f"{torchrun_cmd}-m verl.trainer.fsdp_sft_trainer "
        f"{config_flags}{hydra_override_str} "
        f"{params.extra_arguments}".strip()
    )
    return cmd


@verl_app.command(name="sft", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def sft_verl(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    output_dir: str = typer.Option(..., help="Where to put results"),
    expname: str = typer.Option("verl-sft", help="Nemo run experiment name"),
    hf_model: str = typer.Option(..., help="Path to the HF model"),
    training_data: str = typer.Option(..., help="Path to the training data"),
    validation_data: str = typer.Option(None, help="Path to the validation data"),
    num_nodes: int = typer.Option(1, help="Number of nodes"),
    num_gpus: int = typer.Option(..., help="Number of GPUs"),
    num_training_jobs: int = typer.Option(1, help="Number of training jobs"),
    wandb_project: str = typer.Option("nemo-skills", help="Weights & Biases project name"),
    disable_wandb: bool = typer.Option(False, help="Disable wandb logging"),
    partition: str = typer.Option(
        None, help="Can specify if need interactive jobs or a specific non-default partition"
    ),
    verl_config_dir: str = typer.Option(None, help="Hydra config path for Verl trainer"),
    verl_config_name: str = typer.Option("sft_trainer", help="Hydra config name for Verl trainer"),
    time_min: str = typer.Option(None, help="If specified, will use as a time-min slurm parameter"),
    run_after: List[str] = typer.Option(
        None, help="Can specify a list of expnames that need to be completed before this one starts"
    ),
    reuse_code: bool = typer.Option(
        True,
        help="If True, will reuse the code from the provided experiment. "
        "If you use it from Python, by default the code will be re-used from "
        "the last submitted experiment in the current Python session, so set to False to disable "
        "(or provide reuse_code_exp to override).",
    ),
    reuse_code_exp: str = typer.Option(
        None,
        help="If specified, will reuse the code from this experiment. "
        "Can provide an experiment name or an experiment object if running from code.",
    ),
    config_dir: str = typer.Option(None, help="Can customize where we search for cluster configs"),
    log_dir: str = typer.Option(
        None,
        help="Can specify a custom location for slurm logs. "
        "If not specified, will be inside `ssh_tunnel.job_dir` part of your cluster config.",
    ),
    exclusive: bool = typer.Option(
        True,
        "--not_exclusive",
        help="If --not_exclusive is used, will NOT use --exclusive flag for slurm",
    ),
    skip_hf_home_check: bool = typer.Option(
        False,
        help="If True, skip checking that HF_HOME env var is defined in the cluster config.",
    ),
    installation_command: str | None = typer.Option(
        None,
        help="An installation command to run before main job. Only affects main task (not server or sandbox). "
        "You can use an arbitrary command here and we will run it on a single rank for each node. "
        "E.g. 'pip install my_package'",
    ),
    dry_run: bool = typer.Option(False, help="If True, will not run the job, but will validate all arguments."),
    _reuse_exp: str = typer.Option(None, help="Internal option to reuse an experiment object.", hidden=True),
    _task_dependencies: List[str] = typer.Option(
        None, help="Internal option to specify task dependencies.", hidden=True
    ),
):
    """Runs Verl FSDP SFT training (verl.trainer.fsdp_sft_trainer)"""

    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = f"{' '.join(ctx.args)}"
    LOG.info("Starting training job")
    LOG.info("Extra arguments that will be passed to the underlying script: %s", extra_arguments)

    cluster_config = get_cluster_config(cluster, config_dir)
    check_if_mounted(cluster_config, output_dir)
    check_if_mounted(cluster_config, hf_model)
    if log_dir:
        check_if_mounted(cluster_config, log_dir)
    else:
        log_dir = output_dir

    if os.path.isabs(training_data):
        check_if_mounted(cluster_config, training_data)
    if validation_data and os.path.isabs(validation_data):
        check_if_mounted(cluster_config, validation_data)

    timeout = get_timeout_str(cluster_config, partition)

    training_params = TrainingParams(
        model=hf_model,
        output_dir=output_dir,
        training_data=training_data,
        validation_data=validation_data or training_data,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        expname=expname,
        disable_wandb=disable_wandb,
        wandb_project=wandb_project,
        timeout=timeout,
        verl_config_dir=verl_config_dir,
        verl_config_name=verl_config_name,
        extra_arguments=extra_arguments,
    )

    train_cmd = get_training_cmd(cluster_config, training_params)

    with get_exp(expname, cluster_config, _reuse_exp) as exp:
        prev_task = _task_dependencies
        for job_id in range(num_training_jobs):
            prev_task = add_task(
                exp,
                cmd=train_cmd,
                task_name=f"{expname}-sft-{job_id}",
                log_dir=f"{log_dir}/training-logs",
                container=cluster_config["containers"]["verl"],
                num_gpus=num_gpus,
                num_nodes=num_nodes,
                num_tasks=1,
                cluster_config=cluster_config,
                partition=partition,
                time_min=time_min,
                run_after=run_after,
                reuse_code=reuse_code,
                reuse_code_exp=reuse_code_exp,
                task_dependencies=[prev_task] if prev_task is not None else None,
                slurm_kwargs={"exclusive": exclusive} if exclusive else None,
                installation_command=installation_command,
                skip_hf_home_check=skip_hf_home_check,
            )

        run_exp(exp, cluster_config, sequential=False, dry_run=dry_run)

    if _reuse_exp:
        return [prev_task]
    return exp


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
