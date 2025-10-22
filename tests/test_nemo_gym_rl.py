from nemo_skills.pipeline.cli import wrap_arguments, grpo_nemo_gym_rl

def run_gym_rl_grpo():
    account_name = "llmservice_nemo_reasoning"
    partition = "interactive"
    cluster = "ord"

    num_nodes=1
    num_gpus=8
    num_training_jobs=1

    repo_dir = "/workspace/nemo-rl"
    data_dir = f"{repo_dir}/3rdparty/Penguin-workspace/Penguin/data/comp_coding"
    training_data = data_dir + "/train.jsonl"
    validation_data = data_dir + "/validation.jsonl"

    run_n = 4
    expname = f"penguin_grpo_qwen3_4binstruct_comp_coding_test_{run_n}"
    gym_config_path = "examples/penguin/grpo_comp_coding_qwen3_4binstruct.yaml"
    model = "Qwen3-4B"

    num_prompts = 4
    num_steps = 3
    
    grpo_nemo_gym_rl(
        ctx=wrap_arguments(
            f"++grpo.val_at_start=false "
            f"++grpo.num_prompts_per_step={num_prompts} "
            f"++grpo.max_num_steps={num_steps} "
        ),
        cluster=cluster,
        expname=expname,
        hf_model=f"/hf_models/{model}",
        output_dir=f"{repo_dir}/results/{expname}",
        partition=partition,
        num_nodes=num_nodes,
        num_gpus=num_gpus,
        num_training_jobs=num_training_jobs,
        training_data=training_data,
        validation_data=validation_data,
        gym_config_path=f"{repo_dir}/{gym_config_path}",
        repo_location=repo_dir,
        backend="fsdp",
        disable_wandb=True,
        dry_run=False,
    )

if __name__ == "__main__":
    run_gym_rl_grpo()