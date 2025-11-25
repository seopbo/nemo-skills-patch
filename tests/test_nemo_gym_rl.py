from nemo_skills.pipeline.cli import wrap_arguments, grpo_nemo_gym_rl

def run_gym_rl_grpo():
    partition = "interactive"
    cluster = "ord"

    num_nodes=1
    num_gpus=8
    num_training_jobs=1

    repo_dir = "/workspace/nemo-rl"
    data_dir = f"{repo_dir}/3rdparty/Penguin-workspace/Penguin/data/comp_coding"
    training_data = data_dir + "/train.jsonl"
    validation_data = data_dir + "/validation.jsonl"

    run_n = 6
    expname = f"penguin_grpo_qwen3_4binstruct_comp_coding_test_{run_n}"
    # gym_config_path = "examples/penguin/grpo_comp_coding_qwen3_4binstruct.yaml"
    model = "Qwen/Qwen3-4B-Instruct-2507"

    vllm_config_path = "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"
    resource_server_config_path = "resources_servers/comp_coding/configs/comp_coding.yaml"

    num_prompts = 2
    num_steps = 2
    
    grpo_nemo_gym_rl(
        ctx=wrap_arguments(
            f"++grpo.val_at_start=false "
            f"++grpo.num_prompts_per_step={num_prompts} "
            f"++grpo.max_num_steps={num_steps} "
            "++policy.max_total_sequence_length=1024 "
            "++policy.dtensor_cfg.tensor_parallel_size=1 "
            "++checkpointing.save_period=10 "
            # "++policy.train_global_batch_size=2 "
            "++policy.train_micro_batch_size=1 "
            "++policy.optimizer.kwargs.lr=1e-6 "
        ),
        cluster=cluster,
        expname=expname,
        hf_model=f"{model}",
        output_dir=f"{repo_dir}/results/{expname}",
        partition=partition,
        num_nodes=num_nodes,
        num_gpus=num_gpus,
        num_training_jobs=num_training_jobs,
        training_data=training_data,
        validation_data=validation_data,
        gym_config_paths=[vllm_config_path, resource_server_config_path],
        backend="fsdp",
        disable_wandb=True,
        with_sandbox=True,
        dry_run=False,
    )

if __name__ == "__main__":
    run_gym_rl_grpo()