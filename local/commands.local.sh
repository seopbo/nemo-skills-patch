ns generate \
    --expname=test \
    --server_type=openai \
    --model=nvidia/openai/gpt-oss-20b \
    ++server.base_url=https://inference-api.nvidia.com/v1/\
    --output_dir=./generation \
    --input_file=./input.jsonl \
    ++prompt_config=./prompt.yaml \
    ++server.api_key_env_var=NV_API_KEY