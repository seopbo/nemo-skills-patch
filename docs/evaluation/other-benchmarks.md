# Other benchmarks

More details are coming soon!

## Supported benchmarks

### arena-hard

- Benchmark is defined in [`nemo_skills/dataset/arena-hard/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/arena-hard/__init__.py)
- Original benchmark source is [here](https://github.com/lmarena/arena-hard-auto).
- Uses `gpt-4.1` as the default judge model for evaluation.
- Uses `gpt-4-0314` model's responses as reference answers (baseline answers) for comparison.

#### Data Preparation

First, prepare the dataset by running the `ns prepare_data` command.

```bash
ns prepare_data arena-hard
```

#### Running the Evaluation

Once the data is prepared, you can run the evaluation. Replace `<...>` placeholders with your cluster and directory paths.

```bash
ns eval \
    --cluster=<CLUSTER_NAME> \
    --model=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
    --server_type=vllm \
    --server_gpus=8 \
    --benchmarks=arena-hard \
    --data_dir=/workspace/ns-data/arena-hard \
    --output_dir=<OUTPUT_DIR> \
    ++parse_reasoning=True \
    ++inference.temperature=0.6 \
    ++inference.top_p=0.95 \
    ++inference.tokens_to_generate=32768
```

#### Verifying Results

After all jobs are complete, you can check the results in `<OUTPUT_DIR>/eval-results/arena-hard/metrics.json`.

```
------------------------------------------- arena-hard -------------------------------------------
evaluation_mode | num_entries | score  | 95_CI         | invalid_scores | avg_tokens | gen_seconds
pass@1          | 500         | 94.82% | (-0.67, 0.69) | 0              | 3878       | 230
```

### arena-hard-v2

- Benchmark is defined in [`nemo_skills/dataset/arena-hard-v2/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/arena-hard-v2/__init__.py)
- Original benchmark source is [here](https://github.com/lmarena/arena-hard-auto).
- Uses `o3-mini-2025-01-31` as the default judge model for evaluation.
- Uses `o3-mini-2025-01-31` model's responses as reference answers (baseline answers) for comparison.

#### Data Preparation

First, prepare the dataset by running the `ns prepare_data` command.

```bash
ns prepare_data arena-hard-v2
```

#### Running the Evaluation

Once the data is prepared, you can run the evaluation. Replace `<...>` placeholders with your cluster and directory paths.

```bash
ns eval \
    --cluster=<CLUSTER_NAME> \
    --model=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
    --server_type=vllm \
    --server_gpus=8 \
    --benchmarks=arena-hard-v2 \
    --data_dir=/workspace/ns-data/arena-hard-v2 \
    --output_dir=<OUTPUT_DIR> \
    ++parse_reasoning=True \
    ++inference.temperature=0.6 \
    ++inference.top_p=0.95 \
    ++inference.tokens_to_generate=32768
```

#### Verifying Results

After all jobs are complete, you can check the results in `<OUTPUT_DIR>/eval-results/arena-hard-v2/metrics.json`.

```
----------------------------------------- arena-hard-v2 ------------------------------------------
evaluation_mode | num_entries | score  | 95_CI         | invalid_scores | avg_tokens | gen_seconds
pass@1          | 750         | 64.15% | (-1.74, 1.55) | 0              | 4309       | 101


### AA-Omniscience

This is a benchmark developed by AA to measure hallucinations in LLMs and penalize confidently-false answers.

- Benchmark is defined in [`nemo_skills/dataset/omniscience/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/omniscience/__init__.py)
- Original benchmark and leaderboard are defined [here](https://artificialanalysis.ai/evaluations/omniscience), and data is [here](https://huggingface.co/datasets/ArtificialAnalysis/AA-Omniscience-Public)

#### Eval Results:
|        Model        | Accuracy | Omni-Index | Hallucination Rate |
| ------------------- | -------- | ---------- | ------------------ |
| Qwen3-8B (Reported) | 12.73%   | -66        | 90.36%             |
| Qwen3-8B (Measured) | 15.17%   | -64.83     | 94.30%             |

#### Notes:
- Note that this benchmark can be quite sensitive to temperature and other sampling parameters, so make sure your settings align well with downstream conditions.
- Also note that there still may be some variance between the public set and the full dataset; however, this set may be used as a way to compare hallucination rates between different checkpoints/models.

#### Configuration: Qwen3-8B with default judge (gemini-2.5-flash-preview-09-2025)
- Make sure to set `DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET=24576` in your environment variables and nemo-skills config to max judge reasoning when using reasoning_effort='high'.

```python
from nemo_skills.pipeline.cli import wrap_arguments, eval

eval(
    ctx=wrap_arguments(
        f"++inference.temperature=0.6 "
        f"++inference.top_p=1.0 "
        f"++inference.top_k=-1 "
        f"++inference.tokens_to_generate=131072 "
        f"++inference.reasoning_effort='high' "
    ),
    cluster="slurm",
    expname="aa-omniscience-eval",
    model="Qwen/Qwen3-8B",
    server_gpus=8,
    server_nodes=1,
    server_type="vllm",
    server_args="--async-scheduling",
    benchmarks="omniscience",
    output_dir="/workspace/experiments/aa-omniscience-eval",
    data_dir="/workspace/data_dir",
    extra_judge_args="++inference.reasoning_effort='high' ++inference.temperature=1.0 ++inference.top_p=0.95 ++inference.top_k=64 " # set max reasoning effort and default temp for judge
)
```

#### Configuration: Qwen3-8B with custom judge (gpt-oss-120b)
```python
from nemo_skills.pipeline.cli import wrap_arguments, eval

eval(
    ctx=wrap_arguments(
        f"++inference.temperature=0.6 "
        f"++inference.top_p=1.0 "
        f"++inference.top_k=-1 "
        f"++inference.tokens_to_generate=131072 "
        f"++inference.reasoning_effort='high' "
    ),
    cluster="slurm",
    expname="aa-omniscience-eval",
    model="Qwen/Qwen3-8B",
    server_gpus=8,
    server_nodes=1,
    server_type="vllm",
    server_args="--async-scheduling",
    judge_model="openai/gpt-oss-120b",
    judge_server_type="vllm",
    judge_server_gpus=8,
    judge_server_args="--async-scheduling  --reasoning-parser GptOss",
    benchmarks="omniscience",
    output_dir="/workspace/experiments/aa-omniscience-eval",
    data_dir="/workspace/data_dir"
)
```