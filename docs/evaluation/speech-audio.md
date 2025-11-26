# Speech & Audio

This section covers evaluation of speech and audio benchmarks, including:

- **Audio understanding**: Tasks that test models' ability to reason about audio content (speech, music, environmental sounds)
- **ASR (Automatic Speech Recognition)**: Speech-to-text transcription tasks
- **AST (Automatic Speech Translation)**: Cross-lingual audio translation tasks

## Supported Server Types

Speech and audio benchmarks support following server types for running evaluation:

| Server Type | Configuration | Best For | Example Models |
|-------------|---------------|----------|----------------|
| **vLLM** | `--server_type=vllm` | Audio-capable models with multimodal support | Qwen2.5-Omni-3B |
| **Megatron** | `--server_type=megatron` | Custom Megatron-based implementations | Megatron-LM checkpoints |


## Supported benchmarks

### MMAU-Pro

MMAU-Pro (Multimodal Audio Understanding - Pro) is a comprehensive benchmark for evaluating audio understanding capabilities across three different task categories:

- **Closed-form questions**: Questions with specific answers evaluated using NVEmbed similarity matching
- **Open-ended questions**: Questions requiring detailed responses, evaluated with LLM-as-a-judge (Qwen 2.5)
- **Instruction following**: Tasks that test the model's ability to follow audio-related instructions

#### Dataset Location

- Benchmark is defined in [`nemo_skills/dataset/mmau-pro/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/mmau-pro/__init__.py)
- Original benchmark source is hosted on [HuggingFace](https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro)

## Preparing MMAU-Pro Data

MMAU-Pro requires audio files for meaningful evaluation. **Audio files are downloaded by default** to ensure proper evaluation.

!!! warning "Running without audio files"
    If you want to evaluate without audio files (not recommended) use
    `--no-audio` flag. In this case you can also set `--skip_data_dir_check`
    as data is very lightweight when audio files aren't being used.

### Data Preparation

To prepare the dataset with audio files:

```bash
export HF_TOKEN=your_huggingface_token
ns prepare_data mmau-pro --data_dir=/path/to/data --cluster=<cluster_name>
```

**What happens:**

- Requires authentication (HuggingFace token via `HF_TOKEN` environment variable)
- Downloads audio archive from HuggingFace and extracts
- Prepares the dataset files for evaluation

### Text-Only Mode (Not Recommended)

If you need to prepare without audio files:

```bash
ns prepare_data mmau-pro --no-audio --skip_data_dir_check
```

## Running Evaluation

### Python API Examples

=== "vLLM (Recommended for Qwen2.5-Omni)"

    ```python
    import os
    from nemo_skills.pipeline.cli import wrap_arguments, eval

    os.environ["HF_TOKEN"] = "your_huggingface_token"
    os.environ["NVIDIA_API_KEY"] = "your_nvidia_api_key"  # For LLM judge

    eval(
        ctx=wrap_arguments(""),
        cluster="oci_iad",
        output_dir="/workspace/mmau-pro-eval",
        benchmarks="mmau-pro",
        server_type="vllm",
        server_gpus=1,
        model="Qwen/Qwen2.5-Omni-3B",
        server_container="/path/to/vllm-openai-audio.sqsh",
        data_dir="/dataset",
        installation_command="pip install sacrebleu",
        server_args=f"--hf_token {os.environ['HF_TOKEN']}",
    )
    ```

=== "Megatron"

    ```python
    import os
    from nemo_skills.pipeline.cli import wrap_arguments, eval

    os.environ["NVIDIA_API_KEY"] = "your_nvidia_api_key"  # For LLM judge

    eval(
        ctx=wrap_arguments(""),
        cluster="oci_iad",
        output_dir="/workspace/mmau-pro-eval",
        benchmarks="mmau-pro",
        server_type="megatron",
        server_gpus=1,
        model="/workspace/checkpoint",
        server_entrypoint="/workspace/megatron-lm/server.py",
        server_container="/path/to/container.sqsh",
        data_dir="/dataset",
        installation_command="pip install sacrebleu",
        server_args="--inference-max-requests 1 --model-config /workspace/checkpoint/config.yaml",
    )
    ```

??? note "Alternative: Command-line usage"

    === "vLLM"

        ```bash
        export HF_TOKEN=your_huggingface_token
        export NVIDIA_API_KEY=your_nvidia_api_key

        ns eval \
          --cluster oci_iad \
          --output_dir /workspace/path/to/output/dir \
          --benchmarks mmau-pro \
          --server_type vllm \
          --server_gpus 1 \
          --model "Qwen/Qwen2.5-Omni-3B" \
          --server_container="/path/to/vllm-openai-audio.sqsh" \
          --data_dir="/dataset" \
          --installation_command "pip install sacrebleu" \
          --server_args "--hf_token ${HF_TOKEN}"
        ```

    === "Megatron"

        ```bash
        export HF_TOKEN=your_huggingface_token
        export NVIDIA_API_KEY=your_nvidia_api_key
        export MEGATRON_PATH=/workspace/path/to/megatron-lm

        ns eval \
            --cluster=oci_iad \
            --output_dir=/workspace/path/to/output/dir \
            --benchmarks=mmau-pro \
            --server_type=megatron \
            --server_gpus=1 \
            --model=/workspace/path/to/checkpoint-tp1 \
            --server_entrypoint=$MEGATRON_PATH/path/to/server.py \
            --server_container=/path/to/server_container.sqsh \
            --data_dir=/dataset \
            --installation_command="pip install sacrebleu" \
            ++max_concurrent_requests=1 \
            --server_args="--inference-max-requests 1 \
                           --model-config /workspace/path/to/checkpoint-tp1/config.yaml"
        ```

## How Evaluation Works

Each category uses a different evaluation strategy:

| Category | Evaluation Method | How It Works |
|----------|-------------------|--------------|
| **Closed-Form** | NVEmbed similarity matching | Model generates short answer; compared to expected answer using embeddings |
| **Open-Ended** | LLM-as-a-judge (Qwen 2.5 7B) | Model generates detailed response; Qwen 2.5 judges quality and correctness |
| **Instruction Following** | Custom evaluation logic | Model follows instructions; evaluator checks adherence |

### Sub-benchmarks

Evaluate individual categories:

- `mmau-pro.closed_form`
- `mmau-pro.open_ended`
- `mmau-pro.instruction_following`

```python
eval(benchmarks="mmau-pro.closed_form", ...)
```

### Using Custom Judge Models

The open-ended questions subset uses an LLM-as-a-judge (by default, Qwen 2.5 7B via NVIDIA API) to evaluate responses. You can customize the judge model for this subset:

=== "Default (NVIDIA API)"

    ```python
    from nemo_skills.pipeline.cli import wrap_arguments, eval
    import os

    os.environ["NVIDIA_API_KEY"] = "your_nvidia_api_key"

    eval(
        ctx=wrap_arguments(""),
        cluster="oci_iad",
        output_dir="/workspace/path/to/mmau-pro-eval",
        benchmarks="mmau-pro.open_ended",  # Only open-ended uses LLM judge
        server_type="megatron",
        server_gpus=1,
        model="/workspace/path/to/checkpoint-tp1",
        # ... other server args ...
    )
    ```

=== "Self-hosted Judge with SGLang"

    !!! warning "Self-hosted Judge Limitation"
        When using a self-hosted judge, evaluate `mmau-pro.open_ended` separately.

    ```python
    from nemo_skills.pipeline.cli import wrap_arguments, eval

    eval(
        ctx=wrap_arguments(""),
        cluster="oci_iad",
        output_dir="/workspace/path/to/mmau-pro-eval",
        benchmarks="mmau-pro.open_ended",  # Only open-ended uses LLM judge
        server_type="megatron",
        server_gpus=1,
        model="/workspace/path/to/checkpoint-tp1",
        # Judge configuration
        judge_model="Qwen/Qwen2.5-32B-Instruct",
        judge_server_type="sglang",
        judge_server_gpus=2,
        # ... other server args ...
    )
    ```

## Understanding Results

After evaluation completes, results are saved in your output directory under `eval-results/`:

```
<output_dir>/
├── eval-results/
│   └── mmau-pro/
│       ├── metrics.json                              # Overall aggregate scores
│       ├── mmau-pro.instruction_following/
│       │   └── metrics.json
│       ├── mmau-pro.closed_form/
│       │   └── metrics.json
│       └── mmau-pro.open_ended/
│           └── metrics.json
```

### Evaluation Output Format

When evaluation completes, results are displayed in formatted tables in the logs:

**Open-Ended Questions:**

```
------------------------------- mmau-pro.open_ended -------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 217        | 467         | 49.79%       | 0.00%     | 625
```

**Instruction Following:**

```
-------------------------- mmau-pro.instruction_following -------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 776        | 310         | 62.06%       | 0.00%     | 87

```

**Closed-Form Questions (Main Category + Sub-categories):**

```
------------------------------- mmau-pro.closed_form ------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 2          | 6581        | 41.69%       | 0.00%     | 4593

---------------------------- mmau-pro.closed_form-sound ---------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 691         | 39.31%       | 0.00%     | 1048

---------------------------- mmau-pro.closed_form-multi ---------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 1          | 6005        | 18.13%       | 0.00%     | 430

------------------------- mmau-pro.closed_form-sound_music ------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 810         | 30.00%       | 0.00%     | 50

---------------------------- mmau-pro.closed_form-music ---------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 5          | 5467        | 44.71%       | 0.00%     | 1418

------------------------ mmau-pro.closed_form-spatial_audio -----------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 5597        | 36.92%       | 0.00%     | 325

------------------------ mmau-pro.closed_form-music_speech ------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 1          | 5658        | 32.60%       | 0.00%     | 46

--------------------- mmau-pro.closed_form-sound_music_speech ---------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 5664        | 28.57%       | 0.00%     | 7

------------------------ mmau-pro.closed_form-sound_speech ------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 5713        | 42.04%       | 0.00%     | 88

--------------------------- mmau-pro.closed_form-speech ---------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 1          | 6312        | 49.04%       | 0.00%     | 891

------------------------- mmau-pro.closed_form-voice_chat -------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 6580        | 56.89%       | 0.00%     | 290
```

**Overall Aggregate Score:**

```
-------------------------------- mmau-pro -----------------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 597        | 6879        | 43.68%       | 0.00%     | 5305
```
