## Nemo-Gym-RL integration with NeMo Skills

### Setup

1. Add `gym-nemo-rl: /lustre/fsw/portfolios/coreai/users/terryk/enroot-images/gitlab-master.nvidia.com/terryk/images/nemo-rl:main-2aea5add.squashfs` to your cluster config .yaml file. This is the container used for nemo-rl with nemo-gym integration.

2. Clone the NeMo-RL repo to your cluster, for now specifically to `/workspace/nemo-rl`:

```
# cd to workspace as defined in your cluster config
git clone https://gitlab-master.nvidia.com/nexus-team/nemo-rl
cd nemo-rl
git checkout bxyu/nemo-gym-integration-main
```

3. Complete these instructions as well to have the repo setup properly (if there are issues with the setup, refer to the [doc](https://docs.google.com/document/d/1z0wLyl6lNpLhLCqd33EH04RdcIl4UTy08_ROEnNRnUA/edit?tab=t.b05geyn4qj77#heading=h.g6xyqy7cjg64)).

```
# Fetch the NeMo Gym submodule, codenamed "Penguin" before release
# You will need to put your NV Github user name and PAT for the first submodule clone
git pull && git submodule update --init --recursive

# Add your creds to the Git submodule to you don't need to do so on every pull
cd 3rdparty/Penguin-workspace/Penguin
git remote set-url origin https://{your NV Github username}:{your NV Github PAT}@github.com/NVIDIA-NeMo/Gym.git
cd ../../..

# Initial setup
source /opt/nemo_rl_venv/bin/activate
uv sync --group={build,docs,dev,test} --extra penguin
```

4. Call the root of your NeMo-RL with Gym integration repo as `REPO_LOCATION` (e.g. `/workspace/nemo-rl` or specify as an argument when running GRPO given below). To circumvent running `ng_prepare_data` (potential errors to handle), copy the contents of the directory `/lustre/fsw/portfolios/llmservice/users/eminasyan/nemo-rl/3rdparty/Penguin-workspace/Penguin/data/comp_coding` on the `dfw` cluster to `{REPO_LOCATION}/3rdparty/Penguin-workspace/Penguin/data/comp_coding`.

With the steps done until here, the job runs into the `Unable to find lockfile at uv.lock` error when building nemo-gym (initializing Penguin [here](https://gitlab-master.nvidia.com/nexus-team/nemo-rl/-/blob/bxyu/nemo-gym-integration-main/examples/penguin/run_grpo_penguin.py#L175)).

To fix the issue, complete the following temporary steps:

5. Comment out lines 202-203 in `{REPO_LOCATION}/3rdpart/Penguin-workspace/Penguin/pyproject.toml`:
```
#[tool.distutils.egg_info]
#egg_base = "cache"
```

6. Add `working_dir` as the `REPO_LOCATION` in `{REPO_LOCATION}/nemo_rl/distributed/virtual_cluster.py` in lines 100-102:
```
runtime_env = {
        "env_vars": env_vars,  # Pass thru all user environment variables
        "working_dir": os.getcwd(),
    }
```

## Running GRPO from NeMo-RL via Gym from NeMo-Skills.

There is a test/example of running GRPO in `./tests/test_nemo_gym_rl.py`, the new command is `grpo_nemo_gym_rl` in the file `nemo_skills/pipeline/nemo_rl/grpo_gym.py`.

# NeMo Skills

NeMo-Skills is a collection of pipelines to improve "skills" of large language models (LLMs). We support everything needed for LLM development, from synthetic data generation, to model training, to evaluation on a wide range of benchmarks. Start developing on a local workstation and move to a large-scale Slurm cluster with just a one-line change.

Here are some of the features we support:

- [Flexible LLM inference](https://nvidia.github.io/NeMo-Skills/pipelines/generation/):
  - Seamlessly switch between API providers, local server and large-scale slurm jobs for LLM inference.
  - Host models (on 1 or many nodes) with [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [vLLM](https://github.com/vllm-project/vllm), [sglang](https://github.com/sgl-project/sglang) or [Megatron](https://github.com/NVIDIA/Megatron-LM).
  - Scale SDG jobs from 1 GPU on a local machine all the way to tens of thousands of GPUs on a slurm cluster.
- [Model evaluation](https://nvidia.github.io/NeMo-Skills/evaluation):
  - Evaluate your models on many popular benchmarks.
    - [**Math (natural language**)](https://nvidia.github.io/NeMo-Skills/evaluation/natural-math): e.g. [aime24](https://nvidia.github.io/NeMo-Skills/evaluation/natural-math/#aime24), [aime25](https://nvidia.github.io/NeMo-Skills/evaluation/natural-math/#aime25), [hmmt_feb25](https://nvidia.github.io/NeMo-Skills/evaluation/natural-math/#hmmt_feb25)
    - [**Math (formal language)**](https://nvidia.github.io/NeMo-Skills/evaluation/formal-math): e.g. [minif2f](https://nvidia.github.io/NeMo-Skills/evaluation/formal-math/#minif2f), [proofnet](https://nvidia.github.io/NeMo-Skills/evaluation/formal-math/#proofnet), [putnam-bench](https://nvidia.github.io/NeMo-Skills/evaluation/formal-math/#putnam-bench)
    - [**Code**](https://nvidia.github.io/NeMo-Skills/evaluation/code): e.g. [swe-bench](https://nvidia.github.io/NeMo-Skills/evaluation/code/#swe-bench), [livecodebench](https://nvidia.github.io/NeMo-Skills/evaluation/code/#livecodebench)
    - [**Scientific knowledge**](https://nvidia.github.io/NeMo-Skills/evaluation/scientific-knowledge): e.g., [hle](https://nvidia.github.io/NeMo-Skills/evaluation/scientific-knowledge/#hle), [scicode](https://nvidia.github.io/NeMo-Skills/evaluation/scientific-knowledge/#scicode), [gpqa](https://nvidia.github.io/NeMo-Skills/evaluation/scientific-knowledge/#gpqa)
    - [**Instruction following**](https://nvidia.github.io/NeMo-Skills/evaluation/instruction-following): e.g. [ifbench](https://nvidia.github.io/NeMo-Skills/evaluation/instruction-following/#ifbench), [ifeval](https://nvidia.github.io/NeMo-Skills/evaluation/instruction-following/#ifeval)
    - [**Long-context**](https://nvidia.github.io/NeMo-Skills/evaluation/long-context): e.g. [ruler](https://nvidia.github.io/NeMo-Skills/evaluation/long-context/#ruler), [mrcr](https://nvidia.github.io/NeMo-Skills/evaluation/long-context/#mrcr), [aalcr](https://nvidia.github.io/NeMo-Skills/evaluation/long-context/#aalcr)
    - [**Tool-calling**](https://nvidia.github.io/NeMo-Skills/evaluation/tool-calling): e.g. [bfcl_v3](https://nvidia.github.io/NeMo-Skills/evaluation/tool-calling/#bfcl_v3)
    - [**Multilingual**](https://nvidia.github.io/NeMo-Skills/evaluation/multilingual): e.g. [mmlu-prox](https://nvidia.github.io/NeMo-Skills/evaluation/multilingual/#mmlu-prox), [FLORES-200](https://nvidia.github.io/NeMo-Skills/evaluation/multilingual/#FLORES-200), [wmt24pp](https://nvidia.github.io/NeMo-Skills/evaluation/multilingual/#wmt24pp)
  - Easily parallelize each evaluation across many slurm jobs, self-host LLM judges, bring your own prompts or change benchmark configuration in any other way.
- [Model training](https://nvidia.github.io/NeMo-Skills/pipelines/training): Train models using [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner/), [NeMo-RL](https://github.com/NVIDIA/NeMo-RL/) or [verl](https://github.com/volcengine/verl).

## News
* [08/22/2025]: Added details for [reproducing evals](https://nvidia.github.io/NeMo-Skills/tutorials/2025/08/22/reproducing-nvidia-nemotron-nano-9b-v2-evals/) for the [NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) model by NVIDIA.
* [08/15/2025]: Added details for [reproducing evals](https://nvidia.github.io/NeMo-Skills/tutorials/2025/08/15/reproducing-llama-nemotron-super-49b-v15-evals/) for the [Llama-3_3-Nemotron-Super-49B-v1_5](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5) model by NVIDIA.
* [07/30/2025]: The datasets used to train OpenReasoning models are released! Math and code are available as part of [Nemotron-Post-Training-Dataset-v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1) and science is available in
[OpenScienceReasoning-2](https://huggingface.co/datasets/nvidia/OpenScienceReasoning-2).
See our [documentation](https://nvidia.github.io/NeMo-Skills/releases/openreasoning/training) for more details.

* [07/18/2025]: We released [OpenReasoning](https://nvidia.github.io/NeMo-Skills/releases/openreasoning/) models! SOTA scores on math, coding and science benchmarks.

![Evaluation Results with pass@1](docs/releases/openreasoning/pass-1.png)

![Evaluation Results with GenSelect](docs/releases/openreasoning/genselect.png)


* [04/23/2025]: We released [OpenMathReasoning](https://nvidia.github.io/NeMo-Skills/openmathreasoning1) dataset and models!

  * OpenMathReasoning dataset has 306K unique mathematical problems sourced from [AoPS forums](https://artofproblemsolving.com/community) with:
      * 3.2M long chain-of-thought (CoT) solutions
      * 1.7M long tool-integrated reasoning (TIR) solutions
      * 566K samples that select the most promising solution out of many candidates (GenSelect)
  * OpenMath-Nemotron models are SoTA open-weight models on math reasoning benchmarks at the time of release!

* [10/03/2024]: We released [OpenMathInstruct-2](https://nvidia.github.io/NeMo-Skills/openmathinstruct2) dataset and models!

  * OpenMathInstruct-2 is a math instruction tuning dataset with 14M problem-solution pairs generated using the Llama3.1-405B-Instruct model.
  * OpenMath-2-Llama models show significant improvements compared to their Llama3.1-Instruct counterparts.

## Getting started

To get started, follow these [steps](https://nvidia.github.io/NeMo-Skills/basics),
browse available [pipelines](https://nvidia.github.io/NeMo-Skills/pipelines) or run `ns --help` to see all available
commands and their options.

You can find more examples of how to use NeMo-Skills in the [tutorials](https://nvidia.github.io/NeMo-Skills/tutorials) page.

We've built and released many popular models and datasets using NeMo-Skills. See all of them in the [Papers & Releases](./releases/index.md) documentation.

You can find the full documentation [here](https://nvidia.github.io/NeMo-Skills/).


## Contributing

We welcome contributions to NeMo-Skills! Please see our [Contributing Guidelines](./CONTRIBUTING.md) for more information on how to get involved.


Disclaimer: This project is strictly for research purposes, and not an official product from NVIDIA.
