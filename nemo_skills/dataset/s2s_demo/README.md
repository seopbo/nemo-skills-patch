# End2end Evaluation of the Duplex Speech2Speech Model Based on Nemo-Skills

The suggested recipe performs inference with a duplex s2s model including TTS. It's based on the incremental decoding scripts. Currently the [demo_20251124](demo_20251124/) test set is supported. Then it runs scoring based on Kevin's script which includes:
- Turn-taking evaluation.
- User ASR quality evaluation.
- Agent TTS quality evaluation.
- Special symbol balance.
- Agent content quality evaluation (LLM based).

In addition to average metrics the scoring recipe saves all kinds of alignments and error scores for each sample.

## Getting started

1. Go to https://inference.nvidia.com/ to get API key.
2. Clone this branch.
3. Create a `.venv` and install nemo skills there.
4. Decide which cluster you want to work on and setup the corresponding cluster configuration. The example configuration for draco_oci (oci_iad) is provided [here](../../../cluster_configs/). You can get more configurations [here](https://gitlab-master.nvidia.com/igitman/nemo-skills-configs/-/tree/main/cluster_configs/v0.7.1?ref_type=heads). Don't forget to update user name, folder names and add `/lustre` to the mounts list.
5. Look into the [config](scripts/s2s_demo_eval_config.yaml) file and make sure that the required artifacts are in the specified places.
6. Check the `data_dir` parameter. It should point to a folder on the cluster with `s2s_demo` folder from `nemo_skills/datasets`. Wav files should be in `s2s_demo/demo_20251124/data`. On draco everything is in `/lustre/fsw/portfolios/llmservice/users/vmendelev/experiments/voicebench_test/data_dir`
7. Adjust `output_dir`.
8. Set the `num_chunks` to be e.g. 8 to make the thing run faster.
9. Set `max_samples` to e.g. 2 if you want a fast run.
11. Make sure that `$NVIDIA_API_KEY` is set to a correct value.
12. Run the below command:

```bash
cd /home/vmendelev/workspace/expressiveness/src/ns_eval && \
source .venv/bin/activate && \
NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 python nemo_skills/dataset/s2s_demo/scripts/run_s2s_demo_eval.py \
  --config nemo_skills/dataset/s2s_demo/scripts/s2s_demo_eval_config.yaml
```

## What is also implemented but not pushed yet

- **Session backend.** The difference from the incremental one is that here we feed the user turn (no added pause), wait for the model to respond, then feed the second one and so on. We can also feed 2 turns in one go and preprogram feeding the second one at a predefined point while the system is responding to the first turn instead of 0s. With this one can enable a dialog between e.g Gemini and our model.
- **Offline backend** based on the old Chen Chen's recipe. Not sure if this is still needed.
- **Voicebench test with native scaling.** Will be pushed into nemo-skills separately.

## Input

Standard nemo-skills `test.jsonl` and corresponding set of wav files. The current demo test set can be found [here](/lustre/fsw/portfolios/llmservice/users/vmendelev/experiments/voicebench_test/data_dir/s2s_demo) and it was obtained from the available lhotse shars via this [script](convert_lhotse_to_eval.py). If you want to add more samples you need to add them to the `test.jsonl` and copy wav to the data folder. Or you can use the [script](convert_lhotse_to_eval.py) to convert another dataset. No reference transcription or segmentation is needed.

## Comparing Multiple Evaluation Results

Use the `compare_eval_results.py` script to compare metrics from multiple model evaluations and generate a Markdown report:

```bash
python nemo_skills/dataset/s2s_demo/scripts/compare_eval_results.py \
    --eval_folders \
        "draco-oci-login-01.draco-oci-iad.nvidia.com:/path/to/eval1:Model A" \
        "draco-oci-login-01.draco-oci-iad.nvidia.com:/path/to/eval2:Model B" \
    --output comparison_report.md
```

The script supports:
- **Remote folders via SSH**: `hostname:/path/to/folder:DisplayName`
- **Local folders**: `/path/to/folder:DisplayName`
- **Mixed**: compare models from different clusters or local/remote

See [example report](scripts/comparison_report.md) for output format.

## TODOs
1. Integrate vLLM and Triton.
2. Add a new offline backend.
3. Add WandB integration.
4. Refactor scoring script.
