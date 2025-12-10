# Recalculating LiveCodeBench Metrics

Script to recalculate LiveCodeBench metrics when code grading failed during the original evaluation.

## Prerequisites

```bash
cd /home/wprazuch/Projects/EvalFactory/Skills
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
source .venv/bin/activate
python scripts/recalculate_livecodebench.py <path-to-eval-results>
```

## Examples

```bash
# Basic usage
python scripts/recalculate_livecodebench.py data/ns_livecodebench/artifacts/eval-results

# Custom output directory
python scripts/recalculate_livecodebench.py data/ns_livecodebench/artifacts/eval-results \
    --output_dir data/my_results

# Adjust grading settings
python scripts/recalculate_livecodebench.py data/ns_livecodebench/artifacts/eval-results \
    --num_processes 8 \
    --timeout 10

# Skip grading (if files already have graded_list)
python scripts/recalculate_livecodebench.py data/ns_livecodebench/artifacts/eval-results \
    --skip_grading
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output_dir` | `<input>_recalculated` | Output directory for metrics.json |
| `--num_processes` | 12 | Parallel processes for code execution |
| `--timeout` | 6 | Timeout per test case (seconds) |
| `--skip_grading` | false | Only recalculate metrics, skip grading |

## What It Does

1. **Checks for missing grades**: Scans output files for `graded_list` field
2. **Runs code grading**: Executes generated code against test cases
3. **Calculates metrics**: Runs `ns summarize_results` to compute pass@k

## Common Error: `num_parallel_requests`

If your original evaluation failed with:
```
TypeError: LiveCodeBenchEvaluatorConfig.__init__() got an unexpected keyword argument 'num_parallel_requests'
```

This script fixes that by running grading with correct parameters.

## Output

```
============================================================
livecodebench
============================================================
evaluation_mode  | num_entries | accuracy
pass@1[avg-of-8] | 315         | 69.48% ± 1.82%
pass@8           | 315         | 81.90%
```

Metrics saved to: `<output_dir>/metrics.json`

