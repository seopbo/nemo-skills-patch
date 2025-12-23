#!/usr/bin/env python3
"""
Compare multiple S2S demo evaluation results and generate a Markdown report.

Usage:
    python compare_eval_results.py \
        --eval_folders "/path/to/eval1:Model A" "/path/to/eval2:Model B" \
        --output report.md

Supports remote folders via SSH:
    python compare_eval_results.py \
        --eval_folders "host1:/path/to/eval1:Model A" "host2:/path/to/eval2:Model B" \
        --output report.md

Each eval folder should contain a metrics.json file.
"""

import argparse
import json
import os
import subprocess
from typing import Optional


def load_metrics_local(metrics_path: str) -> Optional[dict]:
    """Load metrics from a local JSON file."""
    if not os.path.exists(metrics_path):
        print(f"Warning: {metrics_path} not found")
        return None
    with open(metrics_path, "r") as f:
        return json.load(f)


def load_metrics_remote(host: str, metrics_path: str) -> Optional[dict]:
    """Load metrics from a remote JSON file via SSH."""
    try:
        result = subprocess.run(["ssh", host, f"cat {metrics_path}"], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"Warning: Failed to fetch {host}:{metrics_path}")
            print(f"  Error: {result.stderr.strip()}")
            return None
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        print(f"Warning: Timeout fetching {host}:{metrics_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON from {host}:{metrics_path}: {e}")
        return None
    except Exception as e:
        print(f"Warning: Error fetching {host}:{metrics_path}: {e}")
        return None


def load_metrics(location: str, host: Optional[str] = None) -> Optional[dict]:
    """Load metrics from a local or remote JSON file."""
    if host:
        return load_metrics_remote(host, location)
    return load_metrics_local(location)


def extract_key_metrics(metrics: dict) -> dict:
    """Extract key metrics from the metrics dict for comparison."""
    dm = metrics.get("dataset_metrics", metrics)

    tt = dm.get("turn_taking", {})
    bi = dm.get("barge_in", {})
    bc = dm.get("backchanneling", {})
    us = dm.get("user_speech", {})
    ags = dm.get("agent_speech", {})
    llm = dm.get("llm_judge", {})

    return {
        "num_samples": dm.get("num_samples_evaluated", 0),
        # Turn-taking
        "tt_latency_ms": tt.get("avg_latency_ms"),
        "tt_precision": tt.get("avg_precision"),
        "tt_recall": tt.get("avg_recall"),
        "tt_f1": tt.get("avg_f1"),
        # Barge-in
        "bi_success_rate": bi.get("avg_success_rate"),
        "bi_latency_ms": bi.get("avg_latency_ms"),
        # Backchanneling
        "bc_accuracy": bc.get("avg_accuracy"),
        # User speech (ASR quality)
        "user_wer": us.get("avg_wer"),
        "oob_ratio": us.get("out_of_bounds_word_ratio"),
        # Agent speech (TTS quality)
        "agent_wer": ags.get("avg_wer"),
        "agent_cer": ags.get("avg_cer"),
        "hallucination_rate": ags.get("hallucination_rate"),
        # LLM Judge ratings (1-5 scale)
        "llm_judge_overall": llm.get("overall", {}).get("avg_rating"),
        "llm_judge_full": llm.get("full", {}).get("avg_rating"),
        "llm_judge_sounded": llm.get("sounded", {}).get("avg_rating"),
    }


def format_value(value, format_spec: str = ".1f", suffix: str = "", na_str: str = "N/A") -> str:
    """Format a value for display."""
    if value is None:
        return na_str
    try:
        return f"{value:{format_spec}}{suffix}"
    except (ValueError, TypeError):
        return str(value)


def determine_best_model(models: list[tuple[str, dict]]) -> tuple[str, str]:
    """
    Determine which model is best based on key metrics.
    Returns (best_model_name, explanation).
    """
    if len(models) == 0:
        return "", "No models to compare."
    if len(models) == 1:
        return models[0][0], f"Only one model ({models[0][0]}) was evaluated."

    # Scoring: higher is better for most metrics, lower is better for latency/WER
    # Weight important metrics
    scores = {name: 0.0 for name, _ in models}
    comparisons = []

    # Metrics where HIGHER is better
    higher_better = [
        ("tt_f1", "Turn-taking F1", 2.0),
        ("tt_precision", "Turn-taking precision", 1.0),
        ("tt_recall", "Turn-taking recall", 1.0),
        ("bi_success_rate", "Barge-in success rate", 1.5),
        ("bc_accuracy", "Backchanneling accuracy", 1.0),
        ("llm_judge_overall", "LLM Judge score", 2.0),
    ]

    # Metrics where LOWER is better
    lower_better = [
        ("tt_latency_ms", "Turn-taking latency", 1.5),
        ("bi_latency_ms", "Barge-in latency", 1.0),
        ("user_wer", "User speech WER", 1.5),
        ("agent_wer", "Agent speech WER", 1.5),
        ("agent_cer", "Agent speech CER", 1.0),
        ("hallucination_rate", "TTS hallucination rate", 1.5),
        ("oob_ratio", "Out-of-bounds word ratio", 0.5),
    ]

    def get_valid_values(metric_key):
        return [(name, m[metric_key]) for name, m in models if m.get(metric_key) is not None]

    # Score higher-is-better metrics
    for metric_key, metric_name, weight in higher_better:
        valid = get_valid_values(metric_key)
        if len(valid) < 2:
            continue
        best_val = max(v for _, v in valid)
        worst_val = min(v for _, v in valid)
        if best_val == worst_val:
            continue
        best_name = [n for n, v in valid if v == best_val][0]
        for name, val in valid:
            normalized = (val - worst_val) / (best_val - worst_val) if best_val != worst_val else 0.5
            scores[name] += normalized * weight
        comparisons.append(f"{best_name} leads in {metric_name} ({format_value(best_val)})")

    # Score lower-is-better metrics
    for metric_key, metric_name, weight in lower_better:
        valid = get_valid_values(metric_key)
        if len(valid) < 2:
            continue
        best_val = min(v for _, v in valid)  # Lower is better
        worst_val = max(v for _, v in valid)
        if best_val == worst_val:
            continue
        best_name = [n for n, v in valid if v == best_val][0]
        for name, val in valid:
            # Invert: lower val -> higher score
            normalized = (worst_val - val) / (worst_val - best_val) if worst_val != best_val else 0.5
            scores[name] += normalized * weight
        comparisons.append(f"{best_name} leads in {metric_name} ({format_value(best_val)})")

    # Find winner
    if not any(scores.values()):
        return "", "Unable to determine best model: insufficient comparable metrics."

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_name, best_score = sorted_scores[0]
    second_name, second_score = sorted_scores[1] if len(sorted_scores) > 1 else ("", 0)

    # Check if clear winner
    score_diff = best_score - second_score
    total_possible = sum(w for _, _, w in higher_better + lower_better)

    if score_diff < 0.1 * total_possible:
        return "", (
            f"No clear winner: {best_name} and {second_name} have very similar overall performance. "
            f"Key differences: {'; '.join(comparisons[:3]) if comparisons else 'none significant'}."
        )

    return best_name, (
        f"**{best_name}** is the best model overall based on weighted metrics analysis. "
        f"Key advantages: {'; '.join(comparisons[:4]) if comparisons else 'balanced performance across all metrics'}."
    )


def generate_report(models: list[tuple[str, dict]], output_path: str):
    """Generate a Markdown comparison report."""
    lines = []
    lines.append("# S2S Demo Evaluation Comparison Report\n")
    lines.append(f"Comparing {len(models)} model(s):\n")
    for name, _ in models:
        lines.append(f"- {name}")
    lines.append("")

    # Summary table
    lines.append("## Metrics Comparison\n")

    # Table header
    header = "| Metric | " + " | ".join(name for name, _ in models) + " |"
    separator = "|" + "|".join(["---"] * (len(models) + 1)) + "|"
    lines.append(header)
    lines.append(separator)

    # Define rows with (display_name, metric_key, format_spec, suffix, higher_better)
    rows = [
        ("Samples Evaluated", "num_samples", "d", "", None),
        ("**Turn-Taking**", None, None, None, None),
        ("  Latency (ms) ↓", "tt_latency_ms", ".1f", "", False),
        ("  Precision (%) ↑", "tt_precision", ".1f", "", True),
        ("  Recall (%) ↑", "tt_recall", ".1f", "", True),
        ("  F1 (%) ↑", "tt_f1", ".1f", "", True),
        ("**Barge-In**", None, None, None, None),
        ("  Success Rate (%) ↑", "bi_success_rate", ".1f", "", True),
        ("  Latency (ms) ↓", "bi_latency_ms", ".1f", "", False),
        ("**Backchanneling**", None, None, None, None),
        ("  Accuracy (%) ↑", "bc_accuracy", ".1f", "", True),
        ("**User Speech (ASR)**", None, None, None, None),
        ("  WER (%) ↓", "user_wer", ".1f", "", False),
        ("  OOB Ratio (words/s) ↓", "oob_ratio", ".3f", "", False),
        ("**Agent Speech (TTS)**", None, None, None, None),
        ("  WER (%) ↓", "agent_wer", ".1f", "", False),
        ("  CER (%) ↓", "agent_cer", ".1f", "", False),
        ("  Hallucination (%) ↓", "hallucination_rate", ".1f", "", False),
        ("**LLM Judge (1-5)**", None, None, None, None),
        ("  Overall Rating ↑", "llm_judge_overall", ".2f", "", True),
        ("  Full Response ↑", "llm_judge_full", ".2f", "", True),
        ("  Sounded Response ↑", "llm_judge_sounded", ".2f", "", True),
    ]

    for display_name, metric_key, fmt, suffix, higher_better in rows:
        if metric_key is None:
            # Section header
            row = f"| {display_name} |" + " |" * len(models)
            lines.append(row)
            continue

        values = []
        raw_values = []
        for _, m in models:
            val = m.get(metric_key)
            raw_values.append(val)
            values.append(format_value(val, fmt, suffix))

        # Highlight best value if applicable
        if higher_better is not None:
            valid_vals = [(i, v) for i, v in enumerate(raw_values) if v is not None]
            if len(valid_vals) >= 2:
                if higher_better:
                    best_idx = max(valid_vals, key=lambda x: x[1])[0]
                else:
                    best_idx = min(valid_vals, key=lambda x: x[1])[0]
                values[best_idx] = f"**{values[best_idx]}**"

        row = f"| {display_name} | " + " | ".join(values) + " |"
        lines.append(row)

    lines.append("")

    # Best model analysis
    lines.append("## Analysis\n")
    best_name, explanation = determine_best_model(models)
    lines.append(explanation)
    lines.append("")

    # Legend
    lines.append("---")
    lines.append("*↑ = higher is better, ↓ = lower is better, **bold** = best value*")

    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report saved to: {output_path}")
    print("\n" + "=" * 60)
    print(report)


def parse_folder_spec(spec: str) -> tuple[Optional[str], str, str]:
    """
    Parse folder specification in format:
      - 'path:name' (local)
      - 'path' (local, auto-name)
      - 'host:path:name' (remote)
      - 'host:path' (remote, auto-name)

    Returns (host, path, name). host is None for local paths.
    """
    parts = spec.split(":")

    if len(parts) == 1:
        # Just path, local
        path = parts[0]
        name = os.path.basename(path.rstrip("/"))
        return None, path, name

    if len(parts) == 2:
        # Could be "path:name" (local) or "host:path" (remote)
        # Heuristic: if first part looks like a path (starts with / or .), it's local
        if parts[0].startswith("/") or parts[0].startswith("."):
            # Local: path:name
            return None, parts[0], parts[1]
        else:
            # Remote: host:path (auto-name)
            host, path = parts[0], parts[1]
            name = os.path.basename(path.rstrip("/"))
            return host, path, name

    if len(parts) == 3:
        # host:path:name (remote with explicit name)
        return parts[0], parts[1], parts[2]

    if len(parts) > 3:
        # host:path:name where path might have colons... unlikely but handle it
        # Assume last part is name, first part is host, middle is path
        host = parts[0]
        name = parts[-1]
        path = ":".join(parts[1:-1])
        return host, path, name

    # Fallback
    return None, spec, os.path.basename(spec.rstrip("/"))


def main():
    parser = argparse.ArgumentParser(
        description="Compare S2S demo evaluation results from multiple folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Local folders:
    python compare_eval_results.py \\
        --eval_folders "/path/to/eval1:Model A" "/path/to/eval2:Model B" \\
        --output comparison_report.md

    # Remote folders (via SSH):
    python compare_eval_results.py \\
        --eval_folders "cluster1:/path/to/eval1:Model A" "cluster2:/path/to/eval2:Model B" \\
        --output comparison_report.md

    # Mixed local and remote:
    python compare_eval_results.py \\
        --eval_folders "/local/path:Local Model" "remote-host:/remote/path:Remote Model" \\
        --output comparison_report.md

Each eval folder should contain a metrics.json file.
Format:
  - Local:  "/path/to/folder:Display Name" or "/path/to/folder"
  - Remote: "hostname:/path/to/folder:Display Name" or "hostname:/path/to/folder"
        """,
    )
    parser.add_argument(
        "--eval_folders",
        nargs="+",
        required=True,
        help="Evaluation folders: 'path:name', 'host:path:name', or just 'path' / 'host:path'",
    )
    parser.add_argument("--output", type=str, default="comparison_report.md", help="Output Markdown file path")
    parser.add_argument(
        "--metrics_file", type=str, default="metrics.json", help="Name of the metrics file in each folder"
    )

    args = parser.parse_args()

    # Load all metrics
    models = []
    for spec in args.eval_folders:
        host, folder_path, model_name = parse_folder_spec(spec)

        # Build metrics path (use / for remote paths too)
        if folder_path.endswith("/"):
            metrics_path = folder_path + args.metrics_file
        else:
            metrics_path = folder_path + "/" + args.metrics_file

        location_str = f"{host}:{metrics_path}" if host else metrics_path

        metrics = load_metrics(metrics_path, host=host)
        if metrics is None:
            print(f"Skipping {model_name}: could not load metrics from {location_str}")
            continue

        key_metrics = extract_key_metrics(metrics)
        models.append((model_name, key_metrics))
        print(f"Loaded metrics for {model_name} from {location_str}")

    if not models:
        print("Error: No valid metrics files found.")
        return 1

    generate_report(models, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
