#!/usr/bin/env python3
"""
Script to recalculate LiveCodeBench metrics.

Runs code grading on output files that are missing graded_list,
then calculates final metrics.

Usage:
    python scripts/recalculate_livecodebench.py <input_dir> [--output_dir <output_dir>]

Examples:
    python scripts/recalculate_livecodebench.py data/ns_livecodebench/artifacts/eval-results
    python scripts/recalculate_livecodebench.py data/ns_livecodebench/artifacts/eval-results --output_dir data/results
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def needs_grading(livecodebench_dir: Path) -> list:
    """Check which files need code grading."""
    files_to_grade = []
    
    jsonl_files = list(livecodebench_dir.glob("*.jsonl"))
    
    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r') as f:
            first_line = f.readline()
            if first_line:
                data = json.loads(first_line)
                if 'graded_list' not in data:
                    files_to_grade.append(jsonl_file)
                    
    return files_to_grade


def run_grading(jsonl_files: list, num_processes: int = 12, timeout: int = 6):
    """Run LiveCodeBench code grading on output files."""
    from nemo_skills.evaluation.evaluator.livecodebench import eval_livecodebench
    
    for jsonl_file in sorted(jsonl_files):
        print(f"\n{'='*60}")
        print(f"Grading: {jsonl_file.name}")
        print(f"{'='*60}")
        try:
            eval_livecodebench({
                'input_file': str(jsonl_file),
                'language': 'python',
                'num_processes': num_processes,
                'timeout': timeout,
            })
            print(f"✅ Completed {jsonl_file.name}")
        except Exception as e:
            print(f"❌ Error grading {jsonl_file.name}: {e}")


def run_metrics(eval_results_dir: Path, output_dir: Path) -> Path:
    """Run metrics calculation."""
    metrics_path = output_dir / "metrics.json"
    
    cmd = [
        "ns", "summarize_results",
        str(eval_results_dir),
        "--save_metrics_path", str(metrics_path)
    ]
    
    print(f"\n{'='*60}")
    print("Calculating metrics...")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0 and metrics_path.exists():
        return metrics_path
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Recalculate LiveCodeBench metrics (runs code grading if needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path to eval-results directory containing livecodebench/"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for metrics.json (default: <input>_recalculated)"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=12,
        help="Number of parallel processes for code grading (default: 12)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=6,
        help="Timeout in seconds per test case (default: 6)"
    )
    parser.add_argument(
        "--skip_grading",
        action="store_true",
        help="Skip code grading, only recalculate metrics"
    )
    
    args = parser.parse_args()
    
    # Find livecodebench directory
    if args.input_dir.name == "livecodebench":
        livecodebench_dir = args.input_dir
        eval_results_dir = args.input_dir.parent
    elif (args.input_dir / "livecodebench").exists():
        livecodebench_dir = args.input_dir / "livecodebench"
        eval_results_dir = args.input_dir
    else:
        print(f"Error: Cannot find livecodebench directory in {args.input_dir}")
        sys.exit(1)
    
    print(f"LiveCodeBench dir: {livecodebench_dir}")
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = eval_results_dir.parent / f"{eval_results_dir.parent.name}_recalculated"
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {args.output_dir}")
    
    # Check if grading is needed
    if not args.skip_grading:
        files_to_grade = needs_grading(livecodebench_dir)
        
        if files_to_grade:
            print(f"\n{len(files_to_grade)} files need code grading")
            run_grading(files_to_grade, args.num_processes, args.timeout)
        else:
            print("\n✅ All files already have graded_list")
    
    # Calculate metrics
    metrics_path = run_metrics(eval_results_dir, args.output_dir)
    
    if metrics_path:
        print(f"\n{'='*60}")
        print(f"✅ Metrics saved to: {metrics_path}")
        print(f"{'='*60}")
    else:
        print("\n❌ Failed to calculate metrics")
        sys.exit(1)


if __name__ == "__main__":
    main()

