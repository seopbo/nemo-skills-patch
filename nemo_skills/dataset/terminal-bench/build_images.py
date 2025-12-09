# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from nemo_skills.inference.eval.terminalbench import get_build_cmd, get_setup_cmd


def read_tasks(jsonl_file):
    """Read all tasks from a test.jsonl file."""
    tasks = []

    with open(jsonl_file, "r") as f:
        for line in f:
            data = json.loads(line)
            tasks.append({"task_id": data["task_id"], "task_dir": data["task_dir"]})

    return tasks


def build(task_id, task_dir, output_dir):
    """Build a Docker container for a task and convert it to SIF format using apptainer."""
    sif_path = os.path.join(output_dir, f"{task_id}.sif")

    # Check if SIF file already exists
    if os.path.exists(sif_path):
        print(f"✓ {task_id} -> {sif_path} (already exists)")
        return True

    try:
        # Build image
        print(f"Building {task_id}...")
        cmd = get_build_cmd(task_id, task_dir, sif_path)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ {task_id} -> {sif_path}")
            return True
        else:
            print(f"✗ Failed to build {task_id}")
            print(f"  Error: {result.stderr.strip()}")
            return False

    except Exception as e:
        print(f"✗ Error building {task_id}: {e}")
        return False


def main():
    """Parse command-line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description="Build Docker images for Terminal-Bench and convert them to Apptainer SIF format."
    )

    parser.add_argument("input_file", help="JSONL file to read task ids from (created by prepare.py)")
    parser.add_argument("output_directory", help="Directory to save SIF files")
    parser.add_argument("--max-workers", "-j", type=int, default=20, help="Number of parallel builds (default: 20)")
    parser.add_argument(
        "--tb-repo", type=str, default="TODO", help="URL of the Terminal-Bench repo to pass to git clone"
    )
    parser.add_argument(
        "--tb-commit", type=str, default="HEAD", help="Which commit to use when cloning the Terminal-Bench repo"
    )

    args = parser.parse_args()

    jsonl_path = args.input_file
    output_dir = args.output_directory
    max_workers = args.max_workers
    tb_repo = args.tb_repo
    tb_commit = args.tb_commit

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read container names from JSONL
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found!")
        sys.exit(1)

    print(f"Reading tasks from {jsonl_path}...")
    tasks = read_tasks(jsonl_path)
    print(f"Found {len(tasks)} tasks to build.")

    # Set up terminal-bench repo
    print(f"Setting up Terminal-Bench repo from {tb_repo}, commit: {tb_commit}")
    cmd = get_setup_cmd(tb_repo, tb_commit)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("✗ Failed to set up terminal-bench repo")
            print(f"  Error: {result.stderr.strip()}")
            return
    except Exception as e:
        print(f"✗ Error while setting up terminal-bench repo: {e}")
        return

    print("Terminal-Bench repo set up successfully. Building tasks.")
    print(f"Output directory: {output_dir}")
    print(f"Using {max_workers} parallel workers.\n")

    # Convert containers in parallel
    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all conversion tasks
        futures = [executor.submit(build, task["task_id"], task["task_dir"], output_dir) for task in tasks]

        # Process completed tasks
        for future in as_completed(futures):
            success = future.result()
            if success:
                successful += 1
            else:
                failed += 1

    print("\nBuild complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(tasks)}")


if __name__ == "__main__":
    main()
