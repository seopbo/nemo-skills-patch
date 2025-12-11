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

"""Prepare ASR Leaderboard datasets for evaluation.

Downloads and formats datasets from the HuggingFace Open ASR Leaderboard.
Supports multiple ASR benchmark datasets:
- AMI
- Earnings22
- Gigaspeech
- LS Clean
- LS Other
- SPGISpeech
- Tedlium
- Voxpopuli

Usage:
    # Prepare JSONL only (no audio download) - fast, for testing
    ns prepare_data asr-leaderboard --no-audio

    # Prepare with audio (default) - downloads audio files
    ns prepare_data asr-leaderboard

    # Prepare specific datasets
    ns prepare_data asr-leaderboard --datasets librispeech_clean librispeech_other
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


# System message for ASR transcription
SYSTEM_MESSAGE = "You are a helpful assistant. /no_think"

# Available datasets and their HuggingFace configurations (HF Open ASR Leaderboard)
# Format: (hf_dataset_name, hf_config, hf_split)
DATASET_CONFIGS = {
    "librispeech_clean": ("librispeech_asr", "clean", "test"),
    "librispeech_other": ("librispeech_asr", "other", "test"),
    "voxpopuli": ("facebook/voxpopuli", "en", "test"),
    "tedlium": ("LIUM/tedlium", "release3", "test"),
    "gigaspeech": ("speechcolab/gigaspeech", "xs", "test"),
    "spgispeech": ("kensho/spgispeech", "S", "test"),
    "earnings22": ("revdotcom/earnings22", None, "test"),
    "ami": ("edinburghcstr/ami", "ihm", "test"),
}


def format_entry(entry, dataset_name, data_dir, with_audio=True):
    """Format a dataset entry for ASR evaluation.

    Args:
        entry: Raw dataset entry with audio and text
        dataset_name: Name of the dataset for subset tracking
        data_dir: Base directory for relative paths
        with_audio: Whether to include audio attachment in messages

    Returns:
        Formatted entry dict with proper fields for audio evaluation
    """
    # Get the transcription text - different datasets use different field names
    text = entry.get("text", "") or entry.get("sentence", "") or entry.get("transcript", "")
    text = text.strip()

    # Create messages with system and user
    system_message = {"role": "system", "content": SYSTEM_MESSAGE}
    user_message = {
        "role": "user",
        "content": "Transcribe the following audio.",
    }

    # Add audio attachment only if with_audio is enabled
    if with_audio:
        audio_info = entry.get("audio", {})
        if isinstance(audio_info, dict):
            audio_path = audio_info.get("path", "")
        else:
            audio_path = str(audio_info) if audio_info else ""

        if audio_path:
            # Convert absolute path to relative path from data_dir
            audio_path = Path(audio_path)
            try:
                # Make path relative to data_dir
                relative_path = audio_path.relative_to(data_dir)
                audio_path = str(relative_path)
            except ValueError:
                # Path is not under data_dir, keep as-is
                audio_path = str(audio_path)

            # Get audio duration if available
            duration = entry.get("duration")
            if duration is None and isinstance(audio_info, dict):
                duration = audio_info.get("duration")

            audio_attachment = {"path": audio_path}
            if duration:
                audio_attachment["duration"] = float(duration)
            user_message["audio"] = audio_attachment

    formatted_entry = {
        "task_type": "ASR_LEADERBOARD",
        "expected_answer": text,
        "messages": [system_message, user_message],
        "subset_for_metrics": dataset_name,
    }

    # Preserve useful metadata
    if "id" in entry:
        formatted_entry["id"] = entry["id"]
    if "speaker_id" in entry:
        formatted_entry["speaker_id"] = entry["speaker_id"]

    return formatted_entry


def prepare_dataset(dataset_name, data_dir, max_samples=None, with_audio=True):
    """Prepare a single ASR dataset.

    Args:
        dataset_name: Key from DATASET_CONFIGS
        data_dir: Directory to save output files
        max_samples: Optional limit on number of samples
        with_audio: Whether to include audio in the output

    Returns:
        Number of samples processed
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    hf_dataset, hf_config, hf_split = DATASET_CONFIGS[dataset_name]

    print(f"Loading {dataset_name} from {hf_dataset} (split={hf_split} only)...")
    try:
        # Use split parameter to only download the test split, not train/validation
        load_kwargs = {
            "split": hf_split,
            "trust_remote_code": True,
        }
        if hf_config:
            dataset = load_dataset(hf_dataset, hf_config, cache_dir=data_dir, **load_kwargs)
        else:
            dataset = load_dataset(hf_dataset, cache_dir=data_dir, **load_kwargs)
    except Exception as e:
        print(f"Warning: Failed to load {dataset_name}: {e}")
        return 0

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    output_file = data_dir / f"{dataset_name}.jsonl"

    print(f"Processing {len(dataset)} samples from {dataset_name}...")

    count = 0
    with open(output_file, "w", encoding="utf-8") as fout:
        for entry in tqdm(dataset, desc=dataset_name):
            formatted = format_entry(entry, dataset_name, data_dir, with_audio=with_audio)
            if formatted["expected_answer"]:  # Skip entries without transcription
                fout.write(json.dumps(formatted) + "\n")
                count += 1

    print(f"Saved {count} samples to {output_file}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare ASR Leaderboard datasets for evaluation")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=list(DATASET_CONFIGS.keys()) + ["all"],
        help="Datasets to prepare (default: all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per dataset (for testing)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Output split name (default: test)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip audio paths in output (creates JSONL with text only, for faster preparation)",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    with_audio = not args.no_audio

    if args.no_audio:
        print("Running in text-only mode (--no-audio). Audio paths will not be included.")
    else:
        print("Running with audio. This may take a while to download audio files.")

    # Handle "all" option
    if "all" in args.datasets:
        datasets_to_prepare = list(DATASET_CONFIGS.keys())
    else:
        datasets_to_prepare = args.datasets

    # Prepare each dataset
    total_samples = 0
    for dataset_name in datasets_to_prepare:
        count = prepare_dataset(dataset_name, data_dir, args.max_samples, with_audio=with_audio)
        total_samples += count

    # Create combined test file with all datasets
    if len(datasets_to_prepare) > 1:
        combined_file = data_dir / f"{args.split}.jsonl"
        print(f"\nCreating combined file: {combined_file}")

        with open(combined_file, "w", encoding="utf-8") as fout:
            for dataset_name in datasets_to_prepare:
                dataset_file = data_dir / f"{dataset_name}.jsonl"
                if dataset_file.exists():
                    with open(dataset_file, encoding="utf-8") as fin:
                        for line in fin:
                            fout.write(line)

        print(f"Combined {total_samples} samples into {combined_file}")
    elif len(datasets_to_prepare) == 1:
        # Rename single dataset file to test.jsonl
        src_file = data_dir / f"{datasets_to_prepare[0]}.jsonl"
        dst_file = data_dir / f"{args.split}.jsonl"
        if src_file.exists() and src_file != dst_file:
            src_file.rename(dst_file)
            print(f"Renamed to {dst_file}")

    print(f"\nTotal: {total_samples} samples prepared")


if __name__ == "__main__":
    main()
