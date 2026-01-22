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

"""Prepare Numb3rs dataset for TN/ITN evaluation.

Numb3rs is a speech dataset for text normalization (TN) and inverse text normalization (ITN) tasks,
containing paired written/spoken forms with corresponding synthetic audio.

Dataset: https://huggingface.co/datasets/NNstuff/Numb3rs

Usage:
    ns prepare_data numb3rs
    ns prepare_data numb3rs --categories CARDINAL DATE MONEY
    ns prepare_data numb3rs --no-audio  # skip saving audio files
"""

import argparse
import json
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

SYSTEM_MESSAGE = "You are a helpful assistant. /no_think"

# Prompt variants for TN/ITN evaluation
PROMPT_NEUTRAL = "Transcribe the audio file into English text."
PROMPT_TN = "Transcribe the audio into written form with numbers as digits (e.g., '$100', '3.14', 'Jan 1')."
PROMPT_ITN = "Transcribe the audio into spoken form with numbers spelled out (e.g., 'one hundred dollars', 'three point one four', 'january first')."

MIN_AUDIO_DURATION = 0.1  # Skip audio shorter than this


def save_audio_and_format_entry(entry, category, audio_dir, sample_idx, with_audio=True):
    """Format a dataset entry and optionally save audio file."""
    # Extract fields from Numb3rs dataset
    original_text = entry.get("original_text", "").strip()  # Written form (TN)
    text = entry.get("text", "").strip()  # Spoken form (ITN)

    if not original_text or not text:
        return None

    # Get audio info
    audio_info = entry.get("audio", {})
    if not isinstance(audio_info, dict) or "array" not in audio_info or "sampling_rate" not in audio_info:
        return None

    audio_array = audio_info["array"]
    sampling_rate = audio_info["sampling_rate"]

    # Skip if audio array is empty or invalid
    if audio_array is None or len(audio_array) == 0:
        return None

    duration = len(audio_array) / sampling_rate

    if duration < MIN_AUDIO_DURATION:
        return None

    # Create sample ID
    sample_id = entry.get("name", f"{category}_{sample_idx}")
    audio_filename = f"{sample_id}.flac"

    # Save audio file
    if with_audio:
        audio_file_path = audio_dir / audio_filename
        sf.write(str(audio_file_path), audio_array, sampling_rate)

    # Container path for evaluation
    audio_filepath = f"/dataset/numb3rs/data/{category}/{audio_filename}"

    # Build messages with placeholder content
    system_message = {"role": "system", "content": SYSTEM_MESSAGE}
    user_message = {
        "role": "user",
        "content": "<PLACEHOLDER>",  # Will be replaced at runtime based on prompt_field
        "audio": {
            "path": audio_filepath,
            "duration": float(duration),
        },
    }

    formatted_entry = {
        "audio_filepath": audio_filepath,
        "duration": float(duration),
        "text_tn": original_text,
        "text_itn": text,
        "expected_answer": text,  # Default to ITN for backward compatibility
        "task_type": "ASR_LEADERBOARD",
        "category": category,
        "sample_id": sample_id,
        "subset_for_metrics": f"numb3rs_{category}",
        "messages": [system_message, user_message],
        # Prompt variants
        "prompt_neutral": PROMPT_NEUTRAL,
        "prompt_tn": PROMPT_TN,
        "prompt_itn": PROMPT_ITN,
    }

    return formatted_entry


def prepare_category(category, dataset, output_dir, with_audio=True):
    """Prepare a single category from the Numb3rs dataset."""
    print(f"\nProcessing category: {category}")

    # Filter dataset by category
    category_samples = [s for s in dataset if s.get("category", "").upper() == category.upper()]

    if not category_samples:
        print(f"No samples found for category: {category}")
        return 0

    print(f"Found {len(category_samples)} samples")

    # Create output directories
    audio_dir = output_dir / "data" / category
    category_dir = output_dir

    if with_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving audio files to {audio_dir}")

    category_dir.mkdir(parents=True, exist_ok=True)

    # Process samples
    count = 0
    skipped = 0
    output_file = category_dir / f"{category}.jsonl"

    with open(output_file, "w", encoding="utf-8") as fout:
        for idx, entry in enumerate(tqdm(category_samples, desc=f"Processing {category}")):
            formatted = save_audio_and_format_entry(entry, category, audio_dir, idx, with_audio=with_audio)
            if formatted is None:
                skipped += 1
                continue

            fout.write(json.dumps(formatted) + "\n")
            count += 1

    if skipped > 0:
        print(f"Skipped {skipped} samples (short audio or invalid)")

    print(f"Saved {count} samples to {output_file}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare Numb3rs dataset for TN/ITN evaluation")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["all"],
        help="Categories to prepare (default: all). Available: ADDRESS, CARDINAL, DATE, DECIMAL, DIGIT, FRACTION, MEASURE, MONEY, ORDINAL, PLAIN, TELEPHONE, TIME",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip saving audio files (JSONL still includes audio paths)",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with_audio = not args.no_audio

    if args.no_audio:
        print("Running without saving audio files.")
    else:
        print("Running with audio. Saving to data/{category}/")

    # Load dataset from HuggingFace
    print("\nLoading Numb3rs dataset from HuggingFace...")
    try:
        dataset = load_dataset("NNstuff/Numb3rs", split="train", trust_remote_code=True)
        print(f"Loaded {len(dataset)} total samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Get all available categories
    all_categories = sorted(set(s.get("category", "").upper() for s in dataset if s.get("category")))
    print(f"Available categories: {', '.join(all_categories)}")

    # Determine which categories to process
    if "all" in args.categories:
        categories_to_prepare = all_categories
    else:
        categories_to_prepare = [c.upper() for c in args.categories]
        # Validate categories
        invalid = set(categories_to_prepare) - set(all_categories)
        if invalid:
            print(f"Warning: Unknown categories will be skipped: {invalid}")
            categories_to_prepare = [c for c in categories_to_prepare if c in all_categories]

    if not categories_to_prepare:
        print("No valid categories to process")
        return

    # Process each category
    total_samples = 0
    for category in categories_to_prepare:
        total_samples += prepare_category(category, dataset, output_dir, with_audio=with_audio)

    # Combine all category JSONLs into test.jsonl
    combined_file = output_dir / "test.jsonl"
    print(f"\nCreating combined file: {combined_file}")

    all_jsonl_files = sorted(output_dir.glob("*.jsonl"))
    category_files = [f for f in all_jsonl_files if f.name != "test.jsonl"]

    combined_count = 0
    with open(combined_file, "w", encoding="utf-8") as fout:
        for category_file in category_files:
            with open(category_file, encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    combined_count += 1
            print(f"  Added {category_file.name}")

    print(f"Combined {combined_count} samples from {len(category_files)} categories into {combined_file}")
    print(f"\nTotal: {total_samples} samples prepared")


if __name__ == "__main__":
    main()
