# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from pathlib import Path

from benchmark_utils import (
    EXTRACT_REGEX,
    SUPPORTED_LANGUAGES,
    Schema,
    copy_other_fields,
    digit_to_letter,
    get_mcq_fields,
    load_include_datasets,
    normalize_entry_field,
)
from tqdm import tqdm


def format_entry(entry, language, il_prompts):
    target_options = [entry[v] for v in Schema.OPTIONS]
    target_question = entry[Schema.QUESTION]
    subject = entry[Schema.SUBJECT]
    expected_answer = digit_to_letter(entry[Schema.ANSWER])
    category = normalize_entry_field(entry, Schema.DOMAIN)
    return {
        "expected_answer": expected_answer,
        "extract_from_boxed": False,
        "extract_regex": EXTRACT_REGEX,
        "subset_for_metrics": language,
        "category": category,
        **get_mcq_fields(
            target_question,
            target_options,
            language,
            subject,
            il_prompts
        ),
        **copy_other_fields(entry),
    }


def write_data_to_file(args, datasets):
    data_dir = Path(__file__).absolute().parent
    output_file = data_dir / f"{args.split}.jsonl"
    with open(output_file, "wt", encoding="utf-8") as fout:
        for dataset, lang in zip(datasets, args.languages):
            for entry in tqdm(dataset, desc=f"Preparing {lang} dataset"):
                entry = format_entry(
                    entry=entry,
                    language=lang,
                    il_prompts=args.il_prompts,
                )
                json.dump(entry, fout, ensure_ascii=False)
                fout.write("\n")


def main(args):
    invalid = set(args.languages) - set(SUPPORTED_LANGUAGES)
    if invalid:
        raise ValueError(f"Unsupported languages: {invalid}")
    datasets = load_include_datasets(args.languages, args.split)
    write_data_to_file(args=args, datasets=datasets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="test",
        choices=("test",),
        help="Dataset split to process.",
    )
    parser.add_argument(
        "--languages",
        default=SUPPORTED_LANGUAGES,
        nargs="+",
        help="Languages to process.",
    )
    # In-Language (IL) Prompts present the prompt instructions in the same language as the sample
    # Default: English Prompts, which provide the prompt instructions in English.
    parser.add_argument(
        "--il_prompts",
        default=False,
        action="store_true",
        help="Use in-language prompts.",
    )
    args = parser.parse_args()
    main(args)
