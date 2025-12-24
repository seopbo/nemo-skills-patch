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
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from lang_libs import (ANSWER_PLACEHOLDER, EXTRACT_REGEX, get_mcq_format,
                       supported_languages)
from tqdm import tqdm

SUPPORTED_LANGUAGES = supported_languages()


# Dataset schema defined in Hugging Face datasets
class Schema:
    ANSWER: str = "answer"
    LANGUAGE: str = "language"
    DOMAIN: str = "domain"
    QUESTION: str = "question"
    SUBJECT: str = "subject"
    COUNTRY: str = "country"
    REGIONAL_FEATURE: str = "regional_feature"
    LEVEL: str = "level"
    OPTIONS: list[str] = [
        "option_a",
        "option_b",
        "option_c",
        "option_d",
    ]  # `option_{x}` fields are available only for base subset
    CHOICES: str = "choices"  # `choices` field is available only for lite subset


def construct_few_shot_examples(languages, num_few_shot_examples):
    # we will use validation set for few-shot examples
    datasets = [
        load_dataset(f"CohereLabs/include-base-44", lang)["validation"]
        for lang in languages
    ]
    few_shot_examples = {}
    for dataset, lang in zip(datasets, languages):
        subject_dict = defaultdict(list)
        for entry in dataset:
            subject_dict[entry[Schema.SUBJECT]].append(entry)
        few_shot_examples[lang] = {
            subject: subject_dict[subject][:num_few_shot_examples]
            for subject in subject_dict
        }
    return few_shot_examples


def retrieve_few_shot_examples(few_shot_examples, language, subject, num_few_shot_examples):
    retrieved_examples = []

    # If the language is not in the few-shot examples, return an empty list
    if language not in few_shot_examples:
        return retrieved_examples
    
    # Prefer the subject-specific few-shot examples
    if subject in few_shot_examples[language]:
        retrieved_examples.extend(few_shot_examples[language][subject])
    
    # If we still need more examples, use the other subjects
    if len(retrieved_examples) < num_few_shot_examples:
        for s in few_shot_examples[language]:
            if s != subject:
                retrieved_examples.append(few_shot_examples[language][s][0])
            
            if len(retrieved_examples) >= num_few_shot_examples:
                break
    return retrieved_examples
    

def digit_to_letter(digit):
    return chr(ord("A") + int(digit))


def get_mcq_fields(description, question, choices, mcq_format, use_answer_prefix=True):
    options_dict = {digit_to_letter(i): option for i, option in enumerate(choices)}
    options_text = "\n".join(
        f"{letter}. {option}" for letter, option in options_dict.items()
    )
    question = "\n".join(
        [
            description,
            mcq_format.q_label,
            question,
            mcq_format.opt_label,
            options_text,
            mcq_format.answer_prefix if use_answer_prefix else "",
        ]
    )
    return {"question": question, "options": options_text, **options_dict}


def get_other_fields(entry):
    return {
        k: (entry.get(k, "") or "").replace(" ", "_")
        for k in [
            Schema.COUNTRY,
            Schema.REGIONAL_FEATURE,
            Schema.LEVEL,
            Schema.SUBJECT,
            Schema.LANGUAGE,
            Schema.DOMAIN,
        ]
    }


def format_entry(entry, args, language, few_shot_examples):
    if args.subset == "lite":
        choices = entry[Schema.CHOICES]
    else:
        choices = [entry[v] for v in Schema.OPTIONS]
    question = entry[Schema.QUESTION]
    subject = entry[Schema.SUBJECT]
    answer = entry[Schema.ANSWER]  # from 0 to 3
    category = (entry.get(args.category, "") or "").replace(" ", "_")

    mcq_format = get_mcq_format(language, il_prompts=args.il_prompts)
    description = mcq_format.task.format(
        subject=subject, answer_placeholder=ANSWER_PLACEHOLDER
    )
    expected_answer = digit_to_letter(answer)  # Convert from [0 to 3] to [A to D]

    # For CoT, we will use the answer prefix
    use_answer_prefix = True

    if len(few_shot_examples) > 0:
        use_answer_prefix = False
        shots = retrieve_few_shot_examples(few_shot_examples, language, subject, args.num_few_shot_examples)
        for shot in shots:
            q = shot[Schema.QUESTION]
            a = digit_to_letter(shot[Schema.ANSWER])
            options_text = "\n".join(
                f"{digit_to_letter(i)}. {shot[v]}"
                for i, v in enumerate(Schema.OPTIONS)
            )
            few_shot_example_text = "\n".join(
                [
                    mcq_format.q_label,
                    q,
                    mcq_format.opt_label,
                    options_text,
                    ANSWER_PLACEHOLDER.replace("X", a),
                ]
            )
            description += f"\n{few_shot_example_text}"

    return {
        "expected_answer": expected_answer,
        "extract_from_boxed": False,
        "extract_regex": EXTRACT_REGEX,
        "subset_for_metrics": language,
        "category": category,
        **get_mcq_fields(description, question, choices, mcq_format, use_answer_prefix),
        **get_other_fields(entry),
    }


def write_data_to_file(args):
    datasets = [
        load_dataset(f"CohereLabs/include-{args.subset}-44", lang)[args.split]
        for lang in args.languages
    ]

    few_shot_examples = {}
    if args.num_few_shot_examples > 0:
        few_shot_examples = construct_few_shot_examples(
            args.languages, args.num_few_shot_examples
        )
    data_dir = Path(__file__).absolute().parent
    output_file = data_dir / f"{args.split}.jsonl"
    with open(output_file, "wt", encoding="utf-8") as fout:
        for dataset, lang in zip(datasets, args.languages):
            for entry in tqdm(
                dataset, desc=f"Preparing {lang} dataset ({args.subset} subset)"
            ):
                entry = format_entry(entry, args, lang, few_shot_examples)
                json.dump(entry, fout, ensure_ascii=False)
                fout.write("\n")


def main(args):
    invalid = set(args.languages) - set(SUPPORTED_LANGUAGES)
    if invalid:
        raise ValueError(f"Unsupported languages: {invalid}")
    write_data_to_file(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="test",
        choices=("test",),
        help="Dataset split to process.",
    )
    parser.add_argument(
        "--subset",
        default="base",
        choices=("base", "lite"),
        help="Subset of the dataset to process.",
    )
    parser.add_argument(
        "--languages",
        default=SUPPORTED_LANGUAGES,
        nargs="+",
        help="Languages to process.",
    )
    parser.add_argument(
        "--category",
        default=Schema.DOMAIN,
        choices=(
            Schema.LANGUAGE,
            Schema.DOMAIN,
            Schema.SUBJECT,
            Schema.REGIONAL_FEATURE,
        ),
        help="Category for aggregation.",
    )
    parser.add_argument(
        "--il_prompts",  # In-Language (IL) Prompts, which present the prompt instructions in the same language as the sample
        default=False,
        action="store_true",
        help="Use in-language prompts.",  # Default: English Prompts, which provide the prompt instructions in English.
    )
    parser.add_argument(
        "--num_few_shot_examples",
        type=int,
        default=0,
        choices=range(6),
        help="Number of few-shot examples to use.",
    )

    args = parser.parse_args()
    main(args)
