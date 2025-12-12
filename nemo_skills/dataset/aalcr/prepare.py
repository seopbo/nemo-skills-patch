# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import os
import random
from pathlib import Path

import tiktoken
from datasets import load_dataset
from tqdm import tqdm

from nemo_skills.utils import get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))

"""
Usage
# default. setup is aalcr (all).
python prepare.py

# prepare subset aalcr_100k.
python prepare.py --max_context_window 100000 --setup test_100k

or
ns prepare_data \
    --data_dir=/workspace/ns-data \
    --cluster=fei-ord \
    aalcr --max_context_window 100000 --setup test_100k
"""


prompt_template = """BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""


URL = "https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR/resolve/main/extracted_text/AA-LCR_extracted-text.zip"


def construct_prompt(docs, question, prompt_template=prompt_template):
    documents_text = "\n\n".join(f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}" for i, doc in enumerate(docs))
    prompt = prompt_template.format(documents_text=documents_text, question=question)
    return prompt


def count_n_tokens(prompt: str, tokenizer_name: str) -> int:
    """
    count tokens with tokenizer, default is cl100k_base. You can use other tokenizers with AutoTokenizer
    """
    enc = tiktoken.get_encoding(tokenizer_name)
    return len(enc.encode(prompt))


def find_optimal_documents(documents, all_documents, documents_keys, question_text, 
                          max_context_window, current_tokens, tokenizer_name, max_aalcr_tokens):
    """
    Find optimal subset of extra documents to maximize token usage while staying under max_context_window.
    Uses exhaustive search to find the best combination.
    
    Args:
        documents: List of current documents
        all_documents: Dictionary of all available documents
        documents_keys: Set of keys for documents already included
        question_text: The question text
        max_context_window: Maximum allowed tokens
        current_tokens: Current token count
        tokenizer_name: Name of tokenizer to use
    
    Returns:
        Tuple of (optimized_prompt, new_token_count)
    """
    documents_keys_extra = list(set(all_documents.keys()) - set(documents_keys))
    if max_aalcr_tokens < max_context_window:
        documents_keys_extra = documents_keys_extra * ((max_context_window - current_tokens) // (max_aalcr_tokens - current_tokens) + 1)

    random.shuffle(documents_keys_extra)
    
    # Use random greedy approach for fast document selection
    budget = max_context_window * 0.9 - current_tokens
    cumulative_tokens = [sum(all_documents[documents_keys_extra[j]]["tokens"] for j in range(i+1)) for i in range(len(documents_keys_extra))]
    closest_idx = max([i for i in range(len(cumulative_tokens)) if cumulative_tokens[i] < budget], default=-1) if cumulative_tokens else -1
    
    best_docs = documents + [all_documents[i]["document"] for i in documents_keys_extra[:closest_idx+1]]
    best_tokens = count_n_tokens(construct_prompt(best_docs, question_text), tokenizer_name)
    documents_keys_extra = documents_keys_extra[closest_idx+1:]
    assert best_tokens <= max_context_window
    
    # Greedily add documents one by one until we can't fit more
    for i, doc_key in enumerate(documents_keys_extra):
        if best_tokens + all_documents[doc_key]["tokens"] <= max_context_window:
            test_docs = best_docs + [all_documents[doc_key]["document"]]
            test_prompt = construct_prompt(test_docs, question_text)
            test_tokens = count_n_tokens(test_prompt, tokenizer_name)
            
            if test_tokens <= max_context_window:
                best_docs = test_docs
                best_tokens = test_tokens
            elif best_tokens >= max_context_window * 0.95:  
                break
    
    if best_tokens > current_tokens:
        return construct_prompt(best_docs, question_text), best_tokens
    
    # Return original if no improvement found
    return construct_prompt(documents, question_text), current_tokens


def find_actual_file(base_path, target_filename):
    # There is some naming inconsistency in the dataset, we need to find the actual file by trying different encoding variations.
    # This is a workaround to avoid the issue. Will let AA know and update the name issue.
    # TODO: This is a temporary solution and will be removed once the issue is fixed.

    """Find the actual file by trying different encoding variations."""

    # Try the exact filename first
    full_path = os.path.join(base_path, target_filename)
    if os.path.exists(full_path):
        LOG.debug("DEBUG find_actual_file: found exact match")
        return target_filename

    # Try with encoding artifacts (dataset expects clean, files have artifacts)
    replacements = [
        ("'", "ΓÇÖ"),  # ASCII apostrophe to encoding artifact (ord: 39)
        (chr(8217), "ΓÇÖ"),  # Right single quotation mark to encoding artifact (ord: 8217)
        (chr(8216), "ΓÇÖ"),  # Left single quotation mark to encoding artifact (ord: 8216)
        ("—", "ΓÇö"),  # em dash to encoding artifact
        ("–", "ΓÇô"),  # en dash to encoding artifact
        ("ş", "s╠º"),  # Turkish character to combining diacritic
    ]

    filename_with_artifacts = target_filename
    for clean, artifact in replacements:
        filename_with_artifacts = filename_with_artifacts.replace(clean, artifact)

    LOG.debug(f"find_actual_file: converted to {repr(filename_with_artifacts)}")
    if filename_with_artifacts == target_filename:
        LOG.debug("find_actual_file: NO CONVERSION HAPPENED!")
        # Show ALL non-alphanumeric characters
        LOG.debug("All special characters in filename:")
        for i, char in enumerate(target_filename):
            if not (char.isalnum() or char in [" ", "_", ".", "-"]):
                LOG.debug(f"pos {i}: {repr(char)} (ord: {ord(char)})")
    else:
        LOG.debug("find_actual_file: conversion successful")

    # Only try artifact version if it's different from original
    if filename_with_artifacts != target_filename:
        artifact_path = os.path.join(base_path, filename_with_artifacts)
        if os.path.exists(artifact_path):
            LOG.debug("find_actual_file: found artifact match")
            return filename_with_artifacts

        LOG.debug("find_actual_file: artifact version doesn't exist")

    # If still not found, try listing directory and matching by normalization
    try:
        for actual_file in os.listdir(base_path):
            # Create normalized versions for comparison
            normalized_target = target_filename
            normalized_actual = actual_file

            # Normalize target by converting clean chars to artifacts
            for clean, artifact in replacements:
                normalized_target = normalized_target.replace(clean, artifact)

            # Normalize actual by converting artifacts to clean chars
            for clean, artifact in replacements:
                normalized_actual = normalized_actual.replace(artifact, clean)

            # Check if they match after normalization
            if normalized_target == actual_file or target_filename == normalized_actual:
                LOG.debug(f"DEBUG find_actual_file: found directory match {repr(actual_file)}")
                return actual_file
    except OSError:
        pass

    LOG.debug("DEBUG find_actual_file: no match found, returning original")
    return target_filename  # Return original if nothing found


def write_data_to_file(output_file, data, txt_file_folder, max_context_window, tokenizer_name):
    all_documents = {}
    for idx, entry in tqdm(enumerate(data), desc=f"Reading all docs"):
        document_set_id = entry["document_set_id"]
        document_category = entry["document_category"]
        data_source_filenames = entry["data_source_filenames"].split(";")
        for data_source_filename in data_source_filenames:
            base_path = f"{txt_file_folder}/{document_category}/{document_set_id}"
            actual_filename = find_actual_file(base_path, data_source_filename)
            key = f"{base_path}/{actual_filename}"
            if key in all_documents:
                continue
            try:
                with open(
                    key,
                    "rt",
                    encoding="utf-8",
                ) as fin:
                    document = fin.read()
                    all_documents[key] = {
                        "document": document,
                        "tokens": count_n_tokens(document, tokenizer_name)
                    }

            except FileNotFoundError:
                if actual_filename != data_source_filename:
                    LOG.debug(f"File {base_path}/{data_source_filename} is missing")
    
    max_aalcr_tokens = count_n_tokens(construct_prompt([item["document"] for item in all_documents.values()], question=data[0]["question"]), tokenizer_name)
    LOG.info(f"Found total {len(all_documents)} documents")
    LOG.info(f"Max tokens of all documents: {max_aalcr_tokens}")

    with open(output_file, "wt", encoding="utf-8") as fout:
        for idx, entry in tqdm(enumerate(data), desc=f"Writing {output_file.name}"):
            entry["index"] = entry.pop("question_id")
            document_set_id = entry.pop("document_set_id")
            document_category = entry["document_category"]
            data_source_filenames = entry.pop("data_source_filenames").split(";")

            # Collect documents
            documents = []
            documents_keys = []
            for data_source_filename in data_source_filenames:
                base_path = f"{txt_file_folder}/{document_category}/{document_set_id}"
                actual_filename = find_actual_file(base_path, data_source_filename)
                key = f"{base_path}/{actual_filename}"
                if key in all_documents:
                    documents.append(all_documents[key]["document"])
                    documents_keys.append(key)

            # Use construct_prompt to format the question with documents
            question_text = entry.pop("question")
            question = construct_prompt(documents, question_text)

            # find n_tokens with tokenizer_name
            n_tokens = count_n_tokens(question, tokenizer_name)

            if max_context_window is not None:
                # Find optimal subset of documents to maximize token usage
                question, n_tokens = find_optimal_documents(
                    documents, all_documents, documents_keys, question_text,
                    max_context_window, n_tokens, tokenizer_name, max_aalcr_tokens
                )

            elif n_tokens != entry["input_tokens"]:  # check if the n_tokens exactly match the input_tokens in the entry
                LOG.warning(f"n_tokens: {n_tokens} != input_tokens: {entry['input_tokens']}")

            entry[f"n_tokens_{tokenizer_name}"] = n_tokens
            entry["question"] = question
            entry["original_question"] = question_text
            entry["expected_answer"] = entry.pop("answer")
            entry["expected_judgement"] = "correct"  # for judgement metric
            # remove unused columns
            entry.pop("data_source_urls")

            fout.write(json.dumps(entry) + "\n")


def prepare_aalcr_data(max_context_window, setup, tokenizer_name):
    # download the provied extracted text files
    # https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR/resolve/main/extracted_text/AA-LCR_extracted-text.zip

    if not os.path.exists(Path(__file__).absolute().parent / "lcr"):
        import zipfile

        import wget

        wget.download(URL)
        zipfile.ZipFile("AA-LCR_extracted-text.zip").extractall(Path(__file__).absolute().parent)
        os.remove("AA-LCR_extracted-text.zip")

    txt_file_folder = Path(__file__).absolute().parent / "lcr"

    dataset = load_dataset("ArtificialAnalysis/AA-LCR")["test"]

    data_dir = Path(__file__).absolute().parent

    output_file = data_dir / f"{setup}.jsonl"
    write_data_to_file(output_file, dataset, txt_file_folder, max_context_window, tokenizer_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare AALCR dataset.")
    parser.add_argument(
        "--max_context_window",
        type=int,
        default=None,
        help="Maximum context window size.",
    )
    parser.add_argument(
        "--setup",
        type=str,
        default="test",
        help="setup name. e.g. test or test_64k",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="cl100k_base",
        help="tokenizer name",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()
    random.seed(args.random_seed)

    # Setup logging
    setup_logging(log_level=logging.DEBUG if args.debug else logging.INFO)

    LOG.info(f"Preparing AA-LCR dataset with additional arguments: {args}")
    prepare_aalcr_data(args.max_context_window, args.setup, args.tokenizer_name)
    LOG.info(f"AA-LCR dataset preparation with setup {args.setup} completed. Use --split={args.setup} to evaluate!")
