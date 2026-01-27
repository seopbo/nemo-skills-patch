#!/usr/bin/env python3
"""
Postprocess classification output (proof vs non-proof).
"""

import argparse
import re

import jsonlines


def parse_classification(generation: str) -> str:
    """Extract classification from model output"""
    pattern = r"CLASSIFICATION:\s*(proof|not_proof)"
    match = re.search(pattern, generation, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "unknown"


def split_by_classification(input_file: str, proof_file: str, non_proof_file: str):
    """Split problems into proof and non-proof categories"""

    proof_problems = []
    non_proof_problems = []

    with jsonlines.open(input_file) as reader:
        for item in reader:
            generation = item.get("generation", "")
            classification = parse_classification(generation)

            item["classification"] = classification

            if classification == "proof":
                proof_problems.append(item)
            else:
                non_proof_problems.append(item)

    # Write outputs
    with jsonlines.open(proof_file, "w") as writer:
        writer.write_all(proof_problems)

    with jsonlines.open(non_proof_file, "w") as writer:
        writer.write_all(non_proof_problems)

    # Print summary
    total = len(proof_problems) + len(non_proof_problems)
    print(f"\n{'=' * 60}")
    print("Classification Summary")
    print(f"{'=' * 60}")
    print(f"Total problems: {total}")
    print(f"Proof problems: {len(proof_problems)} ({len(proof_problems) / total * 100:.1f}%)")
    print(f"Non-proof problems: {len(non_proof_problems)} ({len(non_proof_problems) / total * 100:.1f}%)")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Split problems by classification")
    parser.add_argument("input_file", help="Input JSONL file with classifications")
    parser.add_argument("proof_file", help="Output file for proof problems")
    parser.add_argument("non_proof_file", help="Output file for non-proof problems")

    args = parser.parse_args()

    split_by_classification(args.input_file, args.proof_file, args.non_proof_file)


if __name__ == "__main__":
    main()
