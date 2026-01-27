#!/usr/bin/env python3
"""
Postprocess proof extraction outputs.
Parses model generations and extracts clean proofs.
"""

import argparse
import json


def extract_proof(generation: str) -> str:
    """Extract proof from model generation"""
    # Remove leading/trailing whitespace
    proof = generation.strip()

    # Check if no proof was found
    if "Proof not found" in proof or "proof not found" in proof.lower():
        return None

    return proof


def main():
    parser = argparse.ArgumentParser(description="Postprocess proof extraction")
    parser.add_argument("input_file", help="Input JSONL file with generations")
    parser.add_argument("output_file", help="Output JSONL file with extracted proofs")

    args = parser.parse_args()

    results = []
    proof_found_count = 0
    no_proof_count = 0

    with open(args.input_file, "r") as f_in:
        for line in f_in:
            item = json.loads(line)
            generation = item.get("generation", "")

            # Extract proof
            proof = extract_proof(generation)

            if proof:
                item["extracted_proof"] = proof
                item["proof_extraction_gen"] = generation
                results.append(item)
                proof_found_count += 1
            else:
                no_proof_count += 1

    # Write output
    with open(args.output_file, "w") as f_out:
        for item in results:
            f_out.write(json.dumps(item) + "\n")

    # Print stats
    total = proof_found_count + no_proof_count
    print(f"\n{'=' * 60}")
    print("Proof Extraction Results")
    print(f"{'=' * 60}")
    print(f"Total items: {total}")
    print(f"Proofs found: {proof_found_count} ({proof_found_count / total * 100:.1f}%)")
    print(f"No proof found: {no_proof_count} ({no_proof_count / total * 100:.1f}%)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
