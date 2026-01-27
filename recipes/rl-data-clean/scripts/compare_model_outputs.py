#!/usr/bin/env python3
"""
Compare outputs from two different models.
Used for Phase 1 experiment to determine if dual-model validation is beneficial.
"""

import argparse
import json
import sys
from collections import defaultdict

import jsonlines


def parse_assessment(generation: str, stage: str) -> dict:
    """Parse quality assessment from generation"""
    import re

    def extract_field(field_name):
        pattern = rf"{field_name}:\s*(.+?)(?=\n[A-Z_]+:|$)"
        match = re.search(pattern, generation, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else None

    def extract_score(field_name):
        pattern = rf"{field_name}:\s*(\d+)"
        match = re.search(pattern, generation, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            return score if 1 <= score <= 5 else None
        return None

    if stage == "proof_quality":
        return {
            "correctness": extract_field("CORRECTNESS"),
            "rigor_score": extract_score("RIGOR_SCORE"),
            "elegance_score": extract_score("ELEGANCE_SCORE"),
            "overall_quality": extract_field("OVERALL_PROOF_QUALITY"),
            "recommendation": extract_field("RECOMMENDATION"),
        }
    elif stage == "imo_readiness":
        score_text = extract_field("IMO_READINESS_SCORE")
        try:
            score = int(score_text) if score_text else None
        except (ValueError, TypeError):
            score = None
        return {
            "olympiad_style": extract_field("OLYMPIAD_STYLE"),
            "imo_readiness_score": score,
            "overall_assessment": extract_field("OVERALL_ASSESSMENT"),
            "recommendation": extract_field("FINAL_RECOMMENDATION"),
        }
    else:
        return {}


def compare_correctness(model1_val: str, model2_val: str) -> dict:
    """Compare correctness assessments"""
    model1_val = (model1_val or "").lower().strip()
    model2_val = (model2_val or "").lower().strip()

    agree = model1_val == model2_val
    return {
        "agree": agree,
        "model1": model1_val,
        "model2": model2_val,
    }


def compare_scores(model1_score: int, model2_score: int, tolerance: int = 1) -> dict:
    """Compare numerical scores with tolerance"""
    if model1_score is None or model2_score is None:
        return {"agree": False, "model1": model1_score, "model2": model2_score}

    diff = abs(model1_score - model2_score)
    agree = diff <= tolerance

    return {
        "agree": agree,
        "diff": diff,
        "model1": model1_score,
        "model2": model2_score,
    }


def compare_models(model1_file: str, model2_file: str, stage: str, output_file: str):
    """Compare two model outputs"""

    # Load data
    model1_data = {item["problem_id"]: item for item in jsonlines.open(model1_file)}
    model2_data = {item["problem_id"]: item for item in jsonlines.open(model2_file)}

    common_ids = set(model1_data.keys()) & set(model2_data.keys())

    if not common_ids:
        print("ERROR: No common problem IDs found between the two files!")
        sys.exit(1)

    print(f"Comparing {len(common_ids)} problems...")

    # Statistics
    stats = {
        "total_problems": len(common_ids),
        "agreements": defaultdict(int),
        "disagreements": defaultdict(int),
        "disagreement_cases": [],
    }

    for prob_id in sorted(common_ids):
        item1 = model1_data[prob_id]
        item2 = model2_data[prob_id]

        gen1 = item1.get("generation", "")
        gen2 = item2.get("generation", "")

        assess1 = parse_assessment(gen1, stage)
        assess2 = parse_assessment(gen2, stage)

        case = {
            "problem_id": prob_id,
            "problem": item1.get("problem", ""),
            "model1": assess1,
            "model2": assess2,
            "comparisons": {},
        }

        # Compare based on stage
        if stage == "proof_quality":
            # Compare correctness
            corr_comp = compare_correctness(assess1["correctness"], assess2["correctness"])
            case["comparisons"]["correctness"] = corr_comp

            if corr_comp["agree"]:
                stats["agreements"]["correctness"] += 1
            else:
                stats["disagreements"]["correctness"] += 1

            # Compare rigor
            rigor_comp = compare_scores(assess1["rigor_score"], assess2["rigor_score"])
            case["comparisons"]["rigor"] = rigor_comp

            if rigor_comp["agree"]:
                stats["agreements"]["rigor"] += 1
            else:
                stats["disagreements"]["rigor"] += 1

            # Compare elegance
            eleg_comp = compare_scores(assess1["elegance_score"], assess2["elegance_score"])
            case["comparisons"]["elegance"] = eleg_comp

            if eleg_comp["agree"]:
                stats["agreements"]["elegance"] += 1
            else:
                stats["disagreements"]["elegance"] += 1

            # Overall agreement: all three must agree
            overall_agree = corr_comp["agree"] and rigor_comp["agree"] and eleg_comp["agree"]

        elif stage == "imo_readiness":
            # Compare olympiad style
            style_comp = compare_correctness(assess1["olympiad_style"], assess2["olympiad_style"])
            case["comparisons"]["olympiad_style"] = style_comp

            if style_comp["agree"]:
                stats["agreements"]["olympiad_style"] += 1
            else:
                stats["disagreements"]["olympiad_style"] += 1

            # Compare IMO readiness score
            score_comp = compare_scores(assess1["imo_readiness_score"], assess2["imo_readiness_score"], tolerance=10)
            case["comparisons"]["imo_readiness_score"] = score_comp

            if score_comp["agree"]:
                stats["agreements"]["imo_readiness_score"] += 1
            else:
                stats["disagreements"]["imo_readiness_score"] += 1

            overall_agree = style_comp["agree"] and score_comp["agree"]

        else:
            overall_agree = False

        case["overall_agreement"] = overall_agree

        if not overall_agree:
            stats["disagreement_cases"].append(case)

    # Calculate agreement rates
    stats["agreement_rates"] = {}
    for key in stats["agreements"]:
        total = stats["agreements"][key] + stats["disagreements"][key]
        if total > 0:
            stats["agreement_rates"][key] = stats["agreements"][key] / total

    overall_agreements = sum(
        1 for case in [model1_data[pid] for pid in common_ids] if case.get("overall_agreement", False)
    )
    stats["overall_agreement_rate"] = overall_agreements / len(common_ids) if common_ids else 0

    # Save results
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"Model Comparison Results: {stage}")
    print(f"{'=' * 70}")
    print(f"Total problems compared: {stats['total_problems']}")
    print("\nAgreement Rates:")
    for key, rate in stats["agreement_rates"].items():
        print(f"  {key:30s}: {rate:6.1%}")
    print(f"\nDisagreement cases: {len(stats['disagreement_cases'])}")
    print(f"  → Saved to: {output_file}")
    print(f"{'=' * 70}\n")

    # Recommendation
    overall_rate = stats.get("overall_agreement_rate", 0)
    print("RECOMMENDATION:")
    if overall_rate >= 0.90:
        print("  ✅ Models agree >90% of the time")
        print("  → Single model is likely sufficient")
    elif overall_rate >= 0.80:
        print("  ⚠️  Models agree 80-90% of the time")
        print("  → Dual-model validation recommended (take intersection)")
    else:
        print("  🔴 Models agree <80% of the time")
        print("  → Dual-model validation STRONGLY recommended")
        print("  → Consider manual review of disagreements")
    print()

    return stats


def main():
    parser = argparse.ArgumentParser(description="Compare outputs from two models")
    parser.add_argument("model1_file", help="Output file from model 1")
    parser.add_argument("model2_file", help="Output file from model 2")
    parser.add_argument("output_file", help="Output comparison results (JSON)")
    parser.add_argument(
        "--stage", required=True, choices=["proof_quality", "imo_readiness"], help="Which assessment stage to compare"
    )
    parser.add_argument("--model1-name", default="Model 1", help="Name of model 1")
    parser.add_argument("--model2-name", default="Model 2", help="Name of model 2")

    args = parser.parse_args()

    compare_models(
        args.model1_file,
        args.model2_file,
        args.stage,
        args.output_file,
    )


if __name__ == "__main__":
    main()
