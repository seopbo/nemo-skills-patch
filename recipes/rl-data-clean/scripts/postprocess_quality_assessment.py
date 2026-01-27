#!/usr/bin/env python3
"""
Postprocess quality assessment outputs.
Parses model generations and filters based on quality thresholds.
"""

import argparse
import re
from typing import Any, Dict, Optional

import jsonlines


def parse_score(text: str, field_name: str) -> Optional[int]:
    """Extract a 1-5 score from model output"""
    pattern = rf"{field_name}:\s*(\d+)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        return score if 1 <= score <= 5 else None
    return None


def parse_field(text: str, field_name: str) -> Optional[str]:
    """Extract a field value from model output"""
    pattern = rf"{field_name}:\s*(.+?)(?=\n[A-Z_]+:|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def parse_problem_quality(generation: str) -> Dict[str, Any]:
    """Parse problem quality assessment output"""
    return {
        "clarity_score": parse_score(generation, "CLARITY_SCORE"),
        "completeness_score": parse_score(generation, "COMPLETENESS_SCORE"),
        "rigor_score": parse_score(generation, "MATHEMATICAL_RIGOR_SCORE"),
        "difficulty": parse_field(generation, "DIFFICULTY_ESTIMATE"),
        "overall_quality": parse_field(generation, "OVERALL_PROBLEM_QUALITY"),
        "recommendation": parse_field(generation, "RECOMMENDATION"),
    }


def parse_discussion_quality(generation: str) -> Dict[str, Any]:
    """Parse discussion quality assessment output"""
    return {
        "meaningful_discussion": parse_field(generation, "MEANINGFUL_DISCUSSION"),
        "solution_present": parse_field(generation, "SOLUTION_PRESENT"),
        "solution_clarity": parse_score(generation, "SOLUTION_CLARITY"),
        "coherence": parse_score(generation, "DISCUSSION_COHERENCE"),
        "multiple_approaches": parse_field(generation, "MULTIPLE_APPROACHES"),
        "overall_quality": parse_field(generation, "OVERALL_DISCUSSION_QUALITY"),
        "recommendation": parse_field(generation, "RECOMMENDATION"),
    }


def parse_proof_quality(generation: str) -> Dict[str, Any]:
    """Parse proof quality assessment output"""
    return {
        "correctness": parse_field(generation, "CORRECTNESS"),
        "rigor_score": parse_score(generation, "RIGOR_SCORE"),
        "elegance_score": parse_score(generation, "ELEGANCE_SCORE"),
        "completeness": parse_field(generation, "COMPLETENESS"),
        "multiple_approaches": parse_field(generation, "MULTIPLE_APPROACHES"),
        "overall_quality": parse_field(generation, "OVERALL_PROOF_QUALITY"),
        "recommendation": parse_field(generation, "RECOMMENDATION"),
    }


def parse_imo_readiness(generation: str) -> Dict[str, Any]:
    """Parse IMO readiness assessment output"""
    score_text = parse_field(generation, "IMO_READINESS_SCORE")
    try:
        score = int(score_text) if score_text else None
    except ValueError:
        score = None

    return {
        "olympiad_style": parse_field(generation, "OLYMPIAD_STYLE"),
        "pedagogical_value": parse_field(generation, "PEDAGOGICAL_VALUE"),
        "difficulty_appropriateness": parse_field(generation, "DIFFICULTY_APPROPRIATENESS"),
        "teachability": parse_field(generation, "PROOF_TEACHABILITY"),
        "rl_suitable": parse_field(generation, "SUITABLE_FOR_RL_TRAINING"),
        "imo_readiness_score": score,
        "overall_assessment": parse_field(generation, "OVERALL_ASSESSMENT"),
        "recommendation": parse_field(generation, "FINAL_RECOMMENDATION"),
    }


def meets_problem_quality_threshold(assessment: Dict, thresholds: Dict) -> bool:
    """Check if problem quality meets thresholds"""
    clarity = assessment.get("clarity_score")
    completeness = assessment.get("completeness_score")
    rigor = assessment.get("rigor_score")

    if clarity is None or completeness is None or rigor is None:
        return False

    return (
        clarity >= thresholds.get("min_clarity", 4)
        and completeness >= thresholds.get("min_completeness", 4)
        and rigor >= thresholds.get("min_rigor", 4)
    )


def meets_discussion_quality_threshold(assessment: Dict, thresholds: Dict) -> bool:
    """Check if discussion quality meets thresholds"""
    meaningful = assessment.get("meaningful_discussion", "").lower()
    solution_present = assessment.get("solution_present", "").lower()

    must_have_discussion = thresholds.get("must_have_discussion", True)
    must_have_solution = thresholds.get("must_have_solution", True)

    if must_have_discussion and meaningful != "yes":
        return False

    if must_have_solution and solution_present != "yes":
        return False

    return True


def meets_proof_quality_threshold(assessment: Dict, thresholds: Dict) -> bool:
    """Check if proof quality meets thresholds"""
    correctness = assessment.get("correctness", "").lower()
    rigor = assessment.get("rigor_score")

    if correctness != "correct":
        return False

    if rigor is None:
        return False

    return rigor >= thresholds.get("min_rigor", 4)


def meets_imo_readiness_threshold(assessment: Dict, thresholds: Dict) -> bool:
    """Check if IMO readiness meets thresholds"""
    score = assessment.get("imo_readiness_score")
    olympiad_style = assessment.get("olympiad_style", "").lower()
    teachability = assessment.get("teachability", "").lower()

    if score is None:
        return False

    if score < thresholds.get("min_score", 80):
        return False

    if thresholds.get("must_be_olympiad_style", False):
        if olympiad_style not in ["yes", "borderline"]:
            return False

    if thresholds.get("must_be_teachable", False):
        if teachability != "yes":
            return False

    return True


def filter_by_stage(
    input_file: str,
    output_high: str,
    output_low: str,
    stage: str,
    thresholds: Dict,
):
    """Filter items based on quality thresholds for a specific stage"""

    parse_funcs = {
        "problem_quality": parse_problem_quality,
        "discussion_quality": parse_discussion_quality,
        "proof_quality": parse_proof_quality,
        "imo_readiness": parse_imo_readiness,
    }

    threshold_funcs = {
        "problem_quality": meets_problem_quality_threshold,
        "discussion_quality": meets_discussion_quality_threshold,
        "proof_quality": meets_proof_quality_threshold,
        "imo_readiness": meets_imo_readiness_threshold,
    }

    parse_func = parse_funcs[stage]
    threshold_func = threshold_funcs[stage]

    high_quality = []
    low_quality = []

    with jsonlines.open(input_file) as reader:
        for item in reader:
            generation = item.get("generation", "")
            assessment = parse_func(generation)

            # Add assessment to item
            item["quality_assessment"] = assessment

            # Check thresholds
            if threshold_func(assessment, thresholds):
                high_quality.append(item)
            else:
                low_quality.append(item)

    # Write outputs
    with jsonlines.open(output_high, "w") as writer:
        writer.write_all(high_quality)

    with jsonlines.open(output_low, "w") as writer:
        writer.write_all(low_quality)

    print(f"\n{'=' * 60}")
    print(f"Stage: {stage}")
    print(f"{'=' * 60}")
    print(f"Total items: {len(high_quality) + len(low_quality)}")
    print(
        f"High quality: {len(high_quality)} ({len(high_quality) / (len(high_quality) + len(low_quality)) * 100:.1f}%)"
    )
    print(f"Low quality: {len(low_quality)} ({len(low_quality) / (len(high_quality) + len(low_quality)) * 100:.1f}%)")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Filter items by quality assessment")
    parser.add_argument("input_file", help="Input JSONL file with generations")
    parser.add_argument("output_high", help="Output file for high-quality items")
    parser.add_argument("output_low", help="Output file for low-quality items")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["problem_quality", "discussion_quality", "proof_quality", "imo_readiness"],
        help="Which assessment stage",
    )

    # Threshold arguments
    parser.add_argument("--min-clarity", type=int, default=4)
    parser.add_argument("--min-completeness", type=int, default=4)
    parser.add_argument("--min-rigor", type=int, default=4)
    parser.add_argument("--must-have-discussion", action="store_true", default=True)
    parser.add_argument("--must-have-solution", action="store_true", default=True)
    parser.add_argument("--min-score", type=int, default=80)
    parser.add_argument("--must-be-olympiad-style", action="store_true")
    parser.add_argument("--must-be-teachable", action="store_true")

    args = parser.parse_args()

    thresholds = {
        "min_clarity": args.min_clarity,
        "min_completeness": args.min_completeness,
        "min_rigor": args.min_rigor,
        "must_have_discussion": args.must_have_discussion,
        "must_have_solution": args.must_have_solution,
        "min_score": args.min_score,
        "must_be_olympiad_style": args.must_be_olympiad_style,
        "must_be_teachable": args.must_be_teachable,
    }

    filter_by_stage(
        args.input_file,
        args.output_high,
        args.output_low,
        args.stage,
        thresholds,
    )


if __name__ == "__main__":
    main()
