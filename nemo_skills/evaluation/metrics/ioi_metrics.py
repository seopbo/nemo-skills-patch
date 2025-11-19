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
import json
import os
import re
from collections import defaultdict

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_float, as_int


def extract_final_cpp_block(text):
    pattern = r"```(?:cpp|Cpp)\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1] if matches else ""


class IOIMetrics(BaseMetrics):
    def __init__(self, **kwargs):
        super().__init__()
        self.reset()
        self.cluster_folder = kwargs.get("cluster_folder", None)

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions)
        if predictions:
            self.predictions_by_problem[predictions[0]["name"]].extend(predictions)

    def _get_score_dict(self, p):
        return {"correct": all(r["score"] > 0 for r in p["test_case_results"].values())}

    def get_problem_score(self, submissions) -> float:
        """
        For a given problem (list of submissions), compute the score as follows:
          - For each subtask, take the maximum score over all submissions.
          - Sum these maximum scores to get the problem score.
        """
        if not submissions:
            return 0.0
        subtask_scores = {}

        for submission in submissions:
            for subtask, result in submission["test_case_results"].items():
                subtask_scores[subtask] = max(subtask_scores.get(subtask, 0), result["score"])
        return sum(subtask_scores.values()), subtask_scores

    def extract_info(self, submission) -> dict:
        subtask = submission["subtask"]
        achieved = submission["test_case_results"].get(subtask, {}).get("score", 0.0)
        return {
            "score": achieved,
            "subtask_score": submission.get("subtask_score", 0.0),
            "tokens": submission.get("num_generated_tokens", 0),
            "code": extract_final_cpp_block(submission.get("generation", "")),
        }

    def get_clusters(self, submissions) -> dict:
        clusters = defaultdict(list)

        for submission in submissions:
            subtask = submission["subtask"]
            # Build an IOI-specific output signature: vector of per-test scores for this subtask
            outputs = submission["test_case_results"].get(subtask, {}).get("outputs", [])
            run_outputs = tuple(float(o.get("score", 0.0)) for o in outputs)
            output_key = run_outputs

            extract_info = self.extract_info(submission)
            if output_key not in clusters:
                clusters[output_key] = {
                    "status": {
                        "Test passed": 0,
                        "Test failed": 0,
                    },
                    "codes": [],
                }
            clusters[output_key]["codes"].append(extract_info)

            achieved = extract_info["score"]
            full = extract_info["subtask_score"]
            if achieved == full:
                clusters[output_key]["status"]["Test passed"] += 1
            else:
                clusters[output_key]["status"]["Test failed"] += 1

        return clusters

    def get_metrics(self):
        total_score = 0.0
        self.problem_scores = {}
        for name, submissions in self.predictions_by_problem.items():
            score, subtasks = self.get_problem_score(submissions)
            self.problem_scores[name] = (score, subtasks)
            total_score += score
        self.print_problem_scores()
        metrics_dict = super().get_metrics()
        # Optionally produce clusters to disk, similar to ICPC behaviour
        if self.cluster_folder:
            os.makedirs(self.cluster_folder, exist_ok=True)
            for name, submissions in self.predictions_by_problem.items():
                clusters = self.get_clusters(submissions)
                final_clusters = {}
                for i, (output_key, cluster) in enumerate(clusters.items()):
                    final_clusters[f"cluster_{i + 1}"] = {
                        "output": output_key,
                        "status": cluster["status"],
                        "codes": cluster["codes"],
                    }
                output_file = os.path.join(self.cluster_folder, f"{name}_cluster.jsonl")
                with open(output_file, "w") as f:
                    json.dump(final_clusters, f, indent=4)

        # Summaries per-problem and global totals
        total_submissions = {name: len(subs) for name, subs in self.predictions_by_problem.items()}
        for name, _ in self.predictions_by_problem.items():
            metrics_dict.setdefault(name, {})
            metrics_dict[name]["total_submissions"] = total_submissions[name]
            per_problem_score, _ = self.problem_scores[name]
            metrics_dict[name]["total_score"] = int(round(per_problem_score))
            metrics_dict[name]["average_number_of_runs"] = float(total_submissions[name])

        # Global roll-up
        avg_runs = (sum(total_submissions.values()) / len(total_submissions)) if total_submissions else 0.0
        metrics_dict["total"] = {
            "total_score": int(round(total_score)),
            "average_number_of_runs": avg_runs,
        }
        metrics_dict["total"]["total_submissions"] = int(sum(total_submissions.values()))
        return metrics_dict

    def reset(self):
        super().reset()
        self.predictions_by_problem = defaultdict(list)
        self.problem_scores = {}

    def print_problem_scores(self):
        print("---------------------------------Problem and subtask scores---------------------------------")
        for name, (achieved_total, achieved_subtasks) in self.problem_scores.items():
            submissions = self.predictions_by_problem[name]
            max_subtasks = {}
            for sub in submissions:
                max_subtasks[sub["subtask"]] = sub["subtask_score"]
            max_total = sum(max_subtasks.values())
            print(f"# {name}: {achieved_total}/{max_total}")
            for subtask, achieved in achieved_subtasks.items():
                print(f"  {subtask}: {achieved}/{max_subtasks[subtask]}")

    def evaluations_to_print(self):
        """Returns all problem names, prefixed with total aggregate."""
        return ["total"] + list(self.problem_scores.keys())

    def metrics_to_print(self):
        """Defines which metrics to display and how to format them."""
        metrics_to_print = {
            "total_submissions": as_int,  # per-problem number of submissions
            "total_score": as_int,  # global sum of problem scores
            "average_number_of_runs": as_float,  # global average runs
        }
        return metrics_to_print
