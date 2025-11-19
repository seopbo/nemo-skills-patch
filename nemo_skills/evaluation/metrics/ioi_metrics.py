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
from collections import defaultdict

from nemo_skills.evaluation.metrics.base import BaseMetrics


class IOIMetrics(BaseMetrics):
    def __init__(self):
        super().__init__()
        self.reset()

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
            return 0.0, {}
        subtask_scores = {}

        for submission in submissions:
            for subtask, result in submission["test_case_results"].items():
                subtask_scores[subtask] = max(subtask_scores.get(subtask, 0), result["score"])
        return sum(subtask_scores.values()), subtask_scores

    def get_metrics(self):
        total_score = 0.0
        self.problem_scores = {}
        for name, submissions in self.predictions_by_problem.items():
            score, subtasks = self.get_problem_score(submissions)
            self.problem_scores[name] = (score, subtasks)
            total_score += score

        per_problem_subtask_scores = {}
        for name, (achieved_total, achieved_subtasks) in self.problem_scores.items():
            submissions = self.predictions_by_problem[name]
            max_subtasks = {}
            for sub in submissions:
                max_subtasks[sub["subtask"]] = sub["subtask_score"]
            max_total = sum(max_subtasks.values())
            per_problem_subtask_scores[name] = {
                "total": {"score": achieved_total, "max_score": max_total},
                "subtasks": {
                    subtask: {"score": achieved, "max_score": max_subtasks[subtask]}
                    for subtask, achieved in achieved_subtasks.items()
                },
            }

        metrics_dict = super().get_metrics()
        for m in metrics_dict.values():
            m["total_score"] = int(total_score)
            m["per_problem_subtask_scores"] = per_problem_subtask_scores
        self.per_problem_subtask_scores = per_problem_subtask_scores
        self.print_problem_scores()
        return metrics_dict

    def reset(self):
        super().reset()
        self.predictions_by_problem = defaultdict(list)
        self.problem_scores = {}
        self.per_problem_subtask_scores = {}

    def evaluations_to_print(self):
        return [f"pass@{self.max_k}"]

    def print_problem_scores(self):
        print("---------------------------------Problem and subtask scores---------------------------------")
        for name, info in self.per_problem_subtask_scores.items():
            total = info["total"]
            print(f"# {name}: {total['score']}/{total['max_score']}")
            for subtask, subinfo in info["subtasks"].items():
                print(f"  {subtask}: {subinfo['score']}/{subinfo['max_score']}")
