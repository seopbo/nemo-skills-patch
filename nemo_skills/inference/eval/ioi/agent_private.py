import logging
import re
import sys
import time
from dataclasses import field

import hydra
import litellm

from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


def extract_code_block(text: str):
    # todo (sean): this is a hack to prevent catching report tags in the CoT, causing parsing errors for gpt-oss.
    text = text.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]
    matches = re.findall(r"```cpp(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


@nested_dataclass(kw_only=True)
class IOIExecutionConfig(GenerateSolutionsConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)
    prompt_config: str = "eval/ioi/agent/solver"
    improve_after_verify_prompt_config: str = "eval/ioi/agent/improve_with_private_test"
    total_steps: int = 60
    time_limit: str | None = None  # Optional wall-clock time limit in 'HH:MM:SS'
    show_k_solutions: int = 0  # Number of previous solutions to include in improve prompt
    retry_solution: int = 5  # Retry count when code block extraction fails


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_ioi_generation_config", node=IOIExecutionConfig)


class IOIExecutionGenerationTask(GenerationTask):
    # Shared formats for composing solution blocks
    SOLUTION_BLOCK_TEMPLATE = (
        "## Solution{title_suffix} ##\n{code}\n## Test Feedback ##\n{feedback}\n## Solution Score ## {score}"
    )
    PREVIOUS_PREAMBLE = (
        "Here are some previous solutions that were submitted to help you design and improve the solution:"
    )

    def __init__(self, cfg: IOIExecutionConfig):
        super().__init__(cfg)
        prompt_kwargs = {
            "examples_type": cfg.examples_type,
        }
        self.prompts = {
            "initial": get_prompt(cfg.prompt_config, **prompt_kwargs),
            "improve_after_verify_solution": get_prompt(cfg.improve_after_verify_prompt_config, **prompt_kwargs),
        }
        # Parse time limit locally for this task only
        self._deadline_ts = None
        if cfg.time_limit:
            parts = cfg.time_limit.split(":")
            if len(parts) != 3:
                raise ValueError("time_limit must be in 'HH:MM:SS' format")
            h, m, s = map(int, parts)
            self._deadline_ts = time.time() + (h * 3600 + m * 60 + s)

        # Keep a sliding window of evaluated solutions with their scores and feedback
        # Each entry: {"solution": str, "score": float, "feedback": str} (order denotes recency)
        self.saved_solutions: list[dict] = []

    def log_example_prompt(self, data):
        pass

    async def _call_llm(self, data_point, all_data, prompt_key, **extra_data):
        combined_dp = {**data_point, **extra_data}
        filled_prompt = self.fill_prompt(combined_dp, all_data, prompt=self.prompts[prompt_key])
        start_t = time.time()
        try:
            llm_out = await super().process_single_datapoint(combined_dp, all_data, prompt=self.prompts[prompt_key])
        except (litellm.exceptions.OpenAIError, Exception) as e:
            print(f"LLM call failed: {e}\nPrompt causing failure:\n{filled_prompt}")
            raise
        gen_time = time.time() - start_t
        return filled_prompt, llm_out, gen_time

    async def _call_llm_with_code_retry(self, data_point, all_data, prompt_key, async_pos: int, **extra_data):
        """Call LLM and ensure a code block exists, retrying up to cfg.retry_solution times."""
        attempts = int(self.cfg.retry_solution)
        last = "```cpp\n # failed to create a solution```"
        for attempt in range(1, attempts + 1):
            filled_prompt, llm_out, gen_time = await self._call_llm(data_point, all_data, prompt_key, **extra_data)
            last = (filled_prompt, llm_out, gen_time)
            if extract_code_block(llm_out["generation"]) is not None:
                return filled_prompt, llm_out, gen_time
            if attempt < attempts:
                print(f"Retry {attempt}/{attempts} at async position {async_pos}.")
        print(f"Failed to create a solution after {attempts} attempts at async position {async_pos}.")
        # Return last even if no block; caller may decide how to handle
        return last

    async def process_single_datapoint(self, data_point, all_data, prompt=None):
        chat_history = []
        num_steps_completed = 0

        # Attempt to resume from latest intermediate state
        async_pos = data_point[self.cfg.async_position_key]
        saved_state = self.load_latest_state(async_pos)
        if saved_state and ("_intermediate" in saved_state) and saved_state["_intermediate"]:
            chat_history = saved_state["steps"]
            cur_generation_response = saved_state["generation"]
            num_steps_completed = int(saved_state["num_steps_completed"])
            self.saved_solutions = list(saved_state["saved_solutions"])
            print(f"[Resume] Restoring pos={async_pos} from step {num_steps_completed}")
        else:
            prompt_txt, solution_response, gen_time = await self._call_llm_with_code_retry(
                data_point, all_data, "initial", async_pos
            )
            cur_generation_response = solution_response["generation"]
            chat_history.append(
                {"prompt": prompt_txt, "response": cur_generation_response, "generation_time": gen_time}
            )

            print("[Initial] Generated initial solution.")
            # Save checkpoint after initial solution
            print(f"[Checkpoint] Saving intermediate pos={async_pos}, step={num_steps_completed}")
            await self.save_intermediate_state(
                async_pos,
                {
                    "generation": cur_generation_response,
                    "steps": chat_history,
                    "num_steps_completed": num_steps_completed,
                    "saved_solutions": self.saved_solutions,
                },
            )

        for step_num in range(num_steps_completed, self.cfg.total_steps):
            # Evaluate the current solution using the external evaluator.
            # Time the external evaluator to capture evaluation latency.
            eval_start_t = time.time()
            eval_results = await self.evaluator.eval_single({**data_point, "generation": cur_generation_response})
            eval_time = time.time() - eval_start_t

            # Record evaluation time for the solution that was just evaluated (last entry in chat_history).
            if chat_history:
                chat_history[-1]["evaluation_time"] = eval_time
            test_case_results = eval_results["test_case_results"]
            if chat_history:
                chat_history[-1]["subtask_scores"] = {k: v["score"] for k, v in test_case_results.items()}

            # Check if all subtasks passed fully (score == 1 for every output)
            if all(all(o["score"] == 1 for o in v["outputs"]) for v in test_case_results.values()):
                print(f"[Success] All test cases passed at step {step_num}.")
                return {
                    "generation": cur_generation_response,
                    "steps": chat_history,
                    "num_steps_completed": num_steps_completed,
                }

            print(f"[Step {step_num + 1}/{self.cfg.total_steps}] Improving based on evaluator feedback.")

            # Prepare a concise failure summary (only non-perfect cases)
            failure_lines = []
            for subtask, info in test_case_results.items():
                for out in info["outputs"]:
                    if out["score"] != 1:
                        failure_lines.append(
                            f"{subtask}:{out['test_name']} score={out['score']} msg={out['run_stderr'].strip()}"
                        )
            failure_summary = "\n".join(failure_lines)

            # Update saved solutions pool with current evaluated solution (score + feedback)
            if self.cfg.show_k_solutions and self.cfg.show_k_solutions > 0:
                subtask_key = data_point["subtask"]
                cur_score = float(test_case_results[subtask_key]["score"])
                cur_code_opt = extract_code_block(cur_generation_response)
                cur_code = cur_code_opt
                self._update_saved_solutions(cur_code, cur_score, failure_summary)

            # Build the 'solution' payload for the next prompt
            subtask_key = data_point["subtask"]
            cur_score_for_block = float(test_case_results[subtask_key]["score"])
            current_code_block = extract_code_block(cur_generation_response)

            if int(self.cfg.show_k_solutions) > 0:
                prev_text = self._build_previous_solutions_text(self.cfg.show_k_solutions)
                current_text = self._build_solution_block(current_code_block, failure_summary, cur_score_for_block)
                solution_payload = prev_text + ("\n\n" if prev_text else "") + current_text
            else:
                # K == 0: include only current solution with its feedback and score
                solution_payload = self._build_solution_block(current_code_block, failure_summary, cur_score_for_block)

            # Ask the LLM to improve the solution given the evaluator feedback.
            prompt_txt, improve_resp, gen_time = await self._call_llm_with_code_retry(
                data_point,
                all_data,
                "improve_after_verify_solution",
                async_pos=async_pos,
                solution=solution_payload,
            )

            num_steps_completed += 1
            cur_generation_response = improve_resp["generation"]

            chat_history.append(
                {"prompt": prompt_txt, "response": cur_generation_response, "generation_time": gen_time}
            )

            # Save checkpoint after each improvement step
            print(f"[Checkpoint] Saving intermediate pos={async_pos}, step={num_steps_completed}")
            await self.save_intermediate_state(
                async_pos,
                {
                    "generation": cur_generation_response,
                    "steps": chat_history,
                    "num_steps_completed": num_steps_completed,
                    "saved_solutions": self.saved_solutions,
                },
            )
            # Time limit check only inside the loop after checkpoint save
            if self._deadline_ts is not None and time.time() >= self._deadline_ts:
                print("[TimeLimit] Reached limit after step save; exiting cleanly.")
                sys.exit(0)

        # Reached maximum steps without passing all tests.
        print("[Failure] Reached max improvement steps without passing all tests.")
        return {
            "generation": cur_generation_response,
            "steps": chat_history,
            "num_steps_completed": num_steps_completed,
        }

    def _update_saved_solutions(self, solution: str, score: float, feedback: str) -> None:
        """Add current solution to sliding window with score-aware eviction.

        Keeps at most cfg.show_k_solutions previous solutions. Evicts lower-scored entries first;
        on ties or if none are lower, evicts the oldest by insertion order.
        """
        k = int(self.cfg.show_k_solutions or 0)
        if k <= 0:
            return

        entry = {"solution": solution, "score": float(score), "feedback": feedback}
        self.saved_solutions.append(entry)

        # Evict to maintain size k+1 (K previous + current)
        while len(self.saved_solutions) > k + 1:
            newest = self.saved_solutions[-1]
            # Candidates strictly worse than newest
            worse_indices = [i for i, e in enumerate(self.saved_solutions[:-1]) if e["score"] < newest["score"]]
            if worse_indices:
                # Among worse, pick with smallest score, then oldest by order
                min_score = min(self.saved_solutions[i]["score"] for i in worse_indices)
                candidates = [i for i in worse_indices if self.saved_solutions[i]["score"] == min_score]
                idx = min(candidates)  # oldest among worst by index
                del self.saved_solutions[idx]
            else:
                # No strictly worse; evict the oldest overall (index 0), not the newest
                idx = 0
                del self.saved_solutions[idx]

    def _build_previous_solutions_text(self, k: int) -> str:
        """Create a formatted text block listing up to k previous solutions.

        Excludes the most recent solution (current) when possible so it can be
        provided separately via the 'solution' field, avoiding duplication.
        """
        if not k or k <= 0:
            return ""
        if not self.saved_solutions:
            return ""

        # Exclude the latest/current solution from the listed previous ones
        prev = self.saved_solutions[:-1]
        if not prev:
            return ""

        # Take up to k most recent from prev, in reverse-chronological order
        tail = prev[-k:]
        tail = list(reversed(tail))

        lines = [self.PREVIOUS_PREAMBLE]
        for idx, e in enumerate(tail, start=1):
            lines.append("")
            # Use shared block template with numbered title suffix
            lines.append(
                self.SOLUTION_BLOCK_TEMPLATE.format(
                    title_suffix=f" {idx}",
                    code=e["solution"],
                    feedback=e["feedback"],
                    score=e["score"],
                )
            )

        return "\n".join(lines)

    def _build_solution_block(self, code: str, feedback: str, score: float, title_suffix: str = "") -> str:
        """Return a single solution block using shared template.

        title_suffix: e.g. " 1" to produce "## Solution 1 ##"; empty to produce "## Solution ##".
        """
        return self.SOLUTION_BLOCK_TEMPLATE.format(
            title_suffix=title_suffix,
            code=code,
            feedback=feedback,
            score=score,
        )


GENERATION_TASK_CLASS = IOIExecutionGenerationTask


@hydra.main(version_base=None, config_name="base_ioi_generation_config")
def ioi_generation(cfg: IOIExecutionConfig):
    cfg = IOIExecutionConfig(_init_nested=True, **cfg)
    LOG.info("Note: IOI Module is being used.")
    LOG.info("Config used: %s", cfg)
    task = IOIExecutionGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(IOIExecutionConfig, server_params=server_params())

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        ioi_generation()
