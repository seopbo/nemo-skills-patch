import asyncio
import logging
import random
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


def extract_failure_summary(text: str):
    text = text.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]
    matches = re.findall(r"```report(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_verdict(text: str):
    """Extract a single memory update verdict from a fenced verdict block.

    The model should output a single boxed verdict in a block like:

    ```verdict
    no
    ```
    or
    ```verdict
    3
    ```
    """
    text = text.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]
    matches = re.findall(r"```verdict(.*?)```", text, re.DOTALL)
    return matches[-1].strip().lower() if matches else None


def extract_verdict_list(text: str):
    """Extract a list of integer indices from a fenced verdict block.

    Example:
    ```verdict
    1, 3, 5
    ```
    """
    text = text.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]
    matches = re.findall(r"```verdict(.*?)```", text, re.DOTALL)
    if not matches:
        return None
    content = matches[-1]
    ints = re.findall(r"-?\d+", content)
    try:
        return [int(x) for x in ints] if ints else None
    except Exception:
        return None


def extract_steps(text: str):
    """Extract integer step count from a fenced steps block:

    ```steps
    17
    ```
    """
    text = text.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]
    matches = re.findall(r"```steps(.*?)```", text, re.DOTALL)
    if not matches:
        return None
    content = matches[-1].strip()
    m = re.search(r"-?\d+", content)
    try:
        return int(m.group(0)) if m else None
    except Exception:
        return None


@nested_dataclass(kw_only=True)
class IOIExecutionConfig(GenerateSolutionsConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)
    prompt_config: str = "eval/ioi/agent/solver"
    improve_after_verify_prompt_config: str = "eval/ioi/agent/improve_with_private_test"
    self_improve_prompt_config: str = "eval/ioi/agent/self_correct/self_improve"
    final_select_prompt_config: str = "eval/ioi/agent/self_correct/final_select"
    memory_batch_select_prompt_config: str = "eval/ioi/agent/self_correct/memory_batch_select"
    memory_verdict_prompt_config: str = "eval/ioi/agent/self_correct/memory_verdict"
    model_choose_steps_prompt_config: str = "eval/ioi/agent/self_correct/choose_steps"
    total_steps: int = 60
    time_limit: str | None = None  # Optional wall-clock time limit in 'HH:MM:SS'
    retry_solution: int = 5  # Retry count when code block extraction fails
    max_char_for_stored_solution: int = 20000
    model_choose_steps: bool = False
    average_model_choose_steps: int = 1
    sample_tests_attempts: int = 0
    per_step_evaluate: bool = True
    # If > 0, enables mixture-of-agents style: generate X solutions in parallel each step
    num_generated_solutions: int = 0


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_ioi_generation_config", node=IOIExecutionConfig)


class IOIExecutionGenerationTask(GenerationTask):
    # Shared formats for composing solution blocks
    SOLUTION_BLOCK_TEMPLATE = "## Solution{title_suffix} ##\n{code}"

    def __init__(self, cfg: IOIExecutionConfig):
        super().__init__(cfg)
        prompt_kwargs = {
            "examples_type": cfg.examples_type,
        }
        self.prompts = {
            "initial": get_prompt(cfg.prompt_config, **prompt_kwargs),
            "self_improve": get_prompt(cfg.self_improve_prompt_config, **prompt_kwargs),
            "final_select": get_prompt(cfg.final_select_prompt_config, **prompt_kwargs),
            "memory_batch_select": get_prompt(cfg.memory_batch_select_prompt_config, **prompt_kwargs),
            "failure_summary": get_prompt("eval/ioi/agent/failure_summary", **prompt_kwargs),
            "memory_verdict": get_prompt(cfg.memory_verdict_prompt_config, **prompt_kwargs),
            "choose_steps": get_prompt(cfg.model_choose_steps_prompt_config, **prompt_kwargs),
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

    async def _decide_steps_with_retry(self, data_point, all_data, async_pos: int) -> int | None:
        """Ask the model to choose number of improvement steps. Retries on parse failure."""
        attempts = int(self.cfg.retry_solution)
        chosen = None
        for attempt in range(1, attempts + 1):
            _, llm_out, _ = await self._call_llm(
                data_point,
                all_data,
                "choose_steps",
                total_steps=int(self.cfg.total_steps),
            )
            chosen = extract_steps(llm_out["generation"])
            if isinstance(chosen, int):
                return chosen
            if attempt < attempts:
                print(f"Retry choose-steps {attempt}/{attempts} at async position {async_pos}.")
        return chosen

    async def _decide_steps_average_parallel(
        self, data_point, all_data, async_pos: int, num_samples: int
    ) -> int | None:
        """Run multiple choose-steps queries in parallel and return the rounded average."""
        n = max(1, int(num_samples or 1))
        tasks = [self._decide_steps_with_retry(data_point, all_data, async_pos) for _ in range(n)]
        results = await asyncio.gather(*tasks)
        ints = [r for r in results if isinstance(r, int)]
        if not ints:
            return None
        return int(round(sum(ints) / len(ints)))

    async def process_single_datapoint(self, data_point, all_data, prompt=None):
        """Mixture-of-agents style: maintain a pool of X solutions per step."""
        chat_history = []
        num_steps_completed = 0

        # ICPC does not have a subtask score, we add it manually (max score is 1)
        if data_point.get("subtask_score") is None:
            data_point["subtask_score"] = "1"

        async_pos = data_point[self.cfg.async_position_key]
        X = max(1, int(getattr(self.cfg, "num_generated_solutions", 0) or 0))
        decided_total_steps = int(self.cfg.total_steps)
        pool_codes: list[str] = []

        # Attempt to resume
        saved_state = self.load_latest_state(async_pos)
        if saved_state and ("_intermediate" in saved_state) and saved_state["_intermediate"]:
            chat_history = saved_state["steps"]
            num_steps_completed = int(saved_state["num_steps_completed"])
            self.saved_solutions = list(saved_state["saved_solutions"])
            pool_codes = list(saved_state.get("pool_solutions", []))
            if getattr(self.cfg, "model_choose_steps", False):
                decided_total_steps = int(saved_state.get("decided_total_steps", decided_total_steps))
            print(
                f"[Resume] (Pool) Restoring pos={async_pos} from step {num_steps_completed} with {len(pool_codes)} solutions"
            )
        else:
            # Initial parallel generation of X solutions
            tasks = [self._call_llm_with_code_retry(data_point, all_data, "initial", async_pos) for _ in range(X)]
            results = await asyncio.gather(*tasks)
            total_tokens = 0
            max_time = 0.0
            pool_codes = []
            for filled_prompt, llm_out, gen_time in results:
                code = extract_code_block(llm_out["generation"])
                if code:
                    pool_codes.append(code)
                total_tokens += int(llm_out.get("num_generated_tokens", 0) or 0)
                if gen_time > max_time:
                    max_time = gen_time
            chat_history.append({"num_generated_tokens": total_tokens, "generation_time": max_time})
            print(f"[Initial Pool] Generated {len(pool_codes)}/{X} solutions.")

            # Decide number of improvement steps (optional)
            if getattr(self.cfg, "model_choose_steps", False):
                num_samples = max(1, int(getattr(self.cfg, "average_model_choose_steps", 1) or 1))
                if num_samples == 1:
                    decided = await self._decide_steps_with_retry(data_point, all_data, async_pos)
                else:
                    decided = await self._decide_steps_average_parallel(
                        data_point, all_data, async_pos, num_samples=num_samples
                    )
                if isinstance(decided, int):
                    decided_total_steps = max(1, min(int(self.cfg.total_steps), decided))
                else:
                    decided_total_steps = int(self.cfg.total_steps)
                print(
                    f"Async Pos : {async_pos} Step : {num_steps_completed} Problem {data_point['id']}: Decided improvement steps = {decided_total_steps} (max {int(self.cfg.total_steps)})"
                )

            # Save checkpoint after initial pool
            print(f"[Checkpoint] (Pool) Saving intermediate pos={async_pos}, step={num_steps_completed}")
            await self.save_intermediate_state(
                async_pos,
                {
                    "id": data_point["id"],
                    "steps": chat_history,
                    "num_steps_completed": num_steps_completed,
                    "saved_solutions": self.saved_solutions,
                    "decided_total_steps": decided_total_steps,
                    "pool_solutions": pool_codes,
                    "num_generated_solutions": X,
                    "_intermediate": True,
                },
            )

        # Iterate improvements
        last_sample_scores: list[float] = [0.0 for _ in pool_codes]
        for step_num in range(num_steps_completed, decided_total_steps):
            # Optional per-step evaluation for the whole pool
            if self.cfg.per_step_evaluate and pool_codes:
                eval_start_t = time.time()
                last_sample_scores = await self._evaluate_codes_sample_scores(pool_codes, data_point)
                eval_time = time.time() - eval_start_t
                print(
                    f"Async Pos : {async_pos} Step : {step_num} Problem {data_point['id']}: Pool evaluation time = {eval_time:.3f}s"
                )
                if chat_history:
                    chat_history[-1]["evaluation_time"] = eval_time
                    chat_history[-1]["sample_scores"] = list(last_sample_scores)
                # Batch update memory: choose up to X from combined memory + pool in one go
                await self._maybe_update_memory_batch(
                    pool_codes=pool_codes,
                    pool_scores=[float(s or 0.0) for s in last_sample_scores],
                    step_index=step_num,
                    chat_history_ref=chat_history,
                    async_pos=async_pos,
                    problem_id=data_point["id"],
                    k_override=X,
                )

            # Early termination: optional attempts limit, require at least one perfect candidate
            attempts_limit = int(getattr(self.cfg, "sample_tests_attempts", 0) or 0)
            if (
                self.cfg.per_step_evaluate
                and attempts_limit > 0
                and step_num >= attempts_limit
                and (not last_sample_scores or max(last_sample_scores) < 1.0)
            ):
                print(
                    f"[Terminate] (Pool) Problem {data_point['id']}: No perfect sample after {attempts_limit} steps; stopping."
                )
                break

            print(f"[Step {step_num + 1}/{self.cfg.total_steps}] (Pool) Self-improving {len(pool_codes)} solutions.")
            pool_text = self._build_solution_pool_text(pool_codes)

            # Generate next pool in parallel (each sees the entire pool)
            tasks = [
                self._call_llm_with_code_retry(
                    data_point,
                    all_data,
                    "self_improve",
                    async_pos=async_pos,
                    solution=pool_text,
                )
                for _ in range(X)
            ]
            results = await asyncio.gather(*tasks)
            total_tokens = 0
            max_time = 0.0
            next_pool_codes: list[str] = []
            for prompt_txt, llm_out, gen_time in results:
                code = extract_code_block(llm_out["generation"])
                if code:
                    next_pool_codes.append(code)
                total_tokens += int(llm_out.get("num_generated_tokens", 0) or 0)
                if gen_time > max_time:
                    max_time = gen_time

            num_steps_completed += 1
            chat_history.append(
                {
                    "prompt": results[0][0] if results else "",
                    "responses_generated": len(next_pool_codes),
                    "generation_time": max_time,
                    "num_generated_tokens": total_tokens,
                }
            )

            # Save checkpoint
            print(f"[Checkpoint] (Pool) Saving intermediate pos={async_pos}, step={num_steps_completed}")
            await self.save_intermediate_state(
                async_pos,
                {
                    "id": data_point["id"],
                    "steps": chat_history,
                    "num_steps_completed": num_steps_completed,
                    "saved_solutions": self.saved_solutions,
                    "decided_total_steps": decided_total_steps,
                    "pool_solutions": next_pool_codes,
                    "num_generated_solutions": X,
                    "_intermediate": True,
                },
            )
            # Time limit check only inside the loop after checkpoint save
            if self._deadline_ts is not None and time.time() >= self._deadline_ts:
                print("[TimeLimit] Reached limit after step save; exiting cleanly.")
                sys.exit(0)

            pool_codes = next_pool_codes

        # Final: choose one solution from the pool
        selected_idx, selected_code = await self._select_final_solution_from_pool(
            pool_codes, last_sample_scores, async_pos, data_point["id"]
        )

        # Final evaluation of memory solutions (if any)
        memory_solutions_results = []
        if self.cfg.per_step_evaluate and self.saved_solutions:
            details_by_idx = await self._evaluate_memory_solutions(data_point, return_details=True)
            for idx, info in details_by_idx.items():
                entry = self.saved_solutions[idx]
                memory_solutions_results.append(
                    {
                        "solution": entry.get("solution") if isinstance(entry, dict) else entry,
                        "step": entry.get("step") if isinstance(entry, dict) else None,
                        "sample_score": info.get("sample_score"),
                        "test_case_results": info.get("test_case_results"),
                    }
                )

        # Return selected solution as final generation text
        final_generation_text = f"```cpp\n{selected_code}\n```" if selected_code else ""
        return {
            "id": data_point["id"],
            "generation": final_generation_text,
            "steps": chat_history,
            "num_steps_completed": num_steps_completed,
            "memory_solutions": memory_solutions_results,
            "num_generated_tokens": sum(step.get("num_generated_tokens", 0) for step in chat_history),
            "selected_index": selected_idx + 1 if selected_idx is not None else None,
        }

    def _normalize_test_case_results(self, test_case_results: dict) -> dict:
        """Normalize evaluator outputs to a common shape:
        {subtask: {"score": float, "outputs": list}}.

        Supports both IOI-style per-subtask dicts and ICPC-style flat dict with
        keys {"score": bool|float, "outputs": list} by mapping the latter to
        a single "overall" subtask.
        """
        # ICPC-style: flat dict with outputs list
        if (
            isinstance(test_case_results, dict)
            and "outputs" in test_case_results
            and "score" in test_case_results
            and isinstance(test_case_results.get("outputs"), list)
        ):
            # ICPC-style: preserve fractional score (proportion of tests passed)
            return {"overall": {"score": float(test_case_results["score"]), "outputs": test_case_results["outputs"]}}

        # IOI-style: dict of subtasks
        return {
            k: {"score": float(v.get("score", 0.0)), "outputs": list(v.get("outputs", []))}
            for k, v in test_case_results.items()
        }

    async def _evaluate_codes_sample_scores(self, codes: list[str], data_point: dict) -> list[float]:
        """Evaluate a list of code strings and return per-code sample pass averages."""
        if not codes:
            return []
        payloads = [{**data_point, "generation": f"```cpp\n{code}\n```"} for code in codes]
        tasks = [self.evaluator.eval_single(payload) for payload in payloads]
        eval_results_list = await asyncio.gather(*tasks)

        def _avg(outputs_list):
            if not outputs_list:
                return 0.0
            try:
                total = len(outputs_list)
                passed = sum(1.0 if float(o.get("score", 0.0)) == 1.0 else 0.0 for o in outputs_list)
                return float(passed / total)
            except Exception:
                return 0.0

        scores: list[float] = []
        for res in eval_results_list:
            tcr = res.get("test_case_results", {})
            outputs_all = tcr.get("outputs", [])
            sample_outputs = [o for o in outputs_all if o.get("test_type") == "sample"]
            scores.append(_avg(sample_outputs))
        return scores

    async def _maybe_update_memory_batch(
        self,
        pool_codes: list[str],
        pool_scores: list[float],
        step_index: int,
        chat_history_ref: list,
        async_pos: int,
        problem_id,
        k_override: int | None = None,
    ) -> None:
        """Batch memory update: show combined memory + pool and select up to K to keep."""
        k = int((k_override if k_override is not None else 0) or 0)
        if k <= 0:
            return
        mem_entries = list(self.saved_solutions)
        mem_len = len(mem_entries)
        total_candidates = mem_len + len(pool_codes)
        if total_candidates == 0:
            return

        # Build combined candidate list text (stable order: memory first, then pool)
        lines: list[str] = [
            "You are curating a small memory of promising solutions.",
            f"Select up to K = {k} total solutions from the combined list below (memory first, then new pool).",
            "Each item shows its sample test pass rate if available.",
        ]
        display_idx = 1
        for e in mem_entries:
            score = float(e.get("sample_score", 0.0) or 0.0)
            lines.append("")
            lines.append(f"[#{display_idx}] source=memory sample_score={score:.3f}")
            lines.append(self._build_solution_block(e.get("solution", ""), title_suffix=f" #{display_idx}"))
            display_idx += 1
        for i, code in enumerate(pool_codes, start=1):
            score = float(pool_scores[i - 1] if (i - 1) < len(pool_scores) else 0.0)
            lines.append("")
            lines.append(f"[#{display_idx}] source=pool sample_score={score:.3f}")
            lines.append(self._build_solution_block(code, title_suffix=f" #{display_idx}"))
            display_idx += 1
        combined_text = "\n".join(lines)

        # Ask the model to select up to k indices to keep
        _, llm_out, _ = await self._call_llm(
            data_point={},
            all_data={},
            prompt_key="memory_batch_select",
            combined_candidates=combined_text,
            k=k,
        )
        indices = extract_verdict_list(llm_out["generation"]) or []

        # Normalize indices: unique, in-range, up to k
        selected: list[int] = []
        for idx in indices:
            if not isinstance(idx, int):
                continue
            if 1 <= idx <= total_candidates and idx not in selected:
                selected.append(idx)
            if len(selected) >= k:
                break

        # Fallback: pick top-k by sample score if selection invalid/empty
        if not selected:
            combined: list[tuple[float, int]] = []
            for j, e in enumerate(mem_entries, start=1):
                combined.append((float(e.get("sample_score", 0.0) or 0.0), j))
            for offset, score in enumerate(pool_scores, start=1):
                combined.append((float(score or 0.0), mem_len + offset))
            combined.sort(key=lambda t: t[0], reverse=True)
            selected = [idx for _, idx in combined[:k]]

        # Rebuild saved_solutions according to selection
        new_saved: list[dict] = []
        for idx in selected:
            if idx <= mem_len:
                # Keep from existing memory
                e = mem_entries[idx - 1]
                code = e.get("solution")
                if code:
                    new_saved.append(
                        {
                            "solution": code,
                            "step": int(e.get("step", step_index)),
                            "sample_score": float(e.get("sample_score", 0.0) or 0.0),
                        }
                    )
            else:
                # Take from current pool
                pool_idx = idx - mem_len - 1
                if 0 <= pool_idx < len(pool_codes):
                    code = pool_codes[pool_idx]
                    score = float(pool_scores[pool_idx] if pool_idx < len(pool_scores) else 0.0)
                    if code:
                        new_saved.append(
                            {
                                "solution": code,
                                "step": int(step_index),
                                "sample_score": float(score),
                            }
                        )
            if len(new_saved) >= k:
                break

        self.saved_solutions = new_saved
        if chat_history_ref:
            chat_history_ref[-1]["memory_candidates"] = [e.get("solution") for e in self.saved_solutions]

    def _build_solution_pool_text(self, codes: list[str]) -> str:
        """Compose a labeled list of solution blocks for the current pool."""
        if not codes:
            return ""
        lines: list[str] = []
        for idx, code in enumerate(codes, start=1):
            lines.append(self.SOLUTION_BLOCK_TEMPLATE.format(title_suffix=f" {idx}", code=code))
            if idx != len(codes):
                lines.append("")
        return "\n".join(lines)

    async def _select_final_solution_from_pool(
        self,
        pool_codes: list[str],
        sample_scores: list[float],
        async_pos: int,
        problem_id,
    ) -> tuple[int | None, str | None]:
        """Ask the model to pick one solution from the current pool using a dedicated prompt; fallback to best score."""
        if not pool_codes:
            return None, None
        # Build pool text with sample scores
        lines: list[str] = []
        for i, code in enumerate(pool_codes, start=1):
            score = 0.0
            if sample_scores and len(sample_scores) >= i:
                score = float(sample_scores[i - 1] or 0.0)
            lines.append(f"[#{i}] sample_score={score:.3f}")
            lines.append(self._build_solution_block(code, title_suffix=f" #{i}"))
            if i != len(pool_codes):
                lines.append("")
        pool_text = "\n".join(lines)

        # Use dedicated final selection prompt; expect an index in a ```verdict``` block
        _, llm_out, _ = await self._call_llm(
            data_point={},
            all_data={},
            prompt_key="final_select",
            pool_solutions=pool_text,
            k=len(pool_codes),
        )
        verdict = extract_verdict(llm_out["generation"])
        selected_idx = None
        if verdict and verdict != "no":
            try:
                display_idx = int(verdict)
                if 1 <= display_idx <= len(pool_codes):
                    selected_idx = display_idx - 1
            except Exception:
                selected_idx = None
        # Fallback: highest sample score or first
        if selected_idx is None:
            if sample_scores:
                try:
                    selected_idx = int(max(range(len(sample_scores)), key=lambda i: float(sample_scores[i] or 0.0)))
                except Exception:
                    selected_idx = 0
            else:
                selected_idx = 0

        code = pool_codes[selected_idx] if (selected_idx is not None and 0 <= selected_idx < len(pool_codes)) else None
        print(
            f"Async Pos : {async_pos} Problem {problem_id}: Selected pool index {selected_idx + 1 if selected_idx is not None else 'n/a'} as final"
        )
        return selected_idx, code

    async def _evaluate_memory_solutions(self, data_point: dict, return_details: bool = False) -> dict:
        """Evaluate all saved memory solutions.

        When return_details is False (default), returns:
            {memory_idx: {subtask: score}}
        When return_details is True, returns:
            {
              memory_idx: {
                "subtask_scores": {subtask: score},
                "test_case_results": <raw evaluator results>,
                "sample_score": <float>
              }
            }
        """
        if not self.saved_solutions:
            return {}

        idx_payloads = []
        for idx, entry in enumerate(self.saved_solutions):
            code = entry.get("solution") if isinstance(entry, dict) else entry
            if not code:
                continue
            idx_payloads.append((idx, {**data_point, "generation": f"```cpp\n{code}\n```"}))

        if not idx_payloads:
            return {}

        eval_tasks = [self.evaluator.eval_single(payload) for _, payload in idx_payloads]
        eval_results_list = await asyncio.gather(*eval_tasks)

        if return_details:
            details_by_idx = {}
            for (idx, _), res in zip(idx_payloads, eval_results_list):
                tcr = res.get("test_case_results", {})
                normalized = self._normalize_test_case_results(tcr)
                # Compute sample average if available
                outputs_all = tcr.get("outputs", [])
                sample_outputs = [o for o in outputs_all if o.get("test_type") == "sample"]

                def _avg(outputs_list):
                    if not outputs_list:
                        return 0.0
                    try:
                        total = len(outputs_list)
                        passed = sum(1.0 if float(o.get("score", 0.0)) == 1.0 else 0.0 for o in outputs_list)
                        return float(passed / total)
                    except Exception:
                        return 0.0

                sample_avg = _avg(sample_outputs)
                details_by_idx[idx] = {
                    "subtask_scores": {k: v["score"] for k, v in normalized.items()},
                    "test_case_results": tcr,
                    "sample_score": sample_avg,
                }
            return details_by_idx
        else:
            scores_by_idx = {}
            for (idx, _), res in zip(idx_payloads, eval_results_list):
                tcr = res.get("test_case_results", {})
                normalized = self._normalize_test_case_results(tcr)
                scores_by_idx[idx] = {k: v["score"] for k, v in normalized.items()}
            return scores_by_idx

    async def _maybe_update_memory(
        self,
        code: str,
        step_index: int,
        sample_score: float,
        chat_history_ref: list,
        async_pos: int,
        problem_id,
        k_override: int | None = None,
    ) -> None:
        """Populate and manage memory of candidate solutions.

        - Initially fill up to K solutions without verdicts.
        - After K filled, ask LLM for a verdict whether to replace an existing one or skip.
        - Only consider code shorter than max_char_for_stored_solution.
        - Store step index and sample score alongside code.
        - Snapshot current candidate solutions (just code strings) into chat history entry.
        """
        k = int((k_override if k_override is not None else 0) or 0)
        if k <= 0 or not code:
            return

        # Enforce character limit
        if int(self.cfg.max_char_for_stored_solution) > 0 and len(code) > int(self.cfg.max_char_for_stored_solution):
            # Snapshot existing candidates for visibility
            if chat_history_ref:
                chat_history_ref[-1]["memory_candidates"] = [e.get("solution") for e in self.saved_solutions]
            return

        # If memory not yet full, just append
        if len(self.saved_solutions) < k:
            self.saved_solutions.append(
                {
                    "solution": code,
                    "step": int(step_index),
                    "sample_score": float(sample_score),
                }
            )
            if chat_history_ref:
                chat_history_ref[-1]["memory_candidates"] = [e.get("solution") for e in self.saved_solutions]
            return

        # Already have K solutions; ask for verdict whether to replace
        memory_text_lines = [
            "You are a competitive programmer curating a small memory of promising, diverse solutions.",
            "Sample tests are basic compared to the private suite, but solutions must pass samples.",
            f"Memory capacity K = {k}.",
            "Here are the current memory candidates with their sample test scores:",
        ]
        # Randomize display order to minimize positional bias
        shuffled_indices = list(range(len(self.saved_solutions)))
        random.shuffle(shuffled_indices)
        display_to_actual_index = {
            display_idx + 1: actual_idx for display_idx, actual_idx in enumerate(shuffled_indices)
        }
        for display_idx, actual_idx in enumerate(shuffled_indices, start=1):
            e = self.saved_solutions[actual_idx]
            memory_text_lines.append("")
            memory_text_lines.append(f"[#{display_idx}] sample_score={e.get('sample_score', 0.0):.3f}")
            memory_text_lines.append(
                self._build_solution_block(e.get("solution", ""), title_suffix=f" #{display_idx}")
            )
        memory_text = "\n".join(memory_text_lines)

        # Compose current solution block
        current_text = self._build_solution_block(code, title_suffix=" (current)")

        # Ask LLM for verdict (no retries needed)
        _, llm_out, _ = await self._call_llm(
            data_point={},
            all_data={},
            prompt_key="memory_verdict",
            memory_list=memory_text,
            current_solution=current_text,
            current_sample_score=f"{float(sample_score):.3f}",
            k=k,
        )
        verdict = extract_verdict(llm_out["generation"])
        if verdict and verdict != "no":
            try:
                display_idx = int(verdict)
                if 1 <= display_idx <= len(self.saved_solutions):
                    actual_idx = display_to_actual_index.get(display_idx)
                    self.saved_solutions[actual_idx] = {
                        "solution": code,
                        "step": int(step_index),
                        "sample_score": float(sample_score),
                    }
                    print(
                        f"Async Pos : {async_pos} Step : {step_index} Problem {problem_id}: Overwritten display position {display_idx} (actual {actual_idx + 1}) in memory"
                    )
                else:
                    print(
                        f"Async Pos : {async_pos} Step : {step_index} Problem {problem_id}: Failed to generate a verdict for memory decision."
                    )
            except Exception:
                print(
                    f"Async Pos : {async_pos} Step : {step_index} Problem {problem_id}: Failed to generate a verdict for memory decision."
                )
        elif verdict is None:
            print(
                f"Async Pos : {async_pos} Step : {step_index} Problem {problem_id}: Failed to generate a verdict for memory decision."
            )

        # Snapshot current candidates
        if chat_history_ref:
            chat_history_ref[-1]["memory_candidates"] = [e.get("solution") for e in self.saved_solutions]

    def _compute_subtask_score(self, subtask_data: dict) -> float:
        """Return fractional score when per-test outputs are available, otherwise fallback to provided score.

        For ICPC-style results, each output contains a per-test score (0/1). We compute the
        proportion of passed tests. For IOI-style, we respect the provided subtask score.
        """
        outputs = subtask_data.get("outputs")
        if isinstance(outputs, list) and outputs:
            try:
                total = len(outputs)
                passed = sum(1.0 if float(o.get("score", 0.0)) == 1.0 else 0.0 for o in outputs)
                return float(passed / total)
            except Exception:
                # Fallback to provided score on any unexpected shape
                pass
        return float(subtask_data.get("score", 0.0))

    def _build_solution_block(self, code: str, title_suffix: str = "") -> str:
        """Return a single solution block using shared template."""
        return self.SOLUTION_BLOCK_TEMPLATE.format(
            title_suffix=title_suffix,
            code=code,
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
