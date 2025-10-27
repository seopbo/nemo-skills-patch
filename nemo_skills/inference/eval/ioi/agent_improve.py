import asyncio
import logging
import os
import re
import sys
import tempfile
import time
from dataclasses import field
from typing import List, Optional

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


def extract_cpp_block(text: str) -> Optional[str]:
    text = text.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]
    matches = re.findall(r"```cpp(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_script_block(text: str) -> Optional[str]:
    text = text.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]
    matches = re.findall(r"```script(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


def build_question_with_history(question: str, prev_solutions: List[str]) -> str:
    if not prev_solutions:
        return question
    parts: List[str] = []
    for idx, sol in enumerate(prev_solutions, start=1):
        parts.append(f"### Proposed Solution {idx} ###\n{sol}\n")
    prev = "\n".join(parts)
    return f"{question}\n\n Here are previous proposed solutions from various agents that you can use for your reference:\n{prev}"


@nested_dataclass(kw_only=True)
class IOIExecutionConfig(GenerateSolutionsConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)
    # Prompts for execution-driven agent
    prompt_config: str = "eval/ioi/agent/solver"
    generate_test_prompt_config: str = "eval/ioi/multiagent/code/generate_test"
    improve_prompt_config: str = "eval/ioi/multiagent/code/improve"

    # Execution agent controls
    execution_steps: int = 100
    test_timeout_s: float = 30.0
    execution_max_output_characters: int = 1000
    # How many previous solutions to include in the question context (0 disables)
    n_prev_solutions: int = 0
    # Number of retries when failing to extract a solution from the LLM output
    n_retry_improve: int = 3
    time_limit: str | None = None  # Optional wall-clock time limit in 'HH:MM:SS'


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_ioi_generation_config", node=IOIExecutionConfig)


class IOIExecutionGenerationTask(GenerationTask):
    def __init__(self, cfg: IOIExecutionConfig):
        super().__init__(cfg)
        prompt_kwargs = {
            "examples_type": cfg.examples_type,
        }
        self.prompts = {
            "initial": get_prompt(cfg.prompt_config, **prompt_kwargs),
            "generate_test": get_prompt(cfg.generate_test_prompt_config, **prompt_kwargs),
            "improve": get_prompt(cfg.improve_prompt_config, **prompt_kwargs),
        }
        # Parse time limit locally for this task only
        self._deadline_ts = None
        if cfg.time_limit:
            parts = cfg.time_limit.split(":")
            if len(parts) != 3:
                raise ValueError("time_limit must be in 'HH:MM:SS' format")
            h, m, s = map(int, parts)
            self._deadline_ts = time.time() + (h * 3600 + m * 60 + s)

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

    async def _exec_generate_test(self, question: str, solution: str, all_data: dict):
        filled, out, t = await self._call_llm({"question": question, "solution": solution}, all_data, "generate_test")
        return filled, out, t

    async def _exec_improve(self, question: str, solution: str, script: str, output: str, all_data: dict):
        filled, out, t = await self._call_llm(
            {"question": question, "solution": solution, "script": script, "output": output}, all_data, "improve"
        )
        return filled, out, t

    async def _exec_compile_and_run_cpp(self, code_string: str, data_point: dict):
        with tempfile.TemporaryDirectory() as temp_dir:
            for original_path, content in data_point.get("grader_files", []):
                filename = os.path.basename(original_path)
                if "checker" in filename or "grader" in filename or not filename.endswith(".h"):
                    continue
                with open(os.path.join(temp_dir, filename), "w") as f:
                    f.write(content)

            executable_path = os.path.join(temp_dir, "a.out")
            compile_command = ["g++", "-I", temp_dir, "-x", "c++", "-o", executable_path, "-"]
            compiler_process = await asyncio.create_subprocess_exec(
                *compile_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, compile_stderr = await compiler_process.communicate(input=code_string.encode())

            if compiler_process.returncode != 0:
                return "", "", compile_stderr.decode()

            run_process = await asyncio.create_subprocess_exec(
                executable_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            run_stdout, run_stderr = await run_process.communicate()
            return run_stdout.decode(), run_stderr.decode(), ""

    async def process_single_datapoint(self, data_point, all_data, prompt=None):
        chat_history = []
        num_steps_completed = 0

        # Attempt to resume from latest intermediate state
        async_pos = data_point[self.cfg.async_position_key]
        saved_state = self.load_latest_state(async_pos)
        starting_solution_code: Optional[str] = None
        prev_solutions: List[str] = []
        if saved_state and saved_state.get("_intermediate", False):
            chat_history = saved_state["steps"]
            cur_generation_response = saved_state["generation"]
            starting_solution_code = extract_cpp_block(cur_generation_response)
            num_steps_completed = int(saved_state["num_steps_completed"])
            prev_solutions = list(saved_state["prev_solutions"])
            print(f"[Resume] Restoring pos={async_pos} from step {num_steps_completed}")
        else:
            prompt_txt, solution_response, gen_time = await self._call_llm(data_point, all_data, "initial")
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
                    "prev_solutions": prev_solutions,
                },
            )
            starting_solution_code = extract_cpp_block(cur_generation_response)
            prev_solutions.append(starting_solution_code)
            if self.cfg.n_prev_solutions > 0:
                prev_solutions = prev_solutions[-int(self.cfg.n_prev_solutions) :]

        # Inline execution-improve loop with checkpointing and resume
        current_solution: Optional[str] = starting_solution_code
        if not current_solution:
            base_question = data_point["question"]
            recent_prev = (
                prev_solutions[-int(self.cfg.n_prev_solutions) :] if int(self.cfg.n_prev_solutions) > 0 else []
            )
            question_for_solver = (
                build_question_with_history(base_question, recent_prev) if recent_prev else base_question
            )

            prompt_txt = None
            solution_response = None
            gen_time = None
            current_solution = None
            # Initial attempt + up to n_retry_improve retries
            for attempt in range(0, int(self.cfg.n_retry_improve) + 1):
                prompt_txt, solution_response, gen_time = await self._call_llm(
                    {**data_point, "question": question_for_solver}, all_data, "initial"
                )
                extracted = extract_cpp_block(solution_response["generation"]) or None
                if extracted:
                    current_solution = extracted
                    break
                if attempt < int(self.cfg.n_retry_improve):
                    print(
                        f"failed to extract solution, retrying... attempt {attempt + 1} out of {int(self.cfg.n_retry_improve)}"
                    )
                else:
                    raise ValueError("Initial solution extraction failed after retries")

            # Log only the successful attempt
            chat_history.append(
                {"prompt": prompt_txt, "response": solution_response["generation"], "generation_time": gen_time}
            )

            prev_solutions.append(current_solution)
            if self.cfg.n_prev_solutions > 0:
                prev_solutions = prev_solutions[-int(self.cfg.n_prev_solutions) :]

        start_step = num_steps_completed if (saved_state and saved_state.get("_intermediate", False)) else 0
        for step_idx in range(start_step, int(self.cfg.execution_steps)):
            base_question = data_point["question"]
            recent_prev = (
                prev_solutions[-int(self.cfg.n_prev_solutions) :] if int(self.cfg.n_prev_solutions) > 0 else []
            )
            question_for_step = (
                build_question_with_history(base_question, recent_prev) if recent_prev else base_question
            )
            await self.save_intermediate_state(
                async_pos,
                {
                    "generation": f"```cpp\n{current_solution}\n```",
                    "steps": chat_history,
                    "num_steps_completed": step_idx,
                    "prev_solutions": prev_solutions,
                },
            )
            if self._deadline_ts is not None and time.time() >= self._deadline_ts:
                print("[TimeLimit] Reached limit after step save; exiting cleanly.")
                sys.exit(0)

            # Per-step log container to group all artifacts for this iteration
            step_log = {"step_idx": step_idx}
            filled_test, out_test, t_test = await self._exec_generate_test(
                question_for_step, current_solution, all_data
            )
            step_log["test_prompt"] = filled_test
            step_log["test_response"] = out_test["generation"]
            step_log["test_generation_time"] = t_test
            script = extract_script_block(out_test["generation"]) or ""
            if not script:
                raise ValueError("Failed to extract test script from generate_test output")
            step_log["test_script"] = script

            run_stdout, run_stderr, compile_stderr = await self._exec_compile_and_run_cpp(script, data_point)
            compiled_ok = len(compile_stderr) == 0
            max_chars = max(0, int(self.cfg.execution_max_output_characters))
            if compiled_ok:
                if max_chars > 0:
                    if len(run_stdout) > max_chars:
                        run_stdout = run_stdout[:max_chars] + "<output cut>"
                    if len(run_stderr) > max_chars:
                        run_stderr = run_stderr[:max_chars] + "<output cut>"
                exec_output = f"STDOUT:\n{run_stdout}\nSTDERR:\n{run_stderr}"
            else:
                if max_chars > 0 and len(compile_stderr) > max_chars:
                    compile_stderr = compile_stderr[:max_chars] + "<output cut>"
                exec_output = f"COMPILATION_ERROR:\n{compile_stderr}"
            step_log["run_stdout"] = run_stdout
            step_log["run_stderr"] = run_stderr
            step_log["compile_stderr"] = compile_stderr

            # Improve with retry on extraction failure
            new_solution = None
            for attempt in range(0, int(self.cfg.n_retry_improve) + 1):
                filled_imp, out_imp, t_imp = await self._exec_improve(
                    question_for_step, current_solution, script, exec_output, all_data
                )
                step_log["improve_prompt"] = filled_imp
                step_log["improve_response"] = out_imp["generation"]
                step_log["improve_generation_time"] = t_imp
                extracted = extract_cpp_block(out_imp["generation"]) or None
                if extracted:
                    new_solution = extracted
                    break
                if attempt < int(self.cfg.n_retry_improve):
                    print(
                        f"failed to extract solution, retrying... attempt {attempt + 1} out of {int(self.cfg.n_retry_improve)}"
                    )
                else:
                    raise ValueError("Failed to extract improved C++ solution after retries")
            current_solution = new_solution
            # Maintain rolling history of previous solutions
            prev_solutions.append(current_solution)
            if self.cfg.n_prev_solutions > 0:
                prev_solutions = prev_solutions[-int(self.cfg.n_prev_solutions) :]

            # Evaluate the newly produced solution, store timing and scores on the last log entry
            step_generation_wrapped = f"```cpp\n{current_solution}\n```"
            eval_start_t = time.time()
            eval_results = await self.evaluator.eval_single({**data_point, "generation": step_generation_wrapped})
            eval_time = time.time() - eval_start_t
            test_case_results = eval_results.get("test_case_results", {})
            step_log["evaluation_time"] = eval_time
            step_log["subtask_scores"] = {k: v["score"] for k, v in test_case_results.items()}

            # Append all per-step data in a single entry
            chat_history.append(step_log)

            # Print success message if all test cases passed
            if all(all(o["score"] == 1 for o in v["outputs"]) for v in test_case_results.values()):
                print(f"[Success] All test cases passed at step {step_idx + 1}.")

            print(f"[IOIExecution] Position {async_pos} Completed {step_idx + 1} steps.")

        final_solution_code = current_solution
        completed_steps = int(self.cfg.execution_steps)

        # Wrap final code for evaluator compatibility
        final_generation_response = f"```cpp\n{final_solution_code}\n```"

        # Evaluate final solution and record timing and scores
        eval_start_t = time.time()
        eval_results = await self.evaluator.eval_single({**data_point, "generation": final_generation_response})
        eval_time = time.time() - eval_start_t
        if chat_history:
            chat_history[-1]["evaluation_time"] = eval_time
            test_case_results = eval_results.get("test_case_results", {})
            chat_history[-1]["subtask_scores"] = {k: v["score"] for k, v in test_case_results.items()}

        # Update steps completed to the loop's final completed step count
        num_steps_completed = completed_steps

        return {
            "generation": final_generation_response,
            "steps": chat_history,
            "num_steps_completed": num_steps_completed,
        }


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
