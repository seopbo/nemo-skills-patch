import json
import logging
import sys
from dataclasses import field

import hydra

from nemo_skills.inference.eval.bfcl import ClientMessageParser, ServerMessageParser
from nemo_skills.inference.eval.bfcl_utils import MAXIMUM_STEP_LIMIT
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.inference.model.utils import is_context_window_exceeded_error
from nemo_skills.prompt.utils import get_prompt, get_token_count
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class AgentToolsConfig(GenerateSolutionsConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)
    prompt_config: str = "eval/ioi/agent/agent_tools_solver"
    use_client_parsing: bool = True
    model_name: str | None = None
    max_steps: int = 30


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_agent_tools_generation_config", node=AgentToolsConfig)


class AgentToolsGenerationTask(GenerationTask):
    def __init__(self, cfg: AgentToolsConfig):
        super().__init__(cfg)
        self.prompt = get_prompt(cfg.prompt_config, examples_type=cfg.examples_type)
        self.message_parser = ClientMessageParser(cfg) if cfg.use_client_parsing else ServerMessageParser(cfg)

    def log_example_prompt(self, data):
        return

    async def _generate_single_assistant_turn(self, inference_state_dict):
        messages = inference_state_dict["messages"]
        tools = inference_state_dict["tools"]
        if self.cfg.system_message:
            messages = [{"role": "system", "content": self.cfg.system_message}] + messages
        input_dict = self.message_parser.construct_input_dict(messages, tools)
        return_dict = {}
        if self.cfg.count_prompt_tokens:
            num_input_tokens = get_token_count(
                self.hf_tokenizer, messages=input_dict["prompt"], tools=input_dict.get("tools", None)
            )
            return_dict["num_input_tokens"] = num_input_tokens
        try:
            output = await self.generate_with_semaphore(**input_dict)
        except Exception as error:
            if is_context_window_exceeded_error(error):
                LOG.warning(f"AgentTools generation failed due to running out of context. {error}")
                return_dict.update({"message": None, "generation": ""})
                return return_dict
            else:
                raise error
        parsed_response = self.message_parser.parse_output_dict(output)
        return_dict.update(parsed_response)
        return return_dict

    def _parse_reasoning_from_message_content(self, model_response_text: str | None):
        if model_response_text is None:
            return None
        if self.cfg.end_reasoning_string in model_response_text:
            return model_response_text.split(self.cfg.end_reasoning_string)[-1].lstrip("\n")
        return ""

    def _build_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "submit_solution",
                    "description": "Compile and run the given C++ code. Set sample=true to run only sample tests.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "C++17 source code to submit"},
                            "sample": {"type": "boolean", "description": "Run only sample tests", "default": False},
                        },
                        "required": ["code"],
                    },
                },
            }
        ]

    async def process_single_datapoint(self, data_point, all_data, prompt=None):
        user_content = self.fill_prompt(data_point, all_data, prompt=self.prompt)
        tools = self._build_tools()
        state_dict = {"messages": [{"role": "user", "content": user_content}], "tools": tools}

        num_generated_tokens_list = []
        num_input_tokens_list = []
        out_of_context = False
        final_solution = ""
        step_count = 0

        while True:
            model_response = await self._generate_single_assistant_turn(state_dict)
            if model_response["message"] is None:
                out_of_context = True
                LOG.info("Quitting generation due to running out of context.")
                break

            num_generated_tokens_list.append(model_response.get("num_generated_tokens", 0))
            if self.cfg.count_prompt_tokens:
                num_input_tokens_list.append(model_response.get("num_input_tokens", 0))

            if self.cfg.parse_reasoning:
                trimmed = self._parse_reasoning_from_message_content(
                    self.message_parser.get_response_text(model_response["message"])
                )
                self.message_parser.set_response_text(model_response["message"], trimmed)

            state_dict["messages"].append(model_response["message"])
            final_solution = self.message_parser.get_response_text(model_response["message"]) or final_solution

            tool_calls = model_response.get("generation", [])
            tool_call_ids = model_response.get("tool_call_ids", [])
            if not isinstance(tool_calls, list) or len(tool_calls) == 0:
                break

            execution_results = []
            for gen in tool_calls:
                try:
                    (name, raw_args) = next(iter(gen.items()))
                    if name != "submit_solution":
                        LOG.warning(f"unknown tool {name}")
                        execution_results.append(json.dumps({"error": f"unknown tool {name}"}))
                        continue
                    args = raw_args
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                        except Exception:
                            LOG.warning(f"invalid arguments {raw_args}")
                            execution_results.append(json.dumps({"error": "invalid arguments"}))
                            continue
                    code = args["code"]
                    sample = bool(args["sample"])
                    eval_payload = {**data_point, "generation": code, "only_sample_tests": sample}
                    eval_result = await self.evaluator.eval_single(eval_payload)
                    execution_results.append(json.dumps(eval_result))
                except Exception as e:
                    LOG.warning(f"error {e}")
                    execution_results.append(json.dumps({"error": str(e)}))

            for execution_result, tool_call_id in zip(execution_results, tool_call_ids):
                state_dict["messages"].append(
                    {"role": "tool", "content": execution_result, "tool_call_id": tool_call_id}
                )

            step_count += 1
            if step_count >= min(int(self.cfg.max_steps), MAXIMUM_STEP_LIMIT):
                LOG.info(f"Forced stop after {min(int(self.cfg.max_steps), MAXIMUM_STEP_LIMIT)} steps.")
                break

        out = {
            "id": data_point["id"],
            "generation": final_solution,
            "num_generated_tokens": sum(num_generated_tokens_list),
            "num_generated_tokens_list": num_generated_tokens_list,
        }
        if self.cfg.count_prompt_tokens:
            out["num_input_tokens"] = sum(num_input_tokens_list)
            out["num_input_tokens_list"] = num_input_tokens_list
        if out_of_context:
            out["error"] = "_ran_out_of_context_"
        return out


GENERATION_TASK_CLASS = AgentToolsGenerationTask


@hydra.main(version_base=None, config_name="base_agent_tools_generation_config")
def agent_tools_generation(cfg: AgentToolsConfig):
    cfg = AgentToolsConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    task = AgentToolsGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(AgentToolsConfig, server_params=server_params())

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        agent_tools_generation()
