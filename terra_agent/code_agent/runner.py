import argparse
import json
import logging
import os
import re

from typing import Any, Dict, List, Optional, Tuple

from terra_agent.code_agent.intercode.envs.ic_env import ACTION_EXEC, AGENT_OBS
from terra_agent.code_agent.intercode.envs.python.python_env import PythonEnv
from terra_agent.code_agent.intercode.utils import TraceLogger
from terra_agent.code_agent.experiments.policies import ChatGPTPolicy
from terra_agent.code_agent.experiments.utils import PROMPT_MAP
from rpyc.utils.classic import obtain
from rpyc.core.netref import BaseNetref

SETTING = "Python 3 Interpreter"
DEFAULT_IMAGE = "intercode-python"


def default_data_path() -> str:
    """Resolve default MBPP dataset path relative to this file."""
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, "data", "python", "mbpp", "ic_mbpp.json")


class CodeAgentRunner:
    def __init__(self, args: argparse.Namespace, trace_logger: Optional[TraceLogger] = None):
        if args.template not in PROMPT_MAP:
            raise ValueError(f"Prompt {args.template} not recognized; options: {list(PROMPT_MAP.keys())}")

        self.args = args
        self.logger = logging.getLogger(self.__class__.__name__)
        log_level = logging.DEBUG if args.verbose else logging.INFO
        self.logger.setLevel(log_level)
        env_kwargs = getattr(args, "env_kwargs", {}) or {}
        if not isinstance(env_kwargs, dict):
            raise ValueError("env_kwargs must be a dictionary when provided.")
        self.env = PythonEnv(
            image_name=args.image_name,
            data_path=args.data_path,
            verbose=args.verbose,
            is_agent=True,
            **env_kwargs,
        )
        self.trace_logger = trace_logger
        self.policy = ChatGPTPolicy(
            language="python",
            setting=SETTING,
            template=args.template,
            dialogue_limit=args.dialogue_limit,
            model=args.model,
            max_response_tokens=_normalize_limit(getattr(args, "max_response_tokens", None)) or 2048,
        )
        self.max_response_tokens = self.policy.max_response_tokens
        self.max_action_chars = _normalize_limit(getattr(args, "max_action_chars", None))
        self.max_output_chars = _normalize_limit(getattr(args, "max_output_chars", None))

        self.log_data: Dict[int, Dict] = {}
        self.log_path = None
        if args.log_dir:
            os.makedirs(args.log_dir, exist_ok=True)
            dataset_name = "interactive"
            if args.data_path:
                dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
            self.log_path = os.path.join(
                args.log_dir,
                f"code_agent_{dataset_name}_{args.model}_{args.max_turns}_turns.json",
            )
        self.auto_submit = not getattr(args, "no_auto_submit", False)

    def run(self) -> None:
        try:
            indices = self._task_indices()
            for idx in indices:
                log_episode, trace_id = self._run_single_task(idx)
                self._maybe_log_task_completion(idx, trace_id, log_episode)
        except KeyboardInterrupt:
            print("Keyboard interrupt detected")
        finally:
            if self.log_path and self.log_data:
                with open(self.log_path, "w") as fp:
                    json.dump(self.log_data, fp, indent=2)
            self.env.close()

    def _task_indices(self) -> List[int]:
        start = self.args.start_index
        if not hasattr(self.env, "data_loader"):
            raise ValueError("Environment has no dataset; provide data_path or use run_record().")
        end = min(start + self.args.num_tasks, len(self.env.data_loader))
        return list(range(start, end))

    def run_record(
        self,
        record: Dict,
        task_id: int = 0,
        request_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """
        Run a single task specified by `record` and return the interaction trace.
        Intended for programmatic use (e.g. API server).
        """
        log_episode, trace_id = self._run_single_task(
            task_id,
            record=record,
            store_log=False,
            request_meta=request_meta,
        )
        self._maybe_log_task_completion(task_id, trace_id, log_episode)
        return log_episode

    def _run_single_task(
        self,
        idx: int,
        record: Optional[Dict] = None,
        store_log: bool = True,
        request_meta: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> Tuple[Dict, Optional[str]]:
        if record is None:
            self.env.reset(idx)
            record = self.env.data_loader.get(idx)
            query = self.env.query
        else:
            self.env.reset()
            query = record.get("query", "")
            self.env.query = query
            self.env.record = record
            self.env.gold = record.get("gold", "N/A")

        self.policy.reset()

        observation = None
        reward = None
        valid_action = None

        turn_history = {"actions": [], "observations": [], "rewards": [], "valid_action": [], "infos": []}

        if self.args.verbose:
            print(f"\n------\nQuery {idx}: {query}")
        query_preview = _shorten_text(query)
        self.logger.info("Starting task %s | query='%s'", idx, query_preview)

        trace_id = self._ensure_trace(idx, query, request_meta, trace_id)

        for turn in range(self.args.max_turns):
            try:
                action, is_code = self.policy.forward(
                    query,
                    observation,
                    reward,
                    self.env.get_available_actions(),
                )
            except (ValueError, TypeError) as err:
                self.logger.exception("Policy error on turn %s: %s", turn, err)
                observation = f"Policy error: {err}"
                reward = 0
                valid_action = False
                turn_history["actions"].append("blocked")
                turn_history["observations"].append(str(observation))
                turn_history["rewards"].append(reward)
                turn_history["valid_action"].append(valid_action)
                debug_info = {
                    "reason": "policy_error",
                    "error": str(err),
                    "step_info": {"action_executed": False},
                }
                self._log_environment_step(idx, trace_id, turn, "blocked", observation, reward, valid_action, debug_info)
                break

            debug_info = {}
            action_for_log = _truncate_text(action, self.max_action_chars)
            if not is_code:
                reward = 0
                observation = self.policy.template.get_retry_msg()
                valid_action = False
                debug_info = {"reason": "non_code_action", "step_info": {"action_executed": False}}
                self.logger.debug("Turn %s ignored non-code action", turn)
                self._log_environment_step(idx, trace_id, turn, action_for_log, observation, reward, valid_action, debug_info)
            else:
                observation, reward, valid_action, debug_info = self._execute_action(action)
                self._log_environment_step(idx, trace_id, turn, action_for_log, observation, reward, valid_action, debug_info)

            if self.args.verbose:
                action_preview = _truncate_text(action, min(self.max_action_chars or 400, 400))
                observation_preview = _truncate_observation(observation, min(self.max_output_chars or 800, 800))
                print(f"- Turn {turn}")
                print(f"-- Action: {action_preview}")
                print(f"-- Observation: {observation_preview}")
                print(f"-- Reward: {reward}")

            self.logger.debug(
                "Turn %s | action='%s' | observation='%s' | reward=%s | valid=%s | debug=%s",
                turn,
                _shorten_text(action),
                _shorten_text(str(observation)),
                reward,
                valid_action,
                debug_info,
            )

            turn_history["actions"].append(action_for_log)
            turn_history["observations"].append(str(observation))
            turn_history["rewards"].append(reward)
            turn_history["valid_action"].append(valid_action)
            turn_history["infos"].append(debug_info)

            if reward == 1:
                break

        max_reward = max(turn_history["rewards"]) if turn_history["rewards"] else 0
        log_episode = {
            "environment": self.env.name,
            "dataset": self.args.data_path,
            "task_id": idx,
            "query": query,
            "turn_history": turn_history,
            "summary": {
                "max_reward": max_reward,
                "max_reward_idx": turn_history["rewards"].index(max_reward) if turn_history["rewards"] else -1,
                "turns_taken": len(turn_history["actions"]),
                "turns_max": self.args.max_turns,
            },
        }
        if record and "hardness" in record:
            log_episode["hardness"] = record["hardness"]

        if self.args.verbose:
            print(f"Query {idx} Finished\n-Reward: {max_reward}\n-Turns: {len(turn_history['actions'])}")
        self.logger.info(
            "Finished task %s | reward=%s | turns=%s/%s",
            idx,
            max_reward,
            len(turn_history["actions"]),
            self.args.max_turns,
        )
        if store_log and self.log_path:
            self.log_data[idx] = log_episode
        return log_episode, trace_id

    def _ensure_trace(
        self,
        task_id: int,
        query: str,
        request_meta: Optional[Dict[str, Any]],
        trace_id: Optional[str],
    ) -> Optional[str]:
        if not self.trace_logger:
            return trace_id
        if trace_id:
            return trace_id

        payload: Dict[str, Any] = {
            "query": query,
            "model": self.args.model,
            "template": self.args.template,
            "max_turns": self.args.max_turns,
            "dataset": self.args.data_path,
        }
        if request_meta:
            payload["request_meta"] = request_meta

        return self.trace_logger.begin_task(task_id, payload)

    def _log_environment_step(
        self,
        task_id: int,
        trace_id: Optional[str],
        turn: int,
        action: str,
        observation: Any,
        reward: Optional[float],
        valid_action: Optional[bool],
        debug_info: Dict[str, Any],
    ) -> None:
        if not self.trace_logger or not trace_id:
            return
        step_info = debug_info.get("step_info") if isinstance(debug_info, dict) else None
        info = step_info if isinstance(step_info, dict) else {}
        self.trace_logger.log_environment_step(
            task_id=task_id,
            trace_id=trace_id,
            turn=turn,
            action=action,
            observation=observation,
            reward=reward,
            valid_action=valid_action,
            info=info,
            debug=debug_info,
        )

    def _maybe_log_task_completion(
        self,
        task_id: int,
        trace_id: Optional[str],
        log_episode: Dict,
    ) -> None:
        if not self.trace_logger or not trace_id:
            return
        turn_history = log_episode.get("turn_history", {})
        observations = turn_history.get("observations", [])
        final_obs = ""
        if observations:
            final_obs = str(observations[-1]).replace("\n", " ")
            if len(final_obs) > 120:
                final_obs = f"{final_obs[:117]}..."
        self.trace_logger.log_agent_response(
            task_id=task_id,
            trace_id=trace_id,
            summary=log_episode.get("summary", {}),
            final_observation=final_obs,
            turn_history=turn_history,
        )

    def reconfigure_policy(self, model: Optional[str] = None, template: Optional[str] = None,
                           dialogue_limit: Optional[int] = None) -> None:
        """
        Update policy configuration. Useful for long-lived service instances.
        """
        changed = False
        if model and model != self.args.model:
            self.args.model = model
            changed = True
        if template and template != self.args.template:
            if template not in PROMPT_MAP:
                raise ValueError(f"Prompt {template} not recognized; options: {list(PROMPT_MAP.keys())}")
            self.args.template = template
            changed = True
        if dialogue_limit is not None and dialogue_limit != self.args.dialogue_limit:
            self.args.dialogue_limit = dialogue_limit
            changed = True
        if changed:
            self.policy = ChatGPTPolicy(
                language="python",
                setting=SETTING,
                template=self.args.template,
                dialogue_limit=self.args.dialogue_limit,
                model=self.args.model,
                max_response_tokens=self.max_response_tokens,
            )

    def set_auto_submit(self, enabled: bool) -> None:
        self.auto_submit = enabled
        if hasattr(self.args, "no_auto_submit"):
            self.args.no_auto_submit = not enabled

    def _execute_action(self, action: str) -> Tuple[str, float, bool, Dict]:
        if self.max_action_chars and len(action) > self.max_action_chars:
            observation = (
                f"Action too long ({len(action)} chars). "
                "Shorten the code and avoid large literals or verbose output."
            )
            debug_info = {
                "reason": "action_too_long",
                "action_chars": len(action),
                "max_action_chars": self.max_action_chars,
                "step_info": {"action_executed": False},
            }
            return observation, 0, False, debug_info
        observation, _, _, info = self.env.step(action)
        observation = _materialize(observation)
        info = _materialize(info)
        valid_action = info.get(ACTION_EXEC, True)
        reward = 0
        debug_info: Dict = {"step_info": info}

        if self.auto_submit and action.startswith("def"):
            func_match = re.match(r"def (\w+)\(", action)
            if func_match:
                func_name = func_match.group(1)
                _, reward, _, submit_info = self.env.step(f"submit {func_name}")
                submit_info = _materialize(submit_info)
                observation = submit_info
                debug_info["submit_info"] = submit_info
                if reward != 1 and AGENT_OBS in submit_info:
                    agent_obs = _materialize(submit_info[AGENT_OBS])
                    failing = False
                    if isinstance(agent_obs, dict):
                        for v in agent_obs.values():
                            local_val = _materialize(v)
                            if isinstance(local_val, dict) and len(local_val.get("error", "")) > 0:
                                failing = True
                                break
                    if failing:
                        observation = "Test case did not pass. Please try again."
                    debug_info["submit_tests"] = agent_obs
                valid_action = submit_info.get(ACTION_EXEC, valid_action)
            else:
                observation = "Unable to identify function name for submission."
                debug_info["submit_error"] = "missing_function_name"
        if self.max_output_chars:
            observation = _truncate_observation(observation, self.max_output_chars)
            debug_info = _truncate_observation(debug_info, self.max_output_chars)
        return observation, reward, valid_action, debug_info


def _materialize(value):
    """
    Convert remote RPyC objects into local Python equivalents recursively.
    """
    if isinstance(value, BaseNetref):
        try:
            value = obtain(value)
        except Exception:
            return value

    if isinstance(value, dict):
        return {_materialize(k): _materialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_materialize(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_materialize(v) for v in value)
    return value


def _shorten_text(text: str, limit: int = 120) -> str:
    if not text:
        return ""
    single_line = " ".join(text.strip().split())
    return single_line[:limit] + ("..." if len(single_line) > limit else "")


def _normalize_limit(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _truncate_text(text: str, limit: Optional[int]) -> str:
    if limit is None or len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated]"


def _truncate_observation(value: Any, limit: Optional[int]) -> Any:
    if limit is None:
        return value
    if isinstance(value, BaseNetref):
        try:
            value = obtain(value)
        except Exception:
            return _truncate_text(str(value), limit)
    if isinstance(value, str):
        return _truncate_text(value, limit)
    if isinstance(value, dict):
        try:
            return {k: _truncate_observation(v, limit) for k, v in value.items()}
        except Exception:
            return _truncate_text(str(value), limit)
    if isinstance(value, list):
        try:
            return [_truncate_observation(v, limit) for v in value]
        except Exception:
            return _truncate_text(str(value), limit)
    if isinstance(value, tuple):
        try:
            return tuple(_truncate_observation(v, limit) for v in value)
        except Exception:
            return _truncate_text(str(value), limit)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the code agent using OpenAI or Anthropic chat models in the Python InterCode environment.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=default_data_path(),
        help="Path to dataset of math problems (default: MBPP subset).",
    )
    parser.add_argument(
        "--image_name",
        type=str,
        default=DEFAULT_IMAGE,
        help="Docker image name that hosts the Python environment.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Chat model to use with the agent (OpenAI or Anthropic).",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="claude_prompt",
        help="Prompt template key to guide the policy.",
    )
    parser.add_argument(
        "--dialogue_limit",
        type=int,
        default=None,
        help="Optional limit on dialogue history retained by the policy.",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=7,
        help="Maximum number of interaction turns per task.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Task index to start from within the dataset.",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=1,
        help="Number of tasks to run from the starting index.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to store JSON logs for completed runs.",
    )
    parser.add_argument(
        "--trace_log_path",
        type=str,
        default=None,
        help="Optional JSONL file path to record agent/environment trace events.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed step-by-step output.",
    )
    parser.add_argument(
        "--no_auto_submit",
        action="store_true",
        help="Disable automatic submit calls after defining functions.",
    )
    parser.add_argument(
        "--max_action_chars",
        type=int,
        default=8000,
        help="Maximum characters allowed in a single action (0 to disable).",
    )
    parser.add_argument(
        "--max_output_chars",
        type=int,
        default=8000,
        help="Maximum characters kept per output field (0 to disable).",
    )
    parser.add_argument(
        "--max_response_tokens",
        type=int,
        default=2048,
        help="Maximum tokens for each model response (0 uses default).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trace_logger = TraceLogger(args.trace_log_path) if args.trace_log_path else None
    runner = CodeAgentRunner(args, trace_logger=trace_logger)
    try:
        runner.run()
    finally:
        if trace_logger:
            trace_logger.close()


MathAgentRunner = CodeAgentRunner
