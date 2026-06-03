import ast
import re
import rpyc

from rpyc.core.netref import BaseNetref
from rpyc.utils.classic import obtain
from typing import Any, Dict, Optional, Tuple

from terra_agent.code_agent.intercode.envs.ic_env import (
    IntercodeEnv,
    ACTION_EXEC, AGENT_OBS, EVAL_OBS, REWARD
)

CONTAINER_PORT = 3006
DEFAULT_HOST_PORT = 3006
RESET_KEYWORD = "RESET_CONTAINER_SPECIAL_KEYWORD"
LOG_PREVIEW_CHARS = 400


class PythonExecutionTimeoutError(RuntimeError):
    def __init__(self, timeout_seconds: Optional[float], command: str):
        self.timeout_seconds = timeout_seconds
        self.command = command
        timeout_text = "unknown" if timeout_seconds is None else f"{timeout_seconds:g}"
        super().__init__(f"Python execution timed out after {timeout_text} seconds.")


class PythonEnv(IntercodeEnv):
    """Gym environment for python shell"""
    name = "ic_python"
    timeout_exception_cls = PythonExecutionTimeoutError

    def __init__(self, image_name: str, **kwargs):
        self.host_port = _normalize_host_port(kwargs.get("host_port"))
        kwargs['ports'] = {f"{CONTAINER_PORT}/tcp": self.host_port}
        super(PythonEnv, self).__init__(image_name, **kwargs)
        # Disable RPyC's default 30s sync timeout so long-running Python executions don't expire
        self.conn = rpyc.connect("localhost", self.host_port, config={"sync_request_timeout": None})
        self.is_agent = kwargs.get("is_agent", False)
        self.execution_timeout_seconds = _normalize_timeout(kwargs.get("execution_timeout_seconds"))
    
    def reset_container(self) -> None:
        self._remote_execute(RESET_KEYWORD)
    
    def exec_action(self, action: str) -> None:
        try:
            if action.strip().startswith("def "):
                if not self.is_agent:
                    function_definition = self.input_multiline_function()
                    action = action + "\n" + function_definition
            else:
                action = self.wrap_with_print(action)
            self.logger.info("Command run: %s", _preview_text(action))
            self.observation = self._remote_execute(action)
            error_text = _mapping_lookup(self.observation, "error", "")
            self.info[ACTION_EXEC] = len(error_text) == 0
        except PythonExecutionTimeoutError:
            self.info[ACTION_EXEC] = False
            self.info["timed_out"] = True
            raise
        except Exception as err:
            self.observation = f"Error executing action: {err}"
            self.info[ACTION_EXEC] = False
    
    def get_reward(self) -> Tuple[float, Dict]:
        MAP_DATASET_TO_REWARD = {
            "ic_apps": self.get_reward_apps,
            "ic_mbpp": self.get_reward_mbpp,
        }
        dataset = self.data_path.split("/")[-1].split(".")[0]
        return MAP_DATASET_TO_REWARD[dataset]()
    
    def close(self, remove_container: bool = False):
        self.logger.info("Beginning environment shutdown...")
        conn = getattr(self, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        container = getattr(self, "container", None)
        if container is None:
            return
        try:
            container.stop()
            self.logger.info("Agent container stopped")
        except Exception:
            pass
        if remove_container:
            try:
                container.remove(force=True)
                self.logger.info("Agent container removed")
            except Exception:
                pass

    def _remote_execute(self, command: str, timeout_seconds: Optional[float] = None):
        timeout_seconds = self.execution_timeout_seconds if timeout_seconds is None else _normalize_timeout(timeout_seconds)
        if timeout_seconds is None:
            return _raise_on_timeout_response(
                _materialize_remote(self.conn.root.execute(command)),
                timeout_seconds,
                command,
            )
        try:
            return _raise_on_timeout_response(
                _materialize_remote(self.conn.root.execute(command, timeout_seconds)),
                timeout_seconds,
                command,
            )
        except TypeError as err:
            raise RuntimeError(
                "Execution timeouts require the updated python execution server. "
                "Rebuild the Docker image with `bash setup.sh`, then recreate the "
                "`intercode-python_ic_ctr` container or restart the API with the "
                "updated container lifecycle logic."
            ) from err
    
    ############################
    ### MARK: Helper methods ###
    ############################
    def input_multiline_function(self):
        lines = []
        while True:
            line = input(". ")
            if len(line) == 0:
                break
            lines.append(line)
        return "\n".join(lines)
    
    def wrap_with_print(self, command):
        # Parse the command as an AST (Abstract Syntax Tree)
        parsed_command = ast.parse(command.strip())

        # Check if the command contains an assignment node, print node, or import
        has_assignment = any(isinstance(node, ast.Assign) for node in ast.walk(parsed_command))
        has_print = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'print' for node in ast.walk(parsed_command))
        has_import = any(isinstance(node, ast.Import) for node in ast.walk(parsed_command))
        is_assert = command.strip().startswith("assert")

        # Wrap the command with "print" if it's not an assignment and does not have a "print" statement
        if not any([has_assignment, has_print, has_import, is_assert]):
            return f"print({command})"
        else:
            return command
    
    ##############################
    ### MARK: Reward functions ###
    ##############################
    def get_reward_apps(self):
        self.info = {}
        return 0.0, self.info

    def get_reward_mbpp(self):
        self.info = {}

        # Get function from `submit` action
        # TODO: Assert that function name is given upon `submit` action
        last_action = self.trajectory[-1][0]
        func_name = last_action.split(" ")[1]

        # Get gold function name, assign to submitted function
        func_name_ref = re.search(r'def (\w+)\(', self.gold).group(1)
        self._remote_execute(f"{func_name_ref} = {func_name}")

        # Run tests against submitted function
        results_pred = {}
        self._remote_execute(self.record["test_setup_code"])
        for test in self.record["tests"]:
            results_pred[test] = self._remote_execute(test)

        # Load gold + run tests
        results_gold = {}
        self._remote_execute(RESET_KEYWORD)
        self._remote_execute(self.record["test_setup_code"])
        self._remote_execute(self.gold)
        for test in self.record["tests"]:
            results_gold[test] = self._remote_execute(test)
        
        self.info["submitted_function"] = func_name
        self.info[AGENT_OBS] = results_pred
        self.info[EVAL_OBS] = results_gold

        # Compute reward
        correct = 0
        for test, output in results_pred.items():
            output_gold = results_gold[test]
            if output == output_gold:
                correct += 1
        self.info[REWARD] = float(correct) / len(results_pred)
        self.reward = self.info[REWARD]

        self.logger.info(f"Info: {self.info}")
        self.logger.info(f"Reward: {self.reward}")
        return self.reward, self.info


def _normalize_timeout(value: Optional[float]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _normalize_host_port(value: Optional[int]) -> int:
    if value is None or value == "":
        return DEFAULT_HOST_PORT
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return DEFAULT_HOST_PORT
    return parsed if parsed > 0 else DEFAULT_HOST_PORT


def _materialize_remote(value: Any) -> Any:
    """
    Convert remote RPyC objects into local Python values recursively.
    """
    if isinstance(value, BaseNetref):
        try:
            value = obtain(value)
        except Exception:
            return value

    if isinstance(value, dict):
        return {_materialize_remote(k): _materialize_remote(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_materialize_remote(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_materialize_remote(v) for v in value)
    return value


def _mapping_lookup(value: Any, key: str, default: Any = None) -> Any:
    """
    Read from local dicts and dict-like RPyC netrefs without relying on `.get()`.
    """
    if isinstance(value, BaseNetref):
        try:
            value = obtain(value)
        except Exception:
            pass

    if isinstance(value, dict):
        return value[key] if key in value else default

    try:
        return value[key] if key in value else default
    except Exception:
        return default


def _preview_text(value: Any, limit: int = LOG_PREVIEW_CHARS) -> str:
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else f"{text[:limit]}... [truncated]"


def _raise_on_timeout_response(response: Any, timeout_seconds: Optional[float], command: str) -> Any:
    if _mapping_lookup(response, "timed_out", False):
        response_timeout = _normalize_timeout(_mapping_lookup(response, "timeout_seconds", timeout_seconds))
        raise PythonExecutionTimeoutError(response_timeout, _preview_text(command))
    return response
