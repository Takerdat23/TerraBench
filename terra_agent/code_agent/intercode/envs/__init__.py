"""InterCode environment exports.

Optional environments are imported lazily so the Python code-agent service does
not require SQL/CTF/SWE dependencies at process startup.
"""

from terra_agent.code_agent.intercode.envs.ic_env import (
    ACTION_EXEC,
    AGENT_OBS,
    CORRUPT_GOLD,
    EVAL_OBS,
    REWARD,
    IntercodeEnv,
)

__all__ = [
    "ACTION_EXEC",
    "AGENT_OBS",
    "CORRUPT_GOLD",
    "EVAL_OBS",
    "REWARD",
    "IntercodeEnv",
    "BashEnv",
    "CTFEnv",
    "PythonEnv",
    "SqlEnv",
    "SWEEnv",
]


def __getattr__(name: str):
    if name == "BashEnv":
        from terra_agent.code_agent.intercode.envs.bash.bash_env import BashEnv

        return BashEnv
    if name == "CTFEnv":
        from terra_agent.code_agent.intercode.envs.ctf.ctf_env import CTFEnv

        return CTFEnv
    if name == "PythonEnv":
        from terra_agent.code_agent.intercode.envs.python.python_env import PythonEnv

        return PythonEnv
    if name == "SqlEnv":
        from terra_agent.code_agent.intercode.envs.sql.sql_env import SqlEnv

        return SqlEnv
    if name == "SWEEnv":
        from terra_agent.code_agent.intercode.envs.swe.swe_env import SWEEnv

        return SWEEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
