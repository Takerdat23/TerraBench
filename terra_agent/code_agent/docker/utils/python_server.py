import multiprocessing
import os
import rpyc
import sys
import threading
from io import StringIO
from typing import Optional

ORIGINAL_GLOBAL = dict(globals())
OUTPUT_LIMIT_ENV = "PYTHON_SERVER_OUTPUT_CHARS"
ERROR_LIMIT_ENV = "PYTHON_SERVER_ERROR_CHARS"
EXECUTION_TIMEOUT_ENV = "PYTHON_SERVER_EXECUTION_TIMEOUT_SECONDS"
WORKDIR_ENV = "PYTHON_SERVER_WORKDIR"
RESET_KEYWORD = "RESET_CONTAINER_SPECIAL_KEYWORD"
DEFAULT_OUTPUT_LIMIT = 8000


def _read_limit(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed


def _truncate(value: str, limit: int) -> tuple[str, bool, int]:
    if limit <= 0:
        return value, False, len(value)
    if len(value) <= limit:
        return value, False, len(value)
    return f"{value[:limit]}... [truncated]", True, len(value)


def _read_timeout(value: Optional[float] = None) -> Optional[float]:
    raw_value = os.getenv(EXECUTION_TIMEOUT_ENV) if value is None else value
    if raw_value in (None, ""):
        return None
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _scope_keys(scope: dict) -> list[str]:
    return list(scope.keys())


def _activate_workdir() -> Optional[str]:
    workdir = os.getenv(WORKDIR_ENV)
    if not workdir:
        return None
    if not os.path.isdir(workdir):
        raise FileNotFoundError(f"{WORKDIR_ENV} does not exist or is not a directory: {workdir}")
    os.chdir(workdir)
    cwd = os.getcwd()
    for entry in (cwd, workdir):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return cwd


def _execution_worker(conn) -> None:
    _activate_workdir()
    scope = dict(ORIGINAL_GLOBAL)

    while True:
        message = conn.recv()
        msg_type = message.get("type")
        if msg_type == "shutdown":
            break
        if msg_type != "execute":
            continue

        command = message["command"]
        output_buffer = StringIO()
        error_buffer = StringIO()
        previous_stdout = sys.stdout
        previous_stderr = sys.stderr

        try:
            if command == RESET_KEYWORD:
                scope = dict(ORIGINAL_GLOBAL)
                _activate_workdir()
            else:
                sys.stdout = output_buffer
                sys.stderr = error_buffer
                exec(command, scope)
        except Exception as err:
            error_message = f"Error: {err}"
        else:
            error_message = ""
        finally:
            sys.stdout = previous_stdout
            sys.stderr = previous_stderr

        output_raw = output_buffer.getvalue().strip()
        error_raw = error_buffer.getvalue().strip()
        if error_message:
            error_raw = "\n".join(part for part in [error_raw, error_message] if part).strip()

        conn.send(
            {
                "output": output_raw,
                "error": error_raw,
                "scope_keys": _scope_keys(scope),
            }
        )


def _get_multiprocessing_context():
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        return multiprocessing.get_context("spawn")


class MyService(rpyc.Service):
    def __init__(self):
        self.globals = dict(ORIGINAL_GLOBAL)
        self._context = _get_multiprocessing_context()
        self._lock = threading.Lock()
        self._worker = None
        self._conn = None
        self._start_worker()

    def on_connect(self, conn):
        pass

    def on_disconnect(self, conn):
        pass

    def _start_worker(self):
        parent_conn, child_conn = self._context.Pipe()
        worker = self._context.Process(target=_execution_worker, args=(child_conn,), daemon=True)
        worker.start()
        child_conn.close()
        self._conn = parent_conn
        self._worker = worker
        self.globals = dict(ORIGINAL_GLOBAL)

    def _terminate_worker(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None

        if self._worker is not None:
            if self._worker.is_alive():
                self._worker.terminate()
                self._worker.join(timeout=1)
            self._worker = None

    def _ensure_worker(self):
        if self._worker is None or not self._worker.is_alive() or self._conn is None:
            self._terminate_worker()
            self._start_worker()

    def close(self):
        with self._lock:
            self._terminate_worker()

    def exposed_execute(self, command, timeout_seconds=None):
        timeout_seconds = _read_timeout(timeout_seconds)
        with self._lock:
            self._ensure_worker()
            timed_out = False
            output_raw = ""
            error_raw = ""

            try:
                self._conn.send({"type": "execute", "command": command})
                if timeout_seconds is not None and not self._conn.poll(timeout_seconds):
                    timed_out = True
                    error_raw = f"Execution timed out after {timeout_seconds:g} seconds."
                    self._terminate_worker()
                    self._start_worker()
                else:
                    response = self._conn.recv()
                    output_raw = response.get("output", "")
                    error_raw = response.get("error", "")
                    self.globals = {name: None for name in response.get("scope_keys", [])}
            except Exception as err:
                error_raw = f"Error: {err}"
                self._terminate_worker()
                self._start_worker()

        output_limit = _read_limit(OUTPUT_LIMIT_ENV, DEFAULT_OUTPUT_LIMIT)
        error_limit = _read_limit(ERROR_LIMIT_ENV, output_limit)
        output, output_truncated, output_chars = _truncate(output_raw, output_limit)
        error, error_truncated, error_chars = _truncate(error_raw, error_limit)

        response = {"output": output, "error": error}
        if timed_out:
            response["timed_out"] = True
        if timeout_seconds is not None:
            response["timeout_seconds"] = timeout_seconds
        if output_truncated:
            response["output_truncated"] = True
            response["output_chars"] = output_chars
        if error_truncated:
            response["error_truncated"] = True
            response["error_chars"] = error_chars
        return response


if __name__ == "__main__":
    from rpyc.utils.server import ThreadPoolServer
    # Disable rpyc's default 30s sync_request_timeout so long-running code can complete
    server = ThreadPoolServer(
        MyService(),
        port=3006,
        protocol_config={"sync_request_timeout": None},
    )
    server.start()
