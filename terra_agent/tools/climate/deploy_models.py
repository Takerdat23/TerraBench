"""
HTTP deployment harness exposing climate model tools as simple REST endpoints.

Running this module will spin up a light-weight HTTP server that forwards POST
requests to the existing Agentscope tool functions for Aurora and
Pangu-Weather. This allows an external agent (or any HTTP client) to trigger
forecasts without having to execute Python code directly within the
ClimateAgent project.

Example:

```bash
python -m tools.deploy_models --host 0.0.0.0 --port 8090 --tools aurora pangu
```

Then, from another process:

```bash
curl -X POST http://localhost:8090/aurora \\
     -H "Content-Type: application/json" \\
     -d '{"surface_path": "...", "pressure_path": "...", "static_path": "..."}'
```
"""

import argparse
import inspect
import json
import logging
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

from agentscope.tool import ToolResponse

from terra_agent.tools.climate.aurora_tool import run_aurora_forecast
from terra_agent.tools.climate.pangu_tool import run_pangu_forecast

ToolCallable = Callable[..., ToolResponse]

LOGGER = logging.getLogger("climate_agent.deployer")


def _json_default(value: Any) -> Any:
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - numpy should already exist
        np = None  # type: ignore

    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, (np.datetime64, np.timedelta64)):
            return value.astype("datetime64[s]").astype(int)
    if hasattr(value, "__json__"):
        return value.__json__()  # type: ignore[attr-defined]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover - fallback for unexpected objects
            pass
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _tool_response_to_payload(name: str, response: ToolResponse) -> Dict[str, Any]:
    metadata = response.metadata or {}
    sanitized_metadata = json.loads(json.dumps(metadata, default=_json_default))
    messages: list[str] = []
    for block in response.content or []:
        text = getattr(block, "text", None)
        if text:
            messages.append(text)
    ok = not sanitized_metadata.get("error")
    return {
        "tool": name,
        "ok": bool(ok),
        "metadata": sanitized_metadata,
        "messages": messages,
    }


def _filter_kwargs(fn: ToolCallable, payload: Dict[str, Any]) -> Dict[str, Any]:
    signature = inspect.signature(fn)
    kwargs: Dict[str, Any] = {}
    missing: list[str] = []
    for param_name, param in signature.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param_name in payload:
            kwargs[param_name] = payload[param_name]
            continue
        if param.default is inspect._empty:
            missing.append(param_name)
    if missing:
        raise ValueError(f"Missing required parameters: {', '.join(missing)}")
    return kwargs


@dataclass
class DeploymentTool:
    """Metadata wrapper for an exposed tool."""

    name: str
    fn: ToolCallable
    description: str

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = _filter_kwargs(self.fn, payload)
        response = self.fn(**kwargs)
        return _tool_response_to_payload(self.name, response)


class ModelDeploymentManager:
    """Registry mapping model names to their callable tool wrappers."""

    def __init__(self) -> None:
        self._tools: Dict[str, DeploymentTool] = {}

    def register(self, name: str, fn: ToolCallable, description: Optional[str] = None) -> None:
        key = name.lower()
        desc = description or (fn.__doc__ or "").strip().split("\n")[0]
        self._tools[key] = DeploymentTool(name=key, fn=fn, description=desc)
        LOGGER.debug("Registered deployment tool '%s'", key)

    def available(self) -> Dict[str, str]:
        return {name: tool.description for name, tool in self._tools.items()}

    def dispatch(self, name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        key = name.lower()
        tool = self._tools.get(key)
        if tool is None:
            raise KeyError(key)
        return tool.invoke(payload)


class DeploymentServer(ThreadingHTTPServer):
    """HTTP server binding that stores the deployment manager."""

    def __init__(self, server_address: tuple[str, int], manager: ModelDeploymentManager) -> None:
        super().__init__(server_address, DeploymentRequestHandler)
        self.manager = manager


class DeploymentRequestHandler(BaseHTTPRequestHandler):
    """Simple JSON-over-HTTP handler for model deployment requests."""

    server: DeploymentServer

    def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _respond_error(self, message: str, status: HTTPStatus) -> None:
        LOGGER.debug("Responding with error %s: %s", status, message)
        self._send_json({"error": True, "message": message}, status=status)

    def log_message(self, format: str, *args: Any) -> None:  # pragma: no cover - suppress std logging
        LOGGER.info("%s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:  # noqa: N802 - method name fixed by BaseHTTPRequestHandler
        path = self.path.split("?", 1)[0]
        if path in ("/", "/tools", "/models"):
            payload = {
                "tools": self.server.manager.available(),
            }
            self._send_json(payload, status=HTTPStatus.OK)
            return
        if path in ("/health", "/healthz", "/status"):
            self._send_json({"status": "ok", "tools": list(self.server.manager.available())})
            return
        self._respond_error(f"Unsupported path '{path}'", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - method name fixed by BaseHTTPRequestHandler
        content_length = int(self.headers.get("Content-Length") or 0)
        raw_body = self.rfile.read(content_length) if content_length else b""
        if not raw_body:
            payload = {}
        else:
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError:
                self._respond_error("Request body must be valid JSON.", HTTPStatus.BAD_REQUEST)
                return

        path = self.path.split("?", 1)[0].strip("/")
        parts = [segment for segment in path.split("/") if segment]
        if not parts:
            self._respond_error("Tool name missing from URL path.", HTTPStatus.NOT_FOUND)
            return
        tool_name = parts[-1]

        try:
            result = self.server.manager.dispatch(tool_name, payload)
        except KeyError:
            self._respond_error(f"Unknown deployment tool '{tool_name}'.", HTTPStatus.NOT_FOUND)
            return
        except ValueError as exc:
            self._respond_error(str(exc), HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:  # pragma: no cover - runtime protection
            LOGGER.exception("Unhandled error during tool execution: %s", exc)
            self._respond_error("Tool execution failed. Inspect server logs for details.", HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json(result, status=HTTPStatus.OK)


DEFAULT_TOOLS: Dict[str, ToolCallable] = {
    "aurora": run_aurora_forecast,
    "pangu": run_pangu_forecast,
}


def build_manager(tool_names: Iterable[str]) -> ModelDeploymentManager:
    manager = ModelDeploymentManager()
    for name in tool_names:
        key = name.lower()
        fn = DEFAULT_TOOLS.get(key)
        if fn is None:
            raise ValueError(f"Unknown tool '{name}'. Available tools: {', '.join(sorted(DEFAULT_TOOLS))}.")
        manager.register(key, fn)
    return manager


def serve(
    *,
    host: str,
    port: int,
    tools: Iterable[str],
) -> None:
    manager = build_manager(tools)
    server = DeploymentServer((host, port), manager)
    LOGGER.info(
        "Serving deployment tools [%s] on http://%s:%d",
        ", ".join(sorted(manager.available())),
        host,
        port,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - manual shutdown
        LOGGER.info("Shutting down deployment server.")
    finally:
        server.server_close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Expose climate model tools over HTTP.")
    parser.add_argument("--host", default="127.0.0.1", help="Host/IP for the HTTP server.")
    parser.add_argument("--port", type=int, default=8080, help="TCP port for the HTTP server.")
    parser.add_argument(
        "--tools",
        nargs="+",
        default=list(DEFAULT_TOOLS.keys()),
        help=f"Subset of tools to expose (available: {', '.join(sorted(DEFAULT_TOOLS))}).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (DEBUG, INFO, WARNING...).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    serve(host=args.host, port=args.port, tools=args.tools)


if __name__ == "__main__":  # pragma: no cover
    main()
