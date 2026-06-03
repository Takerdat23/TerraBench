# InterCode Code Agent Service

This walkthrough shows how to stand up the InterCode Python execution environment and expose the Code Agent FastAPI server so other agents (e.g., Agentscope workflows) can call it.

## 1. Create Environment
```bash
cd /path/to/TerraBench
conda env create -f environment.yml
conda activate terrabench
pip install -e .
```

## 2. Configure API Keys
Add your OpenAI key either as an environment variable:
```bash
export OPENAI_API_KEY=""
```
or in `key.cfg` (same directory as `runner.py`; `keys.cfg` is also supported for backwards compatibility):
```
OPENAI_API_KEY: ''
ANTHROPIC_API_KEY: ''
```

If you plan to run the agent with [Anthropic](https://www.anthropic.com) Claude models instead of OpenAI, populate the `ANTHROPIC_API_KEY` variable the same way (e.g., `export ANTHROPIC_API_KEY=""`) and pass a Claude model name like `claude-3.5` or `claude-3.5-100k` via the `--model` flag or the service payload.

You can also pick the default chat model for the FastAPI deployment by setting `CODE_AGENT_MODEL` before launching `uvicorn`. For example:
```bash
export CODE_AGENT_MODEL="claude-haiku-4-5-20251001"
```
If you do not explicitly set this environment variable, the service falls back to `gpt-4`.
The service still accepts legacy `MATH_AGENT_*` environment variables and the
`/math-agent` routes for existing callers, but new TerraBench code should use
`CODE_AGENT_*` and `/code-agent`.

If you want the API to handle multiple code-agent requests in parallel, set the
service pool size before launching `uvicorn`:
```bash
export CODE_AGENT_POOL_SIZE="4"
```
Each pool slot gets its own isolated execution container. Requests beyond the
pool size wait until a slot becomes available.
By default the pool binds host ports starting at `3006` (`3006`, `3007`, `3008`, ...).
If you need a different range, set:
```bash
export CODE_AGENT_POOL_PORT_BASE="3200"
```
Idle slots are dispatched in round-robin order by default. If you want the pool
to prefer the coolest idle container instead, set:
```bash
export CODE_AGENT_LOAD_BALANCING_STRATEGY="least_loaded"
```
If you want to keep routing away from containers that are still burning CPU after
their last request, set a CPU guardrail as well:
```bash
export CODE_AGENT_MAX_CONTAINER_CPU_PERCENT="90"
```

The execution container mounts the main TerraBench repository at `/workspace`
by default and uses that as the Python working directory. This avoids path
mismatches when the API is launched from `terra_agent/code_agent` but requests refer to
files in the parent project. By default the mount is read-only; generated files
should still be written under `/uploads` so the API can return them as artifacts.
You can override or disable the workspace mount before launching `uvicorn`:
```bash
export CODE_AGENT_WORKSPACE_HOST_DIR="/path/to/TerraBench"
export CODE_AGENT_WORKSPACE_CONTAINER_DIR="/workspace"
export CODE_AGENT_WORKSPACE_READ_ONLY="true"
# set CODE_AGENT_WORKSPACE_HOST_DIR="none" to disable the workspace mount
```

By default, those execution containers are only isolated by process/container state.
They still share the same host CPU and RAM unless you set explicit resource caps.
To prevent one bad request from starving the rest of the pool, you can set per-slot
container limits before launching `uvicorn`:
```bash
export CODE_AGENT_CONTAINER_NANO_CPUS="1000000000"
export CODE_AGENT_CONTAINER_MEM_LIMIT="2g"
export CODE_AGENT_CONTAINER_PIDS_LIMIT="256"
```
Those values apply to each slot independently. For example, with
`CODE_AGENT_POOL_SIZE="4"` and `CODE_AGENT_CONTAINER_MEM_LIMIT="2g"`, the pool can
consume up to roughly `8 GiB` across all execution containers.

By default, each Python execution inside a request times out after `30` seconds.
You can override that deployment default with:
```bash
export CODE_AGENT_EXECUTION_TIMEOUT_SECONDS="45"
```
You can also override this per request with `execution_timeout_seconds`.
When a Python step exceeds the timeout, the execution worker is terminated and
restarted immediately so queued requests do not get stuck behind a hung run.
That means interpreter state from the timed-out step is discarded.
In addition, the API now treats that timeout as a slot-health event: the
timed-out request fails with HTTP `504`, and only the affected pool slot is
recycled (container removed and recreated) before it goes back into service.
The rest of the pool stays online.

If you want to disable execution timeouts entirely for the deployment, set:
```bash
export CODE_AGENT_DISABLE_TIMEOUT="true"
```
This is a hard override: while it is enabled, the service ignores both
`CODE_AGENT_EXECUTION_TIMEOUT_SECONDS` and per-request `execution_timeout_seconds`.

Verbose terminal tracing is disabled by default to avoid flooding the console with
large generated code blocks or observations. If you want the turn-by-turn terminal
prints back, enable them explicitly before starting `uvicorn`:
```bash
export CODE_AGENT_VERBOSE="true"
```

## 3. Build Environment Containers
InterCode uses Docker for its execution sandboxes. Make sure Docker Desktop/daemon is running, then build the images:
```bash
bash setup.sh
```

## 4. Start the Code Agent API
Install the web dependencies (if they are not already present) and launch the FastAPI service:
```bash
cd /path/to/TerraBench
uvicorn terra_agent.code_agent.api:app --host 0.0.0.0 --port 8000
```
From the TerraBench repository root you can also run:
```bash
scripts/start_code_agent.sh
```
With the default pool size of `1`, the first request will automatically spawn the
`intercode-python` container named `intercode-python_ic_ctr`. For pool sizes
greater than `1`, the service creates one container per slot, named
`intercode-python_ic_ctr_0`, `intercode-python_ic_ctr_1`, and so on, using host
ports `3006`, `3007`, `3008`, ... by default.
If you change the execution server code or timeout behavior, rebuild the Docker image with `bash setup.sh` before restarting the API so the container picks up the updated server.

## 5. Call the API
Send tasks to the agent with any HTTP client. Example `curl` request:
```bash
curl -X POST http://localhost:8000/code-agent \
     -H "Content-Type: application/json" \
     -d '{
           "query": "Compute the mean and standard deviation of the list.",
           "variables": {"numbers": [12, 18, 21, 30, 27]},
           "max_turns": 4,
           "execution_timeout_seconds": 15
         }'
```
The response includes the full interaction trace (code actions, observations, rewards, and raw execution details) that you can relay back to your calling agent.

To inspect live pool execution state and see which slot is currently busy, call:
```bash
curl http://localhost:8000/pool-status
```
The response includes the pool size, waiting request count, slot/container mapping,
current running task preview, runtime duration, last completion status, and the
configured CPU/memory/PID caps per slot. It also reports slot recycle metadata
(`recycle_count`, `last_recycled_at`, `last_recycle_reason`) so you can see
which slot was rebuilt after a timeout.

Trace-level input/output for both the agent and the python environment are also written to `logs/code_agent_trace.jsonl` as the service runs; each JSONL record captures the request, each environment step, and the final response summary.

### Uploading Large Artifacts
If your task requires data that would overwhelm the JSON payload, upload it first and then reference the returned identifier in your `variables`. The service exposes `POST /code-agent/upload`, which accepts `multipart/form-data` with a `file` field and optional `metadata` (JSON string). Example:

```bash
curl -X POST http://localhost:8000/code-agent/upload \
     -H "X-Upload-Token: ${CODE_AGENT_UPLOAD_TOKEN}" \
     -F "file=@/path/to/era5.nc" \
     -F 'metadata={"source":"era5","notes":"Full resolution"}'
```

The response looks like:

```json
{
  "file_id": "5f03b8b6e7204c0bb2bc7573cd07716c",
  "filename": "era5.nc",
  "size_bytes": 123456,
  "sha256": "...",
  "download_url": "http://localhost:8000/code-agent/upload/5f03b8b6e7204c0bb2bc7573cd07716c",
  "expires_at": "2024-07-03T18:42:11.912Z"
}
```

Include the identifier in your code-agent request as a variable ending with `_upload_id`, for example:

```json
{
  "query": "Summarize the ERA5 statistics.",
  "variables": {
    "era5_upload_id": "5f03b8b6e7204c0bb2bc7573cd07716c"
  }
}
```

Before running the task the service resolves these references, copies the staged file into the execution container, and exposes its path (e.g., `era5_path`) to the prompt so the agent code can load it directly. Uploaded files are stored in `./uploads` (configurable via `CODE_AGENT_UPLOAD_DIR`), mounted into the container at `/uploads`, and optionally deleted after use when `CODE_AGENT_UPLOAD_DELETE_AFTER_USE=1`.

## 6. Shut Down
Stop the API with `Ctrl+C` and clean up the container when you are done:
```bash
docker stop intercode-python_ic_ctr
```

You now have a reusable code agent endpoint backed by the InterCode Python environment. Hook it into your agent framework by issuing HTTP POST requests with the prompt and any structured inputs you want the model to reference. For advanced usage (e.g., adjusting models, prompt templates, auto-submit behavior), check `terra_agent/code_agent/service.py` for available configuration options.

## 🪪 License
MIT. Check `LICENSE.md`.
