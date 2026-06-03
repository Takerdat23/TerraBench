# Contributing

TerraBench is structured so contributors can add benchmark items, metrics, and tools without editing unrelated modules.

## Pull Requests

- Keep changes scoped.
- Add tests for new schemas, metrics, loaders, or tools.
- Run `pytest tests/` and `python scripts/run_smoke_test.py`.
- Avoid adding large data files, credentials, raw traces, or generated outputs.

## Adding A Tool

Implement `BaseTool` and register it:

```python
from terra_agent.tools.base import BaseTool
from terra_agent.tools.registry import register_tool

@register_tool
class MyTool(BaseTool):
    name = "my_tool"
    group = "simulation"
    description = "Example tool."
```

## Adding A Metric

Add a pure function under `terrabench/evaluation/` and register it through `terrabench.registry.metric_registry`.

## Adding Benchmark Items

Use the schema in `docs/benchmark_schema.md`. Public examples must be sanitized and must not include answer leakage from held-out benchmark items.

## Citation Expectations

If you use this repository in research, cite the TerraBench paper and any external data or simulator dependencies used in your workflow.
