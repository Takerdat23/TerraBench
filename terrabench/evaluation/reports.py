"""Report generation from saved metric files."""

from __future__ import annotations

from pathlib import Path

from terrabench.evaluation.bootstrap import _load_metric_rows
from terrabench.utils.json_utils import write_json


def generate_report(metrics_dir: str | Path, out_dir: str | Path) -> dict:
    rows = _load_metric_rows(metrics_dir)
    names = sorted({name for row in rows for name in row})
    summary = {}
    for name in names:
        values = [row[name] for row in rows if name in row]
        summary[name] = {"mean": sum(values) / len(values), "n": len(values)} if values else {"mean": 0.0, "n": 0}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metric_summary.json", summary)
    lines = ["# TerraBench Metric Summary", ""]
    for name, payload in summary.items():
        lines.append(f"- `{name}`: mean={payload['mean']:.4f}, n={payload['n']}")
    (out / "metric_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
