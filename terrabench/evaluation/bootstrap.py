"""Bootstrap confidence intervals over saved metric rows."""

from __future__ import annotations

import csv
import random
from pathlib import Path

from terrabench.utils.json_utils import write_json


def _load_metric_rows(metrics_dir: str | Path) -> list[dict[str, float]]:
    rows = []
    for path in Path(metrics_dir).glob("*.csv"):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({key: float(value) for key, value in row.items() if key != "item_id" and value not in {"", None}})
    for path in Path(metrics_dir).glob("*.json"):
        payload = __import__("json").loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "metrics" in payload:
            rows.append({key: float(value) for key, value in payload["metrics"].items()})
    return rows


def bootstrap_metrics(metrics_dir: str | Path, out_dir: str | Path, *, n: int = 200, seed: int = 7) -> dict:
    rows = _load_metric_rows(metrics_dir)
    rng = random.Random(seed)
    metric_names = sorted({name for row in rows for name in row})
    summary = {}
    for name in metric_names:
        samples = []
        values = [row[name] for row in rows if name in row]
        if not values:
            continue
        for _ in range(n):
            draw = [rng.choice(values) for _ in values]
            samples.append(sum(draw) / len(draw))
        samples.sort()
        summary[name] = {
            "mean": sum(values) / len(values),
            "ci_low": samples[int(0.025 * (len(samples) - 1))],
            "ci_high": samples[int(0.975 * (len(samples) - 1))],
            "n": len(values),
        }
    write_json(Path(out_dir) / "bootstrap_summary.json", summary)
    return summary
