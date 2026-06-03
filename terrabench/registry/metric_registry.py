"""Metric registry for extension points."""

from __future__ import annotations

from collections.abc import Callable

MetricFn = Callable[..., object]
_METRICS: dict[str, MetricFn] = {}


def register_metric(name: str, fn: MetricFn) -> MetricFn:
    _METRICS[name] = fn
    return fn


def get_metric(name: str) -> MetricFn:
    return _METRICS[name]


def list_metrics() -> list[str]:
    return sorted(_METRICS)
