"""Dataset registry for benchmark releases."""

from __future__ import annotations

_DATASETS: dict[str, dict] = {}


def register_dataset(name: str, metadata: dict) -> None:
    _DATASETS[name] = dict(metadata)


def get_dataset(name: str) -> dict:
    return dict(_DATASETS[name])


def list_datasets() -> list[str]:
    return sorted(_DATASETS)
