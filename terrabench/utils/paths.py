"""Central path handling for TerraBench."""

from __future__ import annotations

from pathlib import Path


def project_root(start: str | Path | None = None) -> Path:
    """Find the repository root by walking upward to `pyproject.toml`."""

    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd().resolve()


def resolve_path(path: str | Path, *, root: str | Path | None = None) -> Path:
    """Resolve a path relative to a configured root without assuming cwd."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    base = Path(root).expanduser().resolve() if root is not None else project_root()
    return (base / candidate).resolve()


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
