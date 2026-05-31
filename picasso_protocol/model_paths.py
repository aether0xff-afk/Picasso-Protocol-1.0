from __future__ import annotations

from pathlib import Path


def resolve_model_path(
    filename: str,
    *,
    project_root: Path,
    cwd: Path | None = None,
) -> Path:
    """Find a model file in the current directory first, then the project root."""
    search_cwd = Path.cwd() if cwd is None else cwd
    candidates = [search_cwd / filename, project_root / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find {filename}. Checked: {checked}")
