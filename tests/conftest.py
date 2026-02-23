"""Pytest configuration for management UI tests."""

from __future__ import annotations

from pathlib import Path
import sys


def _ensure_busy_ongoing_on_path() -> None:
    """Add the local busy-38-ongoing project root to import resolution."""

    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    candidate_roots = [
        repo_root.parent / "busy-38-ongoing",
        repo_root / "busy-38-ongoing",
    ]
    for candidate in candidate_roots:
        if candidate.is_dir():
            candidate_path = str(candidate)
            if candidate_path not in sys.path:
                sys.path.insert(0, candidate_path)


_ensure_busy_ongoing_on_path()
