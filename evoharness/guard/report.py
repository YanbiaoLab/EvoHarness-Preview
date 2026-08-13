# EvoHarness original: run manifests for honest, reproducible reporting
# (plan sections 4/7: manifests, holdout hash, cost accounting).
"""Run manifests: frozen at start, finalized at exit.

A manifest written only at the end describes only the runs that survived.
The runs that died -- storage stalls, sqlite faults, kills, all three
observed on this project's own GPU box -- left NO record of what they were:
which commit, which configs, which plugins. Every post-mortem then starts by
reconstructing the run's identity from shell history.

So the identity is frozen BEFORE the first model call, and the outcome is
appended at exit. A crash leaves the frozen part behind, which is exactly
the part a post-mortem needs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = 2


def _jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return obj


def _atomic_write(path: Path, payload: dict) -> None:
    """Never leave a half-written manifest: a reader that catches a torn
    JSON file cannot tell it from corruption."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def code_provenance(root: Path | str = ".") -> dict:
    """Which code actually ran: commit, branch, and whether the tree was
    clean. `dirty` matters as much as the hash -- a run from a dirty tree is
    attributable to no commit at all, and saying so honestly beats recording
    a hash that reviewers will wrongly trust."""

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "branch": None, "dirty": None}
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def start_manifest(path: Path | str, **sections: Any) -> dict:
    """Freeze the run's identity before any work happens.

    Called again on the same path -- a resume -- it does not rewrite the
    frozen sections: the original identity stands, and the resume is recorded
    as an event with its own provenance, because the code may have changed
    between the crash and the restart and that difference is precisely what a
    post-mortem will want.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        manifest = json.loads(path.read_text())
        manifest.setdefault("resumes", []).append(
            {"at": time.time(), "code": code_provenance()}
        )
        manifest["status"] = "running"
        _atomic_write(path, manifest)
        return manifest
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": time.time(),
        "status": "running",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "code": code_provenance(),
    }
    for key, value in sections.items():
        manifest[key] = _jsonable(value)
    _atomic_write(path, manifest)
    return manifest


def finalize_manifest(path: Path | str, **sections: Any) -> dict:
    """Append the outcome without touching the frozen identity."""
    path = Path(path)
    manifest = json.loads(path.read_text()) if path.exists() else {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": time.time(),
    }
    for key, value in sections.items():
        manifest[key] = _jsonable(value)
    manifest["status"] = "completed"
    manifest["finished_at"] = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, manifest)
    return manifest


def write_manifest(path: Path | str, **sections: Any) -> dict:
    """Single-shot manifest (legacy callers and simple tools). New runs
    should freeze identity with start_manifest and close with
    finalize_manifest instead."""
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": time.time(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    for key, value in sections.items():
        manifest[key] = _jsonable(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(Path(path), manifest)
    return manifest
