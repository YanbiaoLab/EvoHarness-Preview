# EvoHarness original: run manifests for honest, reproducible reporting
# (plan sections 4/7: manifests, holdout hash, cost accounting).
"""Run manifest writer."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = 1


def _jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return obj


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(path: Path | str, **sections: Any) -> dict:
    """Write a manifest JSON; each keyword becomes a section (dataclasses are
    serialized). Returns the manifest dict."""
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
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest
