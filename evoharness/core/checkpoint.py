# EvoHarness original (verl-inspired checkpoint/resume discipline; plan
# section 7 requires graceful budget stops with resumable state).
# Intentional deviation from upstream: JSON everywhere, never pickle —
# human-readable, diffable, no deserialization hazards.
"""Checkpoint helpers: atomic JSON writes and config fingerprinting.

The checkpoint FORMAT is a run-directory convention, not a binary file:
run.db (PopulationStore, the primary state) + append-only JSONL logs
(metrics/experience) + budget.json + checkpoint.json (loop counters, RNG
state, component states). Resume requires a file-backed PopulationStore.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

CHECKPOINT_SCHEMA_VERSION = 1


def atomic_write_json(path: Path, data: dict) -> None:
    """Write via temp file + rename so a crash never leaves a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def append_jsonl(path: Path, payload: dict) -> None:
    """Append-only JSONL: one fsynced line per call.

    The other half of the run-dir persistence discipline: atomic_write_json
    replaces whole documents; this appends facts that must never be
    rewritten (evidence, decisions, reference history, inbox records).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def config_fingerprint(*configs: Any) -> str:
    """Stable hash over effective configuration and assembly descriptors.

    Dataclasses cover ordinary configuration objects; JSON-compatible
    mappings let recipes include their effective plugin stack without
    introducing another checkpoint-only model.
    """
    payload = [
        dataclasses.asdict(config)
        if dataclasses.is_dataclass(config) and not isinstance(config, type)
        else config
        for config in configs
    ]
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
