"""Load, materialize, and hash the IMO solver seed files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .protocol import CandidateSpec


def seed_directory() -> Path:
    return Path(__file__).with_name("seed_agent")


def load_seed_files(candidate: CandidateSpec) -> dict[str, str]:
    root = seed_directory()
    files = {}
    for rel in candidate.mutable_files:
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"seed candidate file missing: {path}")
        files[rel] = path.read_text(encoding="utf-8")
    return files


def materialize_seed(candidate: CandidateSpec, destination: Path) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for rel, content in load_seed_files(candidate).items():
        path = destination / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return destination


def seed_sha256(candidate: CandidateSpec) -> str:
    payload = json.dumps(
        load_seed_files(candidate),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()
