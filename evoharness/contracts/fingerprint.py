"""Deterministic JSON identities for frozen public specifications."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any


class FingerprintError(ValueError):
    """A spec contains a value without stable JSON semantics."""


def canonical_payload(value: Any) -> Any:
    to_payload = getattr(value, "to_payload", None)
    if callable(to_payload):
        return canonical_payload(to_payload())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return canonical_payload(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return canonical_payload(value.value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FingerprintError("non-finite floats are not valid spec values")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise FingerprintError("spec mapping keys must be strings")
            normalized[key] = canonical_payload(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [canonical_payload(item) for item in value]
    raise FingerprintError(
        f"{type(value).__name__} has no stable spec representation"
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonical_payload(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def spec_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_object_json(value: Mapping[str, Any] | None = None) -> str:
    normalized = canonical_payload({} if value is None else value)
    if not isinstance(normalized, dict):
        raise FingerprintError("component config must be a JSON object")
    return canonical_json(normalized)
