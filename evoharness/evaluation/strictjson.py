"""Generic strict-JSON helpers: the package's only true leaf module.

这些助手不认识 Evidence,也不 import 包内任何东西——严格标量校验与
深度冻结/解冻是通用能力。领域不变量在 evidence.py,不在这里。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def strict_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def optional_bool(value: object, name: str) -> bool | None:
    if value is None:
        return None
    return strict_bool(value, name)


def optional_finite_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise TypeError(f"{name} must be a finite number or None")
    return float(value)


def freeze_json(value: Any, path: str) -> Any:
    """Validate a JSON value and recursively make containers immutable."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            frozen[key] = freeze_json(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            freeze_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} contains non-JSON value {type(value).__name__}")


def thaw_json(value: Any) -> Any:
    """Return ordinary JSON containers from a deeply frozen value."""

    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


__all__ = [
    "freeze_json",
    "optional_bool",
    "optional_finite_float",
    "strict_bool",
    "strict_int",
    "thaw_json",
]
