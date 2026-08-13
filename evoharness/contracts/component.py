"""Serializable identities for runtime components."""

from __future__ import annotations

import json
import inspect
import hashlib
from dataclasses import dataclass
from typing import Any

from .fingerprint import canonical_object_json, spec_hash


def qualified_name(value: object) -> str:
    if inspect.isfunction(value) or inspect.ismethod(value):
        return f"{value.__module__}.{value.__qualname__}"
    cls = value if isinstance(value, type) else value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def implementation_sha256(value: object) -> str | None:
    """Best-effort source identity; explicit versions remain authoritative."""

    target = value
    if not (inspect.isfunction(value) or inspect.ismethod(value)):
        target = value if isinstance(value, type) else value.__class__
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        return None
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ComponentSpec:
    role: str
    name: str
    version: str = "1"
    config_json: str = "{}"

    def __post_init__(self) -> None:
        for field_name in ("role", "name", "version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        try:
            config = json.loads(self.config_json)
        except json.JSONDecodeError as exc:
            raise ValueError("config_json must be valid JSON") from exc
        if not isinstance(config, dict):
            raise ValueError("config_json must encode a JSON object")
        object.__setattr__(self, "config_json", canonical_object_json(config))

    @classmethod
    def create(
        cls,
        role: str,
        name: str,
        *,
        version: str = "1",
        config: dict[str, Any] | None = None,
    ) -> "ComponentSpec":
        return cls(role, name, version, canonical_object_json(config))

    @classmethod
    def for_object(
        cls,
        role: str,
        value: object,
        *,
        version: str,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> "ComponentSpec":
        resolved_config = dict(config or {})
        source_hash = implementation_sha256(value)
        if source_hash is not None:
            resolved_config.setdefault("implementation_sha256", source_hash)
        return cls.create(
            role,
            name or qualified_name(value),
            version=version,
            config=resolved_config,
        )

    def require_role(self, expected: str) -> None:
        if self.role != expected:
            raise ValueError(
                f"component {self.name!r} has role {self.role!r}; "
                f"expected {expected!r}"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "name": self.name,
            "version": self.version,
            "config": json.loads(self.config_json),
        }

    @property
    def hash(self) -> str:
        return spec_hash(self.to_payload())
