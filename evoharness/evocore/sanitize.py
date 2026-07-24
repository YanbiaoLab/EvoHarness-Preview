"""Egress sanitizers for cold trace items(安全边界,fail-closed)。

冷 store 里可能含敏感产物;流向不同 audience(optimizer/human)前必须
过滤。默认全删,只放行白名单 key —— 未知 audience 得到空 dict。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class TraceSanitizer(Protocol):
    def sanitize_item(self, item: dict, audience: str) -> dict: ...
    def sanitize_summary(self, summary: dict, audience: str) -> dict: ...


@dataclass(frozen=True)
class AllowlistSanitizer:
    """按 audience 白名单放行 key;未知 audience → 全删(fail-closed)。"""

    item_allow: dict[str, frozenset[str]]
    summary_allow: dict[str, frozenset[str]]

    def sanitize_item(self, item: dict, audience: str) -> dict:
        allowed = self.item_allow.get(audience, frozenset())
        return {k: v for k, v in item.items() if k in allowed}

    def sanitize_summary(self, summary: dict, audience: str) -> dict:
        allowed = self.summary_allow.get(audience, frozenset())
        return {k: v for k, v in summary.items() if k in allowed}


__all__ = ["AllowlistSanitizer", "TraceSanitizer"]
