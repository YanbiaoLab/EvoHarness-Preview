"""Measurement Migration Panel(设计文档契约 C4 的执行件)。

Measurement / Evaluator 变更前,用同一批候选在新旧口径下的证据面板量化
迁移会改变哪些决策。报告永远 requires_human_approval——全局秩相关再高
也不自动放行:冠军互换、阈值附近翻转、关键 pairwise 逆序正是相关系数
会隐藏的东西。

面板只能由携带 namespace 的证据构建(I-2 之前的历史 run 没有 namespace
元数据,无法回填入面板)。这是本模块与 EvidenceEnvelope 的硬耦合,
也是刻意的:没有身份的分数不配参与迁移决策。
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

from evoharness.evaluation import EvidenceEnvelope, ScoreNamespace

from .models import ResearchDecision
from .reference import ReferenceRecord, ReferenceStore


class MigrationPanelError(ValueError):
    """面板输入不合法:namespace 混杂、候选不重叠或证据不可信。"""


class MigrationApprovalRequired(RuntimeError):
    """换 namespace 的 baseline 只能带着 approve-protocol-change 决定建立。"""


@dataclass(frozen=True)
class MigrationReport:
    old_namespace: ScoreNamespace
    new_namespace: ScoreNamespace
    n_pairs: int
    spearman: float | None          # None = 样本不足或秩退化
    champion_old: str
    champion_new: str
    champion_changed: bool
    top_k: int
    top_k_overlap: float
    decision_flips: int             # "能否胜过旧冠军"的判定翻转数
    flipped_candidates: tuple[str, ...]
    pairwise_inversions: int
    inverted_pairs_sample: tuple[tuple[str, str], ...]
    cost_old_usd: float
    cost_new_usd: float
    requires_human_approval: bool = True

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["old_namespace"] = self.old_namespace.to_json()
        payload["new_namespace"] = self.new_namespace.to_json()
        payload["flipped_candidates"] = list(self.flipped_candidates)
        payload["inverted_pairs_sample"] = [
            list(pair) for pair in self.inverted_pairs_sample
        ]
        return {"schema_version": 1, **payload}


def _score_map(
    envelopes: list[EvidenceEnvelope], label: str
) -> tuple[ScoreNamespace, dict[str, float], float]:
    if not envelopes:
        raise MigrationPanelError(f"{label} panel is empty")
    namespaces = {env.namespace for env in envelopes}
    if len(namespaces) != 1:
        raise MigrationPanelError(
            f"{label} panel mixes {len(namespaces)} namespaces"
        )
    scores: dict[str, float] = {}
    cost = 0.0
    for env in envelopes:
        if not env.evaluation_valid or env.fitness is None:
            raise MigrationPanelError(
                f"{label} panel contains no-verdict evidence "
                f"({env.candidate_id})"
            )
        if env.admissible is False:
            raise MigrationPanelError(
                f"{label} panel contains inadmissible evidence "
                f"({env.candidate_id})"
            )
        if env.candidate_id in scores:
            raise MigrationPanelError(
                f"{label} panel has duplicate candidate {env.candidate_id}"
            )
        scores[env.candidate_id] = env.fitness
        cost += env.budget_used_usd or 0.0
    return namespaces.pop(), scores, cost


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        result = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2 + 1
            for k in range(i, j + 1):
                result[order[k]] = average
            i = j + 1
        return result

    rx, ry = ranks(xs), ranks(ys)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    var_x = sum((a - mean_x) ** 2 for a in rx)
    var_y = sum((b - mean_y) ** 2 for b in ry)
    if var_x == 0 or var_y == 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def compare_measurements(
    old: list[EvidenceEnvelope],
    new: list[EvidenceEnvelope],
    *,
    top_k: int = 5,
    direction: str = "maximize",
) -> MigrationReport:
    if direction not in {"maximize", "minimize"}:
        raise MigrationPanelError(f"invalid direction: {direction!r}")
    old_ns, old_scores, cost_old = _score_map(old, "old")
    new_ns, new_scores, cost_new = _score_map(new, "new")
    if old_ns == new_ns:
        raise MigrationPanelError(
            "old and new panels share one namespace — nothing migrated; "
            "same-namespace comparison belongs to PromotionPolicy"
        )
    shared = sorted(set(old_scores) & set(new_scores))
    if len(shared) < 2:
        raise MigrationPanelError(
            f"panel needs at least 2 shared candidates, got {len(shared)}"
        )
    sign = 1.0 if direction == "maximize" else -1.0

    def ordering(scores: dict[str, float]) -> list[str]:
        return sorted(shared, key=lambda c: (-sign * scores[c], c))

    order_old = ordering(old_scores)
    order_new = ordering(new_scores)
    champion_old, champion_new = order_old[0], order_new[0]

    k = min(top_k, len(shared))
    overlap = len(set(order_old[:k]) & set(order_new[:k])) / k

    def beats_old_champion(scores: dict[str, float], candidate: str) -> bool:
        return sign * (scores[candidate] - scores[champion_old]) > 0

    flipped = tuple(
        c for c in shared
        if c != champion_old
        and beats_old_champion(old_scores, c)
        != beats_old_champion(new_scores, c)
    )

    inversions = []
    for i, a in enumerate(shared):
        for b in shared[i + 1:]:
            delta_old = sign * (old_scores[a] - old_scores[b])
            delta_new = sign * (new_scores[a] - new_scores[b])
            if delta_old * delta_new < 0:
                inversions.append((a, b))

    return MigrationReport(
        old_namespace=old_ns,
        new_namespace=new_ns,
        n_pairs=len(shared),
        spearman=_spearman(
            [old_scores[c] for c in shared],
            [new_scores[c] for c in shared],
        ),
        champion_old=champion_old,
        champion_new=champion_new,
        champion_changed=champion_old != champion_new,
        top_k=k,
        top_k_overlap=overlap,
        decision_flips=len(flipped),
        flipped_candidates=flipped,
        pairwise_inversions=len(inversions),
        inverted_pairs_sample=tuple(inversions[:5]),
        cost_old_usd=cost_old,
        cost_new_usd=cost_new,
    )


def establish_baseline(
    store: ReferenceStore,
    record: ReferenceRecord,
    *,
    report: MigrationReport,
    decision: ResearchDecision,
    now=time.time,
) -> ReferenceRecord:
    """迁移后的新 baseline:面板报告 + approve-protocol-change 决定,缺一不可。"""
    if decision.action != "approve-protocol-change":
        raise MigrationApprovalRequired(
            "establishing a baseline on a new namespace requires an "
            f"approve-protocol-change decision, got {decision.action!r}"
        )
    if record.namespace != report.new_namespace:
        raise MigrationPanelError(
            "record namespace does not match the migration report"
        )
    current = store.current()
    if current is not None:
        if current.namespace == record.namespace:
            raise MigrationPanelError(
                "same namespace — use PromotionPolicy/promote()"
            )
        if current.namespace != report.old_namespace:
            raise MigrationPanelError(
                "current reference is not on the report's old namespace"
            )
    return store.rebase(
        record,
        migration_decision={
            "experiment_id": decision.experiment_id,
            "action": decision.action,
            "actor": decision.actor,
            "reason": decision.reason,
            "created_at": decision.created_at,
            "report": report.to_json(),
        },
        now=now,
    )
