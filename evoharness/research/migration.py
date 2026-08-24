"""Measurement Migration Panel

"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

from evoharness.contracts import spec_hash
from evoharness.evaluation import EvidenceEnvelope, ScoreNamespace

from .inbox import InboxError, InboxStore
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
    champion_old_fitness: float
    champion_new_fitness: float
    champion_old_evidence_id: str
    champion_new_evidence_id: str
    champion_changed: bool
    top_k: int
    top_k_overlap: float
    decision_flips: int             # "能否胜过旧冠军"的判定翻转数
    flipped_candidates: tuple[str, ...]
    pairwise_inversions: int
    inverted_pairs_sample: tuple[tuple[str, str], ...]
    cost_old_usd: float
    cost_new_usd: float
    evidence_refs: tuple[str, ...]
    requires_human_approval: bool = True

    @property
    def hash(self) -> str:
        return spec_hash({"kind": "measurement_migration_report", **self.to_json()})

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["old_namespace"] = self.old_namespace.to_json()
        payload["new_namespace"] = self.new_namespace.to_json()
        payload["flipped_candidates"] = list(self.flipped_candidates)
        payload["inverted_pairs_sample"] = [
            list(pair) for pair in self.inverted_pairs_sample
        ]
        payload["evidence_refs"] = list(self.evidence_refs)
        return {"schema_version": 1, **payload}


def _score_map(
    envelopes: list[EvidenceEnvelope], label: str
) -> tuple[
    ScoreNamespace,
    dict[str, float],
    dict[str, str],
    float,
    tuple[str, ...],
]:
    if not envelopes:
        raise MigrationPanelError(f"{label} panel is empty")
    namespaces = {env.namespace for env in envelopes}
    if len(namespaces) != 1:
        raise MigrationPanelError(
            f"{label} panel mixes {len(namespaces)} namespaces"
        )
    scores: dict[str, float] = {}
    evidence_by_candidate: dict[str, str] = {}
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
        evidence_by_candidate[env.candidate_id] = env.evidence_id
        cost += env.budget_used_usd or 0.0
    evidence_refs = tuple(sorted({env.evidence_id for env in envelopes}))
    return namespaces.pop(), scores, evidence_by_candidate, cost, evidence_refs


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i_: values[i_])
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
    old_ns, old_scores, old_evidence, cost_old, old_refs = _score_map(old, "old")
    new_ns, new_scores, new_evidence, cost_new, new_refs = _score_map(new, "new")
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
        champion_old_fitness=old_scores[champion_old],
        champion_new_fitness=new_scores[champion_new],
        champion_old_evidence_id=old_evidence[champion_old],
        champion_new_evidence_id=new_evidence[champion_new],
        champion_changed=champion_old != champion_new,
        top_k=k,
        top_k_overlap=overlap,
        decision_flips=len(flipped),
        flipped_candidates=flipped,
        pairwise_inversions=len(inversions),
        inverted_pairs_sample=tuple(inversions[:5]),
        cost_old_usd=cost_old,
        cost_new_usd=cost_new,
        evidence_refs=tuple(dict.fromkeys((*old_refs, *new_refs))),
    )


def establish_baseline(
    store: ReferenceStore,
    record: ReferenceRecord,
    *,
    report: MigrationReport,
    inbox: InboxStore,
    request_id: str,
    experiment_id: str,
    now=time.time,
) -> ReferenceRecord:
    """Establish a new-namespace baseline from one bound Inbox approval."""
    try:
        request, decision = inbox.require_decision(
            request_id,
            kind="protocol-change",
            action="approve-protocol-change",
            experiment_id=experiment_id,
            subject_hash=report.hash,
        )
    except InboxError as exc:
        raise MigrationApprovalRequired(str(exc)) from exc
    if record.namespace != report.new_namespace:
        raise MigrationPanelError(
            "record namespace does not match the migration report"
        )
    if (
        record.candidate_id != report.champion_new
        or record.evidence_id != report.champion_new_evidence_id
        or record.fitness != report.champion_new_fitness
    ):
        raise MigrationPanelError(
            "new baseline record does not match the approved panel champion"
        )
    current = store.current()
    if current is None:
        raise MigrationPanelError("measurement migration requires an old baseline")
    if current.namespace == record.namespace:
        raise MigrationPanelError(
            "same namespace — use PromotionPolicy/promote()"
        )
    if current.namespace != report.old_namespace:
        raise MigrationPanelError(
            "current reference is not on the report's old namespace"
        )
    if (
        current.candidate_id != report.champion_old
        or current.evidence_id != report.champion_old_evidence_id
        or current.fitness != report.champion_old_fitness
    ):
        raise MigrationPanelError(
            "current reference does not match the migration panel champion"
        )
    return store.rebase(
        record,
        migration_decision={
            "experiment_id": decision.experiment_id,
            "request_id": request.request_id,
            "migration_report_hash": report.hash,
            "action": decision.action,
            "actor": decision.actor,
            "reason": decision.reason,
            "created_at": decision.created_at,
            "report": report.to_json(),
        },
        now=now,
    )
