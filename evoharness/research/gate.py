"""白名单说"哪些动作原则上不必惊动人",执行门说"此时此地是否真的可以"。

只有白名单是不够的:漂移的协议、记不全的账、没人批过的预算,都能从一张
纯名单底下走过去。门在放行前把三件事一起验掉,任一条不成立就转人工。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .inbox import InboxStore
from .store import ResearchStore, ResearchStoreError

# 已冻结协议、预先批准预算内的低风险动作。仅此五类可以不经人工。
AUTO_EXECUTABLE = frozenset({
    "re-evaluate-within-approved-budget",
    "repair-within-approved-experiment",
    "quarantine-hard-regression",
    "record-outcome",
    "record-low-risk-finding",
})

# 会花钱的自动动作,额外要一本可信的总账和一个预先批准的上限。
# 记录类与隔离类不在此列:把"隔离已确认的硬退步"压在预算门后面,
# 等于让没钱变成继续用坏参照物的理由。
SPENDS_BUDGET = frozenset({
    "re-evaluate-within-approved-budget",
    "repair-within-approved-experiment",
})


class AutoExecutionDenied(RuntimeError):
    """门不放行:这个事件此时此地必须交给人。"""


def requires_human(event_kind: str) -> bool:
    # 白名单外一律要人:路由器不认识的事件不是默认放行,是默认要人。
    return event_kind not in AUTO_EXECUTABLE


@dataclass(frozen=True)
class BudgetLedger:
    """一个实验花了多少、被批准花多少,以及这两个数字值不值得信。"""

    approved_usd: float
    spent_usd: float
    trusted: bool
    reasons: tuple[str, ...] = ()

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.approved_usd - self.spent_usd)

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return {"schema_version": 1, **payload, "remaining_usd": self.remaining_usd}


@dataclass(frozen=True)
class GateDecision:
    event_kind: str
    experiment_id: str
    allowed: bool
    reasons: tuple[str, ...] = ()
    ledger: BudgetLedger | None = None

    def require(self) -> "GateDecision":
        if not self.allowed:
            raise AutoExecutionDenied(
                f"{self.event_kind} cannot run unattended on "
                f"{self.experiment_id}: " + "; ".join(self.reasons)
            )
        return self

    def to_json(self) -> dict:
        return {
            "schema_version": 1,
            "event_kind": self.event_kind,
            "experiment_id": self.experiment_id,
            "allowed": self.allowed,
            "reasons": list(self.reasons),
            "ledger": self.ledger.to_json() if self.ledger else None,
        }


class AutoExecutionGate:
    """三条前提同时成立才放行:协议已冻结、总账可信、预算预先批准。

    ``runs_root`` 是 run 目录的父目录。总成本只在 BudgetMeter 的
    ``budget.json`` 里(它同时计提案 LLM 花费与评测花费);
    ExperimentOutcome 只记评测成本,拿它当总账是少算,所以读不到
    metered 总额的实验一律判为账本不可信。
    """

    def __init__(
        self,
        store: ResearchStore,
        inbox: InboxStore | None = None,
        *,
        runs_root: Path | str | None = None,
    ):
        self._store = store
        self._inbox = inbox
        self._runs_root = Path(runs_root) if runs_root is not None else None

    def check(
        self,
        event_kind: str,
        *,
        experiment_id: str,
        projected_cost_usd: float = 0.0,
    ) -> GateDecision:
        if requires_human(event_kind):
            return GateDecision(
                event_kind, experiment_id, False,
                ("outside the auto-executable whitelist",),
            )
        try:
            self._store.load_experiment(experiment_id)
        except (ResearchStoreError, ValueError) as exc:
            # 内容与记录的 spec_hash 对不上也走这里:协议漂移了。
            return GateDecision(
                event_kind, experiment_id, False,
                (f"protocol is not frozen: {exc}",),
            )

        if event_kind not in SPENDS_BUDGET:
            if projected_cost_usd:
                return GateDecision(
                    event_kind, experiment_id, False,
                    (
                        f"{event_kind} is classified as spending nothing but "
                        f"projects {projected_cost_usd} USD — the whitelist "
                        "entry is wrong, not the caller",
                    ),
                )
            return GateDecision(event_kind, experiment_id, True)

        if (
            isinstance(projected_cost_usd, bool)
            or not isinstance(projected_cost_usd, (int, float))
            or not math.isfinite(projected_cost_usd)
            or projected_cost_usd < 0
        ):
            return GateDecision(
                event_kind, experiment_id, False,
                ("projected cost must be a non-negative finite number",),
            )

        ledger = self.budget(experiment_id)
        reasons: list[str] = []
        if not ledger.trusted:
            reasons.extend(ledger.reasons)
        if ledger.approved_usd <= 0:
            reasons.append(
                "no pre-approved budget: freeze an approved cost estimate or "
                "get a budget-expansion card approved"
            )
        elif ledger.spent_usd + projected_cost_usd > ledger.approved_usd:
            reasons.append(
                f"would spend {ledger.spent_usd + projected_cost_usd:.4f} USD "
                f"against an approved {ledger.approved_usd:.4f} USD"
            )
        return GateDecision(
            event_kind, experiment_id, not reasons, tuple(reasons), ledger
        )

    def budget(self, experiment_id: str) -> BudgetLedger:
        reasons: list[str] = []
        approved = self._approved_usd(experiment_id, reasons)
        spent, trusted = self._spent_usd(experiment_id, reasons)
        return BudgetLedger(
            approved_usd=approved,
            spent_usd=spent,
            trusted=trusted,
            reasons=tuple(reasons),
        )

    def _approved_usd(self, experiment_id: str, reasons: list[str]) -> float:
        spec = self._store.load_experiment(experiment_id)
        approved = 0.0
        if (spec.approved_by or "").strip() and spec.estimated_cost_usd:
            approved = float(spec.estimated_cost_usd)
        else:
            reasons.append(
                "frozen spec carries no approved cost estimate"
            )
        # 每张扩容卡写的是绝对上限,批准过的最高一张就是当前天花板。
        for request, decision in self._approved_expansions(experiment_id):
            del decision
            try:
                requested = float(
                    json.loads(request.payload)["requested_budget_usd"]
                )
            except (KeyError, TypeError, ValueError):
                reasons.append(
                    f"approved expansion {request.request_id} has no readable "
                    "requested amount"
                )
                continue
            approved = max(approved, requested)
        return approved

    def _approved_expansions(self, experiment_id: str):
        if self._inbox is None:
            return []
        return [
            (request, decision)
            for request, decision in self._inbox.answered(experiment_id)
            if request.kind == "budget-expansion" and decision.action == "approve"
        ]

    def _spent_usd(
        self, experiment_id: str, reasons: list[str]
    ) -> tuple[float, bool]:
        try:
            outcomes = self._store.outcomes(experiment_id)
        except (TypeError, ValueError) as exc:
            reasons.append(f"outcome ledger is unreadable: {exc}")
            return 0.0, False

        spent = 0.0
        trusted = True
        if self._runs_root is None and outcomes:
            reasons.append(
                "gate has no runs_root — metered totals cannot be located"
            )
            trusted = False
        for outcome in outcomes:
            metered = self._metered_total(outcome.run_id)
            if metered is None:
                # 只有评测成本可查,提案 LLM 花费无从对账——少算的账不是账。
                if self._runs_root is not None:
                    reasons.append(
                        f"run {outcome.run_id} has no metered total; only its "
                        "evaluation cost is recorded"
                    )
                trusted = False
                spent += max(0.0, outcome.eval_cost_usd)
                continue
            spent += metered
        return spent, trusted

    def _metered_total(self, run_id: str) -> float | None:
        if self._runs_root is None:
            return None
        path = self._runs_root / run_id / "budget.json"
        if not path.exists():
            return None
        try:
            spent = float(json.loads(path.read_text())["spent_usd"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if not math.isfinite(spent) or spent < 0:
            return None
        return spent
