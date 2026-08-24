import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from evoharness.contracts import spec_hash
from evoharness.evaluation import EvidenceEnvelope

from .assessment import AssessmentGuard, ClaimAssessment, Verdict
from .claims import Claim, ClaimKind
from .gate import AutoExecutionGate, requires_human
from .inbox import DecisionRequest, InboxStore
from .migration import MigrationReport
from .models import ExperimentOutcome
from .reference import ReferenceRecord

# 需要人的事件各自绑定一种卡:事件与卡型对不上就是接线错误,不是可以
# 将就的小事——错卡意味着错的动作集合。
EVENT_CARD_KIND = {
    "announce-breakthrough": "breakthrough",
    "expand-budget": "budget-expansion",
    "change-measurement-protocol": "protocol-change",
    "accept-ranking-flip": "ranking-flip",
}


class RoutingError(RuntimeError):
    """事件无法被路由:该要人却没有卡,或卡型与事件不符。"""


@dataclass(frozen=True)
class Notification:
    """通知但不阻塞:新失败类别、证据冲突、成本异常。"""

    kind: str
    message: str
    refs: tuple[str, ...] = ()
    created_at: float = 0.0

def conflict_notification(
    a: ClaimAssessment, b: ClaimAssessment, *, created_at: float
) -> Notification:
    if a.claim_hash != b.claim_hash:
        raise ValueError("conflict requires two assessments of one claim")
    if {a.status, b.status} != {Verdict.SUPPORTED, Verdict.CONTRADICTED}:
        raise ValueError("conflict means supported vs contradicted")
    return Notification(
        kind="conflicting-evidence",
        message=(
            f"claim {a.claim_kind} has both supported and contradicted "
            f"assessments (assessors {a.assessor_hash} / {b.assessor_hash})"
        ),
        refs=tuple(a.evidence_refs) + tuple(b.evidence_refs),
        created_at=created_at,
    )

def protocol_change_request(
        report: MigrationReport,
        *,
        experiment_id: str,
        created_at: float,
        estimated_costs: tuple[tuple[str, float], ...] = (),

) -> DecisionRequest:
    risky = report.champion_changed or report.decision_flips > 0
    return DecisionRequest(
        kind="protocol-change",
        experiment_id=experiment_id,
        question=(
            "Measurement 迁移待审:"
            f"{report.old_namespace.measurement_hash} → "
            f"{report.new_namespace.measurement_hash}。是否批准新 baseline?"
        ),
        alternatives=(
            "approve-protocol-change:采纳新口径,建立新 baseline",
            "request-more-evidence:扩大面板重测",
            "veto:维持旧口径",
        ),
        recommended_action=(
            "request-more-evidence" if risky else "approve-protocol-change"
        ),
        evidence_refs=report.evidence_refs,
        uncertainty=(
            f"spearman={report.spearman}, champion_changed="
            f"{report.champion_changed}, flips={report.decision_flips}, "
            f"inversions={report.pairwise_inversions}"
        ),
        consequence_of_waiting=(
            "旧口径继续作为 baseline;新测量产生的分数处于新 namespace,"
            "在批准前不可与任何历史分数比较"
        ),
        estimated_costs=estimated_costs,
        subject_hash=report.hash,
        payload=json.dumps(report.to_json(), ensure_ascii=False),
        created_at=created_at,
    )


def breakthrough_request(
    assessment: ClaimAssessment,
    *,
    experiment_id: str,
    candidate_id: str,
    created_at: float,
    estimated_costs: tuple[tuple[str, float], ...] = (),
) -> DecisionRequest:
    if assessment.claim_kind != "objective_met":
        raise ValueError("breakthrough card requires an objective_met claim")
    if assessment.status is not Verdict.SUPPORTED:
        raise ValueError(
            "breakthrough card requires a SUPPORTED assessment — "
            "an unsupported claim has nothing to announce"
        )

    return DecisionRequest(
        kind="breakthrough",
        experiment_id=experiment_id,
        question=(
            f"候选 {candidate_id} 的 objective_met 获独立 Verifier 支持。"
            "是否确认为 Breakthrough?"
        ),
        alternatives=(
            "approve:确认突破,进入对外披露流程",
            "request-more-evidence:要求复现/更大验证",
            "veto:不认定",
        ),
        recommended_action="request-more-evidence",
        evidence_refs=tuple(assessment.evidence_refs),
        uncertainty="; ".join(assessment.reasons),
        consequence_of_waiting="不对外宣布;候选保持 reference 候选资格",
        estimated_costs=estimated_costs,
        subject_hash=assessment.claim_hash,
        payload=json.dumps(assessment.to_json(), ensure_ascii=False),
        created_at=created_at,
    )


def budget_expansion_request(
    *,
    experiment_id: str,
    current_budget_usd: float,
    requested_budget_usd: float,
    justification: str,
    created_at: float,
) -> DecisionRequest:
    if not justification.strip():
        raise ValueError("budget expansion requires a justification")
    if (
        not math.isfinite(current_budget_usd)
        or not math.isfinite(requested_budget_usd)
        or current_budget_usd < 0
        or requested_budget_usd <= current_budget_usd
    ):
        raise ValueError("expansion must request more than current budget")
    budget_subject = {
        "experiment_id": experiment_id,
        "current_budget_usd": current_budget_usd,
        "requested_budget_usd": requested_budget_usd,
        "justification": justification,
    }
    return DecisionRequest(
        kind="budget-expansion",
        experiment_id=experiment_id,
        question=(
            f"实验 {experiment_id} 申请预算 "
            f"{current_budget_usd:.2f} → {requested_budget_usd:.2f} USD:"
            f"{justification}"
        ),
        alternatives=(
            "approve:批准扩容",
            "revise:改预算另议",
            "veto:维持原预算",
        ),
        recommended_action="veto",
        consequence_of_waiting="实验在原预算内继续,耗尽即停",
        estimated_costs=(
            ("approve", requested_budget_usd - current_budget_usd),
        ),
        subject_hash=spec_hash({"kind": "budget_expansion", **budget_subject}),
        payload=json.dumps(budget_subject, ensure_ascii=False),
        created_at=created_at,
    )


def ranking_flip_request(
    *,
    experiment_id: str,
    previous: ReferenceRecord,
    promoted: ReferenceRecord,
    created_at: float,
    reasons: tuple[str, ...] = (),
    evidence_refs: tuple[str, ...] = (),
    estimated_costs: tuple[tuple[str, float], ...] = (),
) -> DecisionRequest:
    if previous.candidate_id == promoted.candidate_id:
        raise ValueError("a ranking flip requires the champion to change")
    previous.namespace.require_comparable(promoted.namespace)
    flip_subject = {
        "experiment_id": experiment_id,
        "previous_candidate_id": previous.candidate_id,
        "previous_evidence_id": previous.evidence_id,
        "previous_fitness": previous.fitness,
        "promoted_candidate_id": promoted.candidate_id,
        "promoted_evidence_id": promoted.evidence_id,
        "promoted_fitness": promoted.fitness,
        "namespace": promoted.namespace.to_json(),
        "policy_hash": promoted.policy_hash,
        "assessor_hash": promoted.assessor_hash,
    }
    return DecisionRequest(
        kind="ranking-flip",
        experiment_id=experiment_id,
        question=(
            f"参照物已换人:{previous.candidate_id} → {promoted.candidate_id}"
            f"(fitness {previous.fitness} → {promoted.fitness})。"
            "新冠军是否留任?"
        ),
        alternatives=(
            "approve:接受新冠军",
            "request-more-evidence:先复现再定",
            "veto:回滚到旧冠军",
        ),
        recommended_action="request-more-evidence",
        evidence_refs=tuple(
            dict.fromkeys(
                (previous.evidence_id, promoted.evidence_id, *evidence_refs)
            )
        ),
        uncertainty="; ".join(reasons or promoted.reasons),
        consequence_of_waiting=(
            "新冠军已经是现任参照物,后续候选都与它比较;不决定等于默许"
        ),
        estimated_costs=estimated_costs,
        subject_hash=spec_hash({"kind": "ranking_flip", **flip_subject}),
        payload=json.dumps(flip_subject, ensure_ascii=False),
        created_at=created_at,
    )


@dataclass(frozen=True)
class RoutedEvent:
    """一次路由的结果:自动执行、提交卡片,还是发出通知。"""

    event_kind: str
    experiment_id: str
    disposition: str            # auto | card | notification
    detail: str = ""
    request_id: str = ""
    # 事件源自哪次 run。活性审计按它回答"这次 run 有没有被路由过",
    # 靠 detail 里的字符串去猜是靠不住的。晋升之类跨 run 的事件留空。
    run_id: str = ""
    refs: tuple[str, ...] = ()
    created_at: float = 0.0

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["refs"] = list(self.refs)
        return {"schema_version": 1, **payload}


class ResearchRouter:
    """把 Assessment/Runner 的事件送进 Inbox,或按白名单自动执行。

    路由器不裁决、不执行动作效果,只保证每个事件都有去处并留下痕迹:
    白名单内的自动执行也写进 routing.jsonl,否则"没提交任何卡"和
    "根本没接线"在事后看起来一模一样。

    提交卡片不阻塞 run——运行照常结束,决定可以隔天再做。
    """

    def __init__(
        self,
        inbox: InboxStore,
        *,
        gate: AutoExecutionGate | None = None,
        runs_root=None,
        now=time.time,
    ):
        self._inbox = inbox
        self._store = inbox.research_store
        self._gate = gate or AutoExecutionGate(
            self._store, inbox, runs_root=runs_root
        )
        self._now = now

    @property
    def inbox(self) -> InboxStore:
        return self._inbox

    @property
    def gate(self) -> AutoExecutionGate:
        return self._gate

    def route(
        self,
        event_kind: str,
        *,
        experiment_id: str,
        card: DecisionRequest | None = None,
        detail: str = "",
        run_id: str = "",
        projected_cost_usd: float = 0.0,
    ) -> RoutedEvent:
        if not requires_human(event_kind):
            if card is not None:
                raise RoutingError(
                    f"{event_kind} is auto-executable — a card would imply "
                    "an approval that nobody is going to give"
                )
            # 白名单只说"原则上不必惊动人";门说此时此地行不行,不行就炸,
            # 不静默降级——降级过的自动执行事后与从未发生无从分辨。
            gated = self._gate.check(
                event_kind,
                experiment_id=experiment_id,
                projected_cost_usd=projected_cost_usd,
            ).require()
            return self._record(RoutedEvent(
                event_kind=event_kind,
                experiment_id=experiment_id,
                disposition="auto",
                detail=detail or "; ".join(gated.reasons),
                run_id=run_id,
                created_at=self._now(),
            ))

        if card is None:
            raise RoutingError(
                f"{event_kind} needs a human decision but no card was built"
            )
        expected = EVENT_CARD_KIND.get(event_kind, "generic")
        if card.kind != expected:
            raise RoutingError(
                f"{event_kind} must carry a {expected!r} card, got {card.kind!r}"
            )
        if card.experiment_id != experiment_id:
            raise RoutingError("card belongs to a different experiment")
        request_id = self._inbox.submit(card)
        return self._record(RoutedEvent(
            event_kind=event_kind,
            experiment_id=experiment_id,
            disposition="card",
            detail=detail or card.question,
            request_id=request_id,
            run_id=run_id,
            refs=card.evidence_refs,
            created_at=self._now(),
        ))

    def notify(
        self,
        notification: Notification,
        *,
        experiment_id: str,
        run_id: str = "",
    ) -> RoutedEvent:
        """通知但不阻塞:进审计流,不进待办队列。"""
        return self._record(RoutedEvent(
            event_kind=notification.kind,
            experiment_id=experiment_id,
            disposition="notification",
            detail=notification.message,
            run_id=run_id,
            refs=notification.refs,
            created_at=notification.created_at or self._now(),
        ))

    def _record(self, event: RoutedEvent) -> RoutedEvent:
        self._store.append_routed_event(event)
        return event

    # -- typed entry points ------------------------------------------------

    def route_assessment(
        self,
        assessment: ClaimAssessment,
        *,
        experiment_id: str,
        candidate_id: str,
        run_id: str = "",
        estimated_costs: tuple[tuple[str, float], ...] = (),
    ) -> tuple[RoutedEvent, ...]:
        events: list[RoutedEvent] = []
        prior = self._store.assessments(experiment_id)
        for old in prior:
            if old.claim_hash != assessment.claim_hash:
                continue
            if {old.status, assessment.status} == {
                Verdict.SUPPORTED, Verdict.CONTRADICTED
            }:
                events.append(self.notify(
                    conflict_notification(
                        old, assessment, created_at=self._now()
                    ),
                    experiment_id=experiment_id,
                    run_id=run_id,
                ))
                break
        if not any(
            old.claim_hash == assessment.claim_hash
            and old.assessor_hash == assessment.assessor_hash
            and old.status is assessment.status
            for old in prior
        ):
            self._store.append_assessment(experiment_id, assessment)

        if (
            assessment.claim_kind == ClaimKind.OBJECTIVE_MET.value
            and assessment.status is Verdict.SUPPORTED
        ):
            events.append(self.route(
                "announce-breakthrough",
                experiment_id=experiment_id,
                run_id=run_id,
                card=breakthrough_request(
                    assessment,
                    experiment_id=experiment_id,
                    candidate_id=candidate_id,
                    created_at=self._now(),
                    estimated_costs=estimated_costs,
                ),
            ))
        return tuple(events)

    def route_promotion(
        self,
        *,
        experiment_id: str,
        previous: ReferenceRecord | None,
        promoted: ReferenceRecord,
        reasons: tuple[str, ...] = (),
    ) -> tuple[RoutedEvent, ...]:
        # 第一个参照物不是翻转,是从无到有——没有排名可以推翻。
        if previous is None or previous.candidate_id == promoted.candidate_id:
            return ()
        return (self.route(
            "accept-ranking-flip",
            experiment_id=experiment_id,
            card=ranking_flip_request(
                experiment_id=experiment_id,
                previous=previous,
                promoted=promoted,
                reasons=reasons,
                created_at=self._now(),
            ),
        ),)

    def route_run(
        self,
        outcome: ExperimentOutcome,
        *,
        run_dir: Path,
        budget_usd: float | None = None,
        guard: AssessmentGuard | None = None,
    ) -> tuple[RoutedEvent, ...]:
        """Runner 事件流的路由入口:一次 run 结束后走完全部出口。"""
        experiment_id = outcome.experiment_id
        events: list[RoutedEvent] = [self.route(
            "record-outcome",
            experiment_id=experiment_id,
            run_id=outcome.run_id,
            detail=(
                f"run {outcome.run_id} stopped_reason={outcome.stopped_reason} "
                f"generations={outcome.generations_completed}/"
                f"{outcome.generations_planned} "
                f"eval_cost={outcome.eval_cost_usd:.4f} USD"
            ),
        )]

        card = budget_expansion_card_for(outcome, budget_usd=budget_usd)
        if card is not None:
            events.append(self.route(
                "expand-budget", experiment_id=experiment_id,
                run_id=outcome.run_id, card=card,
            ))

        guard = guard or AssessmentGuard()
        for envelope in read_run_evidence(run_dir):
            # 只评没有 verifier 判决就不该被评的东西:objective_met 为 None
            # 意味着无人裁定,评它只会得到一串 unknown 噪声。
            if envelope.objective_met is None:
                continue
            events.extend(self.route_assessment(
                guard.assess(
                    Claim(ClaimKind.OBJECTIVE_MET, envelope.candidate_id),
                    envelope,
                ),
                experiment_id=experiment_id,
                candidate_id=envelope.candidate_id,
                run_id=outcome.run_id,
            ))
        return tuple(events)


def read_run_evidence(run_dir: Path) -> list[EvidenceEnvelope]:
    path = Path(run_dir) / "evidence.jsonl"
    if not path.exists():
        return []
    return [
        EvidenceEnvelope.from_json(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def budget_expansion_card_for(
    outcome: ExperimentOutcome, *, budget_usd: float | None
) -> DecisionRequest | None:
    """预算耗尽而实验未跑完时的扩容卡;其余情况没有可问的问题。"""
    if budget_usd is None or budget_usd <= 0:
        return None
    if outcome.stopped_reason != "budget":
        return None
    remaining = outcome.generations_planned - outcome.generations_completed
    if remaining <= 0:
        return None
    completed = outcome.generations_completed
    extra = (outcome.eval_cost_usd / completed * remaining) if completed else 0.0
    basis = (
        f"已完成 {completed} 代花费 {outcome.eval_cost_usd:.4f} USD,"
        f"按同样速率跑完剩余 {remaining} 代需追加 {extra:.4f} USD"
    )
    if extra <= 0:
        # 一代都没跑完就烧光,没有速率可外推——按当前预算翻倍并说明。
        extra = budget_usd
        basis = (
            f"预算在完成第一代前耗尽(已花 {outcome.eval_cost_usd:.4f} USD),"
            "无速率可外推,按当前预算翻倍申请"
        )
    return budget_expansion_request(
        experiment_id=outcome.experiment_id,
        current_budget_usd=budget_usd,
        requested_budget_usd=budget_usd + extra,
        justification=(
            f"run {outcome.run_id} 因预算停在 "
            f"{outcome.generations_completed}/{outcome.generations_planned} 代;"
            f"{basis}"
        ),
        created_at=outcome.created_at,
    )
