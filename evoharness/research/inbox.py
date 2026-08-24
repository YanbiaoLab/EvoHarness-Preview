"""Typed, transactional human-decision inbox for the Research Layer.

Requests and their answers share the ResearchStore SQLite ledger.  A
``request_id`` therefore has at most one answer across threads and processes,
and ``pending`` is derived from that single source of truth rather than from a
second status file.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import asdict, dataclass, field

from evoharness.contracts import canonical_json, spec_hash

from .models import DECISION_ACTIONS, ResearchDecision
from .store import ResearchStore


class InboxError(RuntimeError):
    """The request is invalid, unknown, unauthorized, or already answered."""


@dataclass(frozen=True)
class DecisionPolicy:
    allowed_actions: tuple[str, ...]
    default_action: str
    approval_required: bool = True


# These are security policy, not caller-provided card data.  Adding a new
# request kind must make its default and allowed effects explicit here.
DECISION_POLICIES: dict[str, DecisionPolicy] = {
    "protocol-change": DecisionPolicy(
        ("approve-protocol-change", "request-more-evidence", "veto"),
        "veto",
    ),
    "breakthrough": DecisionPolicy(
        ("approve", "request-more-evidence", "veto"),
        "veto",
    ),
    "budget-expansion": DecisionPolicy(
        ("approve", "revise", "veto"),
        "veto",
    ),
    # The flip already happened — promotion is not blocked on a human.  The
    # card asks whether the new champion stays; ``veto`` means roll back.
    "ranking-flip": DecisionPolicy(
        ("approve", "request-more-evidence", "veto"),
        "veto",
    ),
    "generic": DecisionPolicy(
        ("approve", "branch", "veto", "revise", "request-more-evidence"),
        "veto",
    ),
}


@dataclass(frozen=True)
class DecisionRequest:
    kind: str
    experiment_id: str
    question: str
    alternatives: tuple[str, ...]
    recommended_action: str
    evidence_refs: tuple[str, ...] = ()
    uncertainty: str = ""
    consequence_of_waiting: str = ""
    estimated_costs: tuple[tuple[str, float], ...] = ()
    subject_hash: str = ""
    payload: str = "{}"
    created_at: float = 0.0
    allowed_actions: tuple[str, ...] = field(init=False)
    default_action: str = field(init=False)
    approval_required: bool = field(init=False)

    def __post_init__(self) -> None:
        for name in ("kind", "experiment_id", "question"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

        policy = DECISION_POLICIES.get(self.kind)
        if policy is None:
            raise ValueError(
                f"unknown decision kind {self.kind!r}; expected one of "
                f"{sorted(DECISION_POLICIES)}"
            )
        object.__setattr__(self, "allowed_actions", policy.allowed_actions)
        object.__setattr__(self, "default_action", policy.default_action)
        object.__setattr__(self, "approval_required", policy.approval_required)

        if self.recommended_action not in policy.allowed_actions:
            raise ValueError(
                "recommended_action must be allowed by the decision-kind policy"
            )
        if not self.alternatives:
            raise ValueError("alternatives must be non-empty")
        if self.kind != "generic" and not self.subject_hash.strip():
            raise ValueError(f"{self.kind} requests require a subject_hash")
        if any(
            not isinstance(ref, str) or not ref.strip()
            for ref in self.evidence_refs
        ):
            raise ValueError("evidence_refs must contain non-empty strings")

        normalized_costs: list[tuple[str, float]] = []
        seen_cost_actions: set[str] = set()
        for action, raw_cost in self.estimated_costs:
            cost = float(raw_cost)
            if action not in policy.allowed_actions:
                raise ValueError(f"cost supplied for disallowed action {action!r}")
            if action in seen_cost_actions:
                raise ValueError(f"duplicate estimated cost for action {action!r}")
            if cost < 0 or not math.isfinite(cost):
                raise ValueError("estimated costs must be non-negative and finite")
            normalized_costs.append((action, cost))
            seen_cost_actions.add(action)
        object.__setattr__(self, "estimated_costs", tuple(normalized_costs))

        try:
            parsed = json.loads(self.payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("payload must be valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("payload must encode a JSON object")
        object.__setattr__(self, "payload", canonical_json(parsed))

    @property
    def request_id(self) -> str:
        payload = asdict(self)
        payload.pop("created_at")
        return spec_hash({"kind_": "decision_request", **payload})

    @property
    def payload_hash(self) -> str:
        return spec_hash(json.loads(self.payload))

    def to_json(self) -> dict:
        payload = asdict(self)
        # JSON 里没有元组。就地转成 list,免得同一张卡在进程内是元组、
        # 过一趟 HTTP 就变成数组,消费方要写两套判断。
        for name in ("alternatives", "evidence_refs", "allowed_actions"):
            payload[name] = list(payload[name])
        payload["estimated_costs"] = [
            [action, cost] for action, cost in self.estimated_costs
        ]
        return {
            "schema_version": 1,
            "request_id": self.request_id,
            **payload,
        }

    @classmethod
    def from_json(cls, data: dict) -> "DecisionRequest":
        if data.get("schema_version") != 1:
            raise ValueError("DecisionRequest schema_version must be 1")
        derived = {
            name: data.get(name)
            for name in ("allowed_actions", "default_action", "approval_required")
        }
        payload = {
            key: value
            for key, value in data.items()
            if key not in {
                "schema_version",
                "request_id",
                "allowed_actions",
                "default_action",
                "approval_required",
            }
        }
        for name in ("alternatives", "evidence_refs"):
            payload[name] = tuple(payload.get(name, ()))
        payload["estimated_costs"] = tuple(
            (action, float(cost))
            for action, cost in payload.get("estimated_costs", ())
        )
        request = cls(**payload)
        recorded_id = data.get("request_id")
        if recorded_id != request.request_id:
            raise ValueError("DecisionRequest content does not match request_id")
        expected_derived = {
            "allowed_actions": request.allowed_actions,
            "default_action": request.default_action,
            "approval_required": request.approval_required,
        }
        normalized_derived = {
            "allowed_actions": tuple(derived["allowed_actions"] or ()),
            "default_action": derived["default_action"],
            "approval_required": derived["approval_required"],
        }
        if normalized_derived != expected_derived:
            raise ValueError("DecisionRequest policy fields do not match kind")
        return request


class InboxStore:
    """A transactional view over requests and ResearchDecisions.

    ``authorized_actors`` is deliberately explicit.  The UI/service adapter
    must derive this identity from its trusted authentication context; a card
    body is never allowed to nominate its own approver.
    """

    def __init__(
        self,
        research_store: ResearchStore,
        *,
        authorized_actors: frozenset[str] | set[str] | tuple[str, ...],
    ):
        self.research_store = research_store
        self.authorized_actors = frozenset(authorized_actors)
        if not self.authorized_actors or any(
            not isinstance(actor, str) or not actor.strip()
            for actor in self.authorized_actors
        ):
            raise ValueError("authorized_actors must contain non-empty identities")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.research_store.ledger_path,
            timeout=30.0,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def submit(self, request: DecisionRequest) -> str:
        # Human decisions may only attach to an already frozen experiment.
        self.research_store.load_experiment(request.experiment_id)
        encoded = canonical_json(request.to_json())
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT payload_json FROM decision_requests WHERE request_id = ?",
                (request.request_id,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO decision_requests(
                        request_id, experiment_id, payload_json, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        request.request_id,
                        request.experiment_id,
                        encoded,
                        request.created_at,
                    ),
                )
            elif existing["payload_json"] != encoded:
                raise InboxError("request_id collision with different content")
        return request.request_id

    def pending(self) -> list[DecisionRequest]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT r.payload_json
                FROM decision_requests AS r
                LEFT JOIN research_decisions AS d
                  ON d.request_id = r.request_id
                WHERE d.request_id IS NULL
                ORDER BY r.sequence
                """
            ).fetchall()
        return [DecisionRequest.from_json(json.loads(row[0])) for row in rows]

    def answered(
        self, experiment_id: str | None = None
    ) -> list[tuple[DecisionRequest, ResearchDecision]]:
        """Requests that already carry a decision, oldest first."""
        query = """
            SELECT r.payload_json, d.payload_json
            FROM decision_requests AS r
            JOIN research_decisions AS d ON d.request_id = r.request_id
            {where}
            ORDER BY d.sequence
        """
        with self._connect() as connection:
            rows = connection.execute(
                query.format(
                    where="WHERE r.experiment_id = ?" if experiment_id else ""
                ),
                (experiment_id,) if experiment_id else (),
            ).fetchall()
        return [
            (
                DecisionRequest.from_json(json.loads(row[0])),
                ResearchDecision.from_json(json.loads(row[1])),
            )
            for row in rows
        ]

    def get(self, request_id: str) -> DecisionRequest:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM decision_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        if row is None:
            raise InboxError(f"unknown request {request_id}")
        return DecisionRequest.from_json(json.loads(row[0]))

    def decision(self, request_id: str) -> ResearchDecision | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM research_decisions WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        if row is None:
            return None
        return ResearchDecision.from_json(json.loads(row[0]))

    def answer(
        self,
        request_id: str,
        *,
        action: str,
        actor: str,
        reason: str,
        source: str = "unknown",
        now=time.time,
    ) -> ResearchDecision:
        if actor not in self.authorized_actors:
            raise InboxError(f"actor {actor!r} is not authorized for this inbox")

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload_json FROM decision_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if row is None:
                raise InboxError(f"unknown request {request_id}")
            request = DecisionRequest.from_json(json.loads(row[0]))
            if action not in request.allowed_actions:
                raise InboxError(
                    f"action {action!r} not allowed for {request.kind} card "
                    f"(allowed: {list(request.allowed_actions)})"
                )
            if connection.execute(
                "SELECT 1 FROM research_decisions WHERE request_id = ?",
                (request_id,),
            ).fetchone() is not None:
                raise InboxError(f"request {request_id} already answered")

            decision = ResearchDecision(
                experiment_id=request.experiment_id,
                action=action,
                reason=reason,
                actor=actor,
                created_at=now(),
                request_id=request_id,
                source=source,
            )
            connection.execute(
                """
                INSERT INTO research_decisions(
                    request_id, experiment_id, payload_json, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    request_id,
                    request.experiment_id,
                    canonical_json(decision.to_json()),
                    decision.created_at,
                ),
            )
        return decision

    def require_decision(
        self,
        request_id: str,
        *,
        kind: str,
        action: str,
        experiment_id: str,
        subject_hash: str,
    ) -> tuple[DecisionRequest, ResearchDecision]:
        request = self.get(request_id)
        decision = self.decision(request_id)
        if decision is None:
            raise InboxError(f"request {request_id} has not been answered")
        if request.kind != kind:
            raise InboxError(f"request kind is {request.kind!r}, expected {kind!r}")
        if decision.action != action:
            raise InboxError(
                f"decision action is {decision.action!r}, expected {action!r}"
            )
        if (
            request.experiment_id != experiment_id
            or decision.experiment_id != experiment_id
        ):
            raise InboxError("decision is attached to a different experiment")
        if request.subject_hash != subject_hash:
            raise InboxError("decision was issued for a different subject")
        return request, decision
