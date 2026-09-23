"""Grading through the verifier: one judge, over HTTP, instead of a local compile.

Before this, a candidate was scored by starting a local `lean` on it and reading
`#print axioms`. Two things were wrong with that once a verifier exists. Two
judges can disagree, and both would keep answering. And the local one is the
expensive one: a cold `lean` re-imports its whole environment per file, while
the verifier keeps a warm worker per environment -- two orders of magnitude
apart on throughput. So with a verifier configured there is no local precheck:
every candidate goes straight to it, and its certification is the only thing
that scores 1.0.

Three rules carried over from the local grader, unchanged in meaning:

- **`passed` means "it compiles", not "it is proved".** A candidate that still
  leans on `sorry` compiles, earns the floor, and can be a parent. The seed is
  exactly such a candidate; were it not passed, no island would ever propose.
- **Fitness is graded, 1.0 only from a certification.**
- **A broken judge is not a wrong answer.** No answer, or an answer that says
  nothing about the candidate -- the environment cannot host the preamble,
  the request was built wrong, the worker died -- raises `InfraError`. The
  candidate is kept, unverified, with its envelope, so it can be checked once
  the verifier is back.

Every answer, verdict or not, goes into the attempt's ledger. The ledger is the
per-candidate record the attempt's outcome is aggregated from (`verdict.py`),
and it is written to the run directory before the verifier is asked, so a
process that dies mid-request leaves the envelope behind.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contract import GoalContract, StatementError, proposition_of
from .envelope import EnvelopeError, SourceEnvelope, preamble_context, split_source
from .graph import NO_VERDICT_OUTCOMES, Outcome
from .policy import AxiomPolicy
from .run_solver import SEED_MAIN, declaration_name
from .verdict import UNVERIFIED, CandidateVerdict

#: Same floor as the local grader, for the same reason: a file that compiles is
#: progress over one that does not, and parent selection needs something to rank.
_COMPILES_FLOOR = 0.30

#: What a request asks the verifier for, unless told otherwise. Leaf requests
#: carry one lemma; an assembled file carries a whole tree and gets longer.
LEAF_LIMITS = {"max_heartbeats": 400_000, "wall_ms": 90_000, "memory_mb": 1_800}
#: How much longer than a request's own wall-clock limit a synchronous call may
#: take: HTTP, queueing behind other requests, and a cold worker's import. A cold
#: Mathlib worker can exceed it; that call then fails as no-answer and the retry
#: finds the worker warm.
REQUEST_MARGIN_S = 120.0
ASSEMBLY_LIMITS = {"max_heartbeats": 4_000_000, "wall_ms": 900_000, "memory_mb": 4_000}


class VerifierUnavailable(RuntimeError):
    """No answer: the verifier could not be reached, or failed on its side."""


class VerifierRejected(ValueError):
    """The verifier refused the request as malformed. A fault on this side."""


class GoalUnresolvable(ValueError):
    """The goal's statement does not elaborate in the verifier's environment."""


@dataclass(frozen=True)
class VerifierClient:
    """The verifier's HTTP API, as JSON in and JSON out. Nothing else."""

    url: str
    timeout_s: float = 960.0

    def _call(self, method: str, path: str, body: Mapping[str, Any] | None = None,
              *, timeout_s: float | None = None) -> dict:
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        request = urllib.request.Request(
            self.url.rstrip("/") + path, data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s or self.timeout_s) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("error", "")
            except Exception:  # noqa: BLE001 - the status code is what matters
                detail = ""
            if 400 <= exc.code < 500 and exc.code != 404:
                raise VerifierRejected(f"{method} {path}: {exc.code} {detail}") from exc
            if exc.code == 404:
                raise KeyError(f"{method} {path}: not found") from exc
            raise VerifierUnavailable(f"{method} {path}: {exc.code} {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            raise VerifierUnavailable(f"{method} {path}: {exc}") from exc

    def health(self) -> dict:
        return self._call("GET", "/v1/health")

    def base(self, base_key: int) -> dict:
        return self._call("GET", f"/v1/bases/{base_key}")["base"]

    def resolve(self, *, proposition: str, context: Mapping[str, Any],
                base: Mapping[str, Any], options: Mapping[str, Any] | None = None) -> dict:
        return self._call("POST", "/v1/goals/resolve", {
            "proposition": proposition, "context": dict(context),
            "options": dict(options or {}), "base": dict(base)})

    def resolve_batch(self, items: Sequence[Mapping[str, Any]], *,
                      base: Mapping[str, Any]) -> list[dict]:
        return self._call("POST", "/v1/goals/resolve-batch",
                          {"base": dict(base), "items": [dict(i) for i in items]})["results"]

    def verify(self, request: Mapping[str, Any]) -> dict:
        """Synchronous. Waits no longer than the request's own wall-clock limit plus
        `REQUEST_MARGIN_S`.

        The bound is what keeps a hung verifier from outliving the goal's lease.
        The lease is sized from the attack's own timeout; a grader blocked on a
        socket for longer would still be "working" when recovery declares it
        dead and a second solver starts on the same lemma. Bounded, a hang
        surfaces as `VerifierUnavailable` -- an infrastructure fault the run
        already knows how to treat -- well inside the lease.
        """

        wall = (request.get("limits") or {}).get("wall_ms")
        bound = wall / 1000 + REQUEST_MARGIN_S if wall else None
        return self._call("POST", "/v1/verifications", request, timeout_s=bound)

    def verification(self, job_id: str) -> dict:
        return self._call("GET", f"/v1/verifications/{job_id}")

    def submit(self, request: Mapping[str, Any]) -> dict:
        """Queue a verification and return at once. The job's state, not its verdict."""

        return self._call("POST", "/v1/jobs", request)

    def job(self, job_id: str) -> dict:
        return self._call("GET", f"/v1/jobs/{job_id}")

    def verify_async(self, request: Mapping[str, Any], *, poll_s: float = 2.0,
                     wait_s: float = 3600.0) -> dict:
        """Submit, then poll until the verdict is in. For work longer than one HTTP call.

        Resubmitting the same request finds the same job -- the idempotency key
        is derived from the request -- so a caller that gave up waiting, or
        lost the response, picks it up again rather than starting it twice. A
        verifier without its database has no job queue; then this is `verify`.
        """

        import time

        try:
            state = self.submit(request)
        except VerifierUnavailable as exc:
            if "no_database" in str(exc) or "503" in str(exc):
                return self.verify(request)
            raise
        deadline = time.monotonic() + wait_s
        while state.get("status") in ("queued", "running"):
            if time.monotonic() > deadline:
                raise VerifierUnavailable(
                    f"job {state.get('job_id')} still {state.get('status')} after {wait_s:g}s; "
                    "it keeps running -- ask again with the same request")
            time.sleep(poll_s)
            state = self.job(state["job_id"])
        return state["outcome"]

    def publish(self, request: Mapping[str, Any]) -> dict:
        return self._call("POST", "/v1/publications", request)

    def retrieve(self, request: Mapping[str, Any]) -> list[dict]:
        return self._call("POST", "/v1/retrieval", request)["hits"]

    def pin(self, base_key: int, *, holder: str, ttl_s: float) -> dict:
        return self._call("POST", "/v1/pins",
                          {"base_key": base_key, "holder": holder, "ttl_s": ttl_s})

    def renew_pin(self, pin_id: str, *, ttl_s: float) -> dict:
        return self._call("POST", f"/v1/pins/{pin_id}/renew", {"ttl_s": ttl_s})

    def release_pin(self, pin_id: str) -> dict:
        return self._call("POST", f"/v1/pins/{pin_id}/release", {})


def verifier_environment(base: Mapping[str, Any]) -> str:
    """The scope string for a graph judged by the verifier in this environment."""

    return f"verifier:{base['fingerprint']}"


# -- goal contracts -------------------------------------------------------------


def ensure_contracts(store, client: VerifierClient, goals, *, base: Mapping[str, Any],
                     preamble: str) -> dict[str, GoalContract | GoalUnresolvable]:
    """The verifier's contract for each goal, resolving the missing ones in one batch.

    A goal whose statement the verifier cannot elaborate comes back as a
    `GoalUnresolvable` rather than raising for the whole batch: it is an answer
    about that goal.
    """

    context = dict(preamble_context(preamble).ctx)
    out: dict[str, GoalContract | GoalUnresolvable] = {}
    missing = []
    for goal in goals:
        stored = store.goal_contract(goal.id)
        if stored is not None:
            out[goal.id] = stored
            continue
        try:
            missing.append((goal, proposition_of(goal.statement)))
        except StatementError as exc:
            out[goal.id] = GoalUnresolvable(str(exc))
    if missing:
        answers = client.resolve_batch(
            [{"proposition": prop, "context": context} for _, prop in missing], base=base)
        for (goal, prop), answer in zip(missing, answers):
            if not answer.get("ok"):
                out[goal.id] = GoalUnresolvable(answer.get("error", "resolution failed"))
                continue
            contract = GoalContract.from_resolved(
                answer["resolved"], base=base, proposition=prop, context=context)
            store.record_goal_contract(goal.id, contract)
            out[goal.id] = contract
    return out


# -- requests -------------------------------------------------------------------


def verification_request(
    envelope: SourceEnvelope,
    contract: GoalContract,
    *,
    minimum_trust: str,
    mode: str = "leaf",
    limits: Mapping[str, int] | None = None,
    route_id: str | None = None,
) -> dict[str, Any]:
    """The verifier's `VerificationRequest`, from one envelope. The one mapping.

    Mirrors the verifier's `from_envelope`: body to source, ctx to context,
    primary root to expected root, the other roots to `auxiliary_roots`,
    imports to imports. The idempotency key is derived from everything that
    decides the answer, so asking twice about the same candidate is answered
    from the verifier's record rather than recompiled.
    """

    if dict(envelope.ctx) != dict(contract.context):
        # The goal key hashes the context; a request under another one can only
        # come back as a mismatch, and would say the candidate proved the wrong
        # thing when it was this side that changed the premises.
        raise ValueError("the envelope's context differs from the goal's resolved context")
    key_material = {
        "goal": contract.goal_key, "envelope": envelope.envelope_hash, "mode": mode,
        "minimum_trust": minimum_trust, "route": route_id,
        "limits": dict(limits or LEAF_LIMITS),
    }
    digest = hashlib.sha256(json.dumps(key_material, sort_keys=True).encode()).hexdigest()
    key = f"evo-{mode}-{digest[:40]}"
    return {
        "schema_version": 1,
        "job_id": key,
        "idempotency_key": key,
        "base": dict(contract.base),
        "goal": dict(contract.goal_key_obj),
        "source": envelope.body,
        "expected_root": envelope.primary_root,
        "context": dict(envelope.ctx),
        "options": dict(contract.options),
        "mode": mode,
        "limits": dict(limits or LEAF_LIMITS),
        "minimum_trust": minimum_trust,
        "route_id": route_id,
        "imports": list(envelope.imports),
        "auxiliary_roots": [r for r in envelope.expected_roots if r != envelope.primary_root],
    }


# -- the ledger -----------------------------------------------------------------


class VerificationLedger:
    """Every candidate's envelope and answer for one attempt. Thread-safe.

    Candidates are graded concurrently. Written through to two JSONL files in
    the run directory when one is given, envelope first: the envelope must be
    on disk before the verifier is asked, so a crash mid-request still leaves
    what was sent.
    """

    def __init__(self, directory: Path | None = None):
        self._lock = threading.Lock()
        self._verdicts: dict[str, CandidateVerdict] = {}
        self._envelopes: dict[str, SourceEnvelope] = {}
        self._by_source: dict[str, str] = {}
        self.directory = Path(directory) if directory else None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    def _append(self, name: str, payload: Mapping[str, Any]) -> None:
        if self.directory:
            with open(self.directory / name, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def envelope(self, envelope: SourceEnvelope) -> None:
        with self._lock:
            if envelope.envelope_hash not in self._envelopes:
                self._envelopes[envelope.envelope_hash] = envelope
                self._by_source[envelope.candidate_file_sha256] = envelope.envelope_hash
                self._append("envelopes.jsonl", envelope.to_dict())

    def record(self, verdict: CandidateVerdict) -> None:
        with self._lock:
            self._verdicts[verdict.candidate_id] = verdict
            self._append("verdicts.jsonl", verdict.to_json())

    def verdicts(self) -> tuple[CandidateVerdict, ...]:
        with self._lock:
            return tuple(self._verdicts.values())

    def envelopes(self) -> tuple[SourceEnvelope, ...]:
        with self._lock:
            return tuple(self._envelopes.values())

    @staticmethod
    def load(directory: Path) -> tuple[tuple[CandidateVerdict, ...], tuple[SourceEnvelope, ...]]:
        """What a ledger left on disk. For recovery: a run that died still asked the
        verifier about its candidates, and those answers are the attempt's record."""

        directory = Path(directory)
        verdicts: dict[str, CandidateVerdict] = {}
        envelopes: dict[str, SourceEnvelope] = {}
        path = directory / "verdicts.jsonl"
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    verdict = CandidateVerdict.from_json(json.loads(line))
                    verdicts[verdict.candidate_id] = verdict
        path = directory / "envelopes.jsonl"
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    envelope = SourceEnvelope.from_dict(json.loads(line))
                    envelopes[envelope.envelope_hash] = envelope
        return tuple(verdicts.values()), tuple(envelopes.values())

    def envelope_for_file(self, text: str) -> str | None:
        """The envelope hash a candidate file was split into, without splitting again."""

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._lock:
            return self._by_source.get(digest)


# -- the grader -----------------------------------------------------------------


#: A top-level declaration keyword at the start of a line.
_DECLARATION = re.compile(
    r"^(?:@\[[^\]]*\]\s*)?(?:(?:private|protected|noncomputable|unsafe|partial)\s+)*"
    r"(theorem|lemma|def|abbrev|instance|structure|inductive|class|axiom|opaque|example)\b",
    re.M,
)


def extra_declarations(body: str, root: str, *, allow_named_under_root: bool) -> list[str]:
    """Top-level declarations in the edit region other than the goal's own.

    Text, not Lean: counting declarations must not cost a compile. With
    `allow_named_under_root`, a helper named under the goal (`root.helper`)
    is not extra: the verifier reports only name-minimal roots, so it folds
    into the goal's own declaration, and nothing it defines can collide with a
    sibling goal's helper when the proof is assembled.
    """

    extra = []
    for match in _DECLARATION.finditer(body):
        name = declaration_name(body[match.start(1):].split(":=", 1)[0])
        if name == root:
            continue
        if allow_named_under_root and name.startswith(root + "."):
            continue
        extra.append(f"{match.group(1)} {name}")
    return extra


@dataclass
class VerifierGrader:
    """A grade function bound to one goal, one verifier, one ledger."""

    client: VerifierClient
    contract: GoalContract
    #: The goal's declaration name, short (the verifier's roots add the prefix).
    root: str
    preamble: str
    policy: AxiomPolicy
    ledger: VerificationLedger = field(default_factory=VerificationLedger)
    #: Whether helpers named under the goal (`root.helper`) may sit beside it.
    #: False is the single-declaration rule: nothing but the goal's own
    #: declaration, because the graph has nowhere to keep anything else.
    allow_helpers: bool = False

    def __call__(self, candidate_dir, ctx) -> dict:
        from evoharness.serve import InfraError

        path = Path(candidate_dir) / SEED_MAIN
        if not path.is_file():
            return {"fitness": 0.0, "passed": False, "fault_kind": "invalid_candidate",
                    "fault": f"no {SEED_MAIN} in the workspace"}
        text = path.read_text(encoding="utf-8")
        source_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        candidate_id = getattr(ctx, "candidate_id", None) or source_sha[:16]

        try:
            envelope = split_source(text, preamble=self.preamble, root=self.root,
                                    name_prefix=self.contract.name_prefix)
        except EnvelopeError as exc:
            return {"fitness": 0.0, "passed": False, "fault_kind": "invalid_candidate",
                    "fault": f"the file breaks the edit rules: {exc}"}
        extra = extra_declarations(envelope.body, self.root,
                                   allow_named_under_root=self.allow_helpers)
        if extra:
            # A verdict on the candidate, made by text: the verifier would
            # refuse the roots anyway, and asking costs a request for nothing.
            rule = (f"helpers must be named under the goal (`{self.root}.helper`)"
                    if self.allow_helpers else "only the goal's own declaration is allowed")
            return {"fitness": 0.0, "passed": False, "fault_kind": "invalid_candidate",
                    "fault": f"{rule}; found: {', '.join(extra)}"}
        self.ledger.envelope(envelope)
        request = verification_request(envelope, self.contract,
                                       minimum_trust=self.policy.minimum_trust)

        def keep(status: str, reason: str | None = None, **kw) -> CandidateVerdict:
            verdict = CandidateVerdict(
                candidate_id=candidate_id, status=status, reason=reason,
                job_id=request["job_id"], envelope_hash=envelope.envelope_hash,
                source_sha256=source_sha, goal_key=self.contract.goal_key, **kw)
            self.ledger.record(verdict)
            return verdict

        try:
            answer = self.client.verify(request)
        except VerifierUnavailable as exc:
            keep(UNVERIFIED, note=str(exc)[:500])
            raise InfraError(f"the verifier did not answer: {exc}") from exc
        except VerifierRejected as exc:
            keep("policy_rejected", "root_contract", note=f"request refused: {exc}"[:500])
            raise InfraError(f"the verifier refused the request as malformed: {exc}") from exc

        return _grade_from(answer, keep, self.policy, InfraError)


def _grade_from(answer: Mapping[str, Any], keep, policy: AxiomPolicy, infra) -> dict:
    status = answer.get("status", "")
    reason = answer.get("reason")
    cert = answer.get("certification") or {}
    messages = "\n".join(answer.get("messages") or [])[:2000]
    verdict = keep(status, reason, certification_id=cert.get("certification_id"),
                   trust=cert.get("trust") or (answer.get("worker_result") or {}).get("trust"),
                   note=messages[:500])
    outcome = verdict.outcome
    if outcome is Outcome.PROVED:
        if not cert or not policy.accepts(cert.get("trust", "tainted")):
            # A certification below this graph's floor cannot come back for a
            # request that asked for the floor. If it does, the two sides
            # disagree about the policy, and that is not a proof here.
            raise infra(f"certification at trust {cert.get('trust')!r} is below "
                        f"minimum_trust={policy.minimum_trust}")
        return {
            "fitness": 1.0, "passed": True,
            "notes": f"certified by the verifier at trust {cert['trust']}",
            "visible_metrics": {"proved": 1, "trust": cert["trust"],
                                "certification_id": cert["certification_id"]},
        }
    if outcome in NO_VERDICT_OUTCOMES:
        raise infra(f"the verifier gave no verdict on the candidate: {status}/{reason}: "
                    f"{messages[:300]}")
    if reason in ("sorry_axiom", "native_unverified"):
        # Compiles; either unfinished, or resting on native_decide the verifier
        # could not recheck. Partial credit, and a parent.
        return {"fitness": _COMPILES_FLOOR, "passed": True, "notes": messages,
                "visible_metrics": {"proved": 0, "verifier": f"{status}/{reason}"}}
    if outcome is Outcome.TIMEOUT:
        return {"fitness": 0.0, "passed": False, "fault_kind": "timeout",
                "fault": "the proof ran past the verifier's time limit", "notes": messages}
    return {"fitness": 0.0, "passed": False, "fault_kind": "task_failure",
            "fault": _FAULTS.get(reason or status, f"{status}/{reason}"), "notes": messages,
            "visible_metrics": {"verifier": f"{status}/{reason}"}}


_FAULTS = {
    "elaboration_error": "the file does not compile",
    "kernel_replay": "a declaration does not pass the kernel",
    "proposition_differs": "the proof is of a different proposition than the goal",
    "meta_declaration": "the declaration is a metaprogram, not a statement",
    "forbidden_axiom": "the proof rests on an axiom outside the policy",
    "trust_below_minimum": "the proof's trust level is below the graph's floor",
    "root_not_replayed": "the goal is declared unsafe or partial",
}


@dataclass
class VerifierBinding:
    """Everything a solver needs to grade one graph's goals through the verifier.

    `contract_for` resolves a goal on first use and persists the answer; the
    CLI builds it over the graph's store, so the solver never touches storage.
    It raises `GoalUnresolvable` for a statement the environment cannot
    elaborate and `VerifierUnavailable` when there is no answer at all.
    """

    client: VerifierClient
    preamble: str
    policy: AxiomPolicy
    contract_for: "Callable[[Any], GoalContract]"
    allow_helpers: bool = False
    #: Asked before each attack; its hits go into the prompt. None: no retrieval.
    retrieval: "Any | None" = None
    #: Told (goal, request_id, contract, hits) so the graph can keep them.
    on_retrieval: "Callable[..., None] | None" = None

    def retrieve(self, goal, contract: GoalContract) -> str:
        """The prompt section for this goal's retrieval, or "" when there is none.

        No answer from the library is not a reason to skip the attack: the
        proof never depended on retrieval, only the prompt does.
        """

        from .retrieval import prompt_section

        if self.retrieval is None:
            return ""
        try:
            request_id, hits = self.retrieval.retrieve(goal, contract)
        except (VerifierUnavailable, VerifierRejected):
            return ""
        if self.on_retrieval is not None:
            self.on_retrieval(goal, request_id, contract, hits)
        return prompt_section(hits)

    def grader(self, goal, contract: GoalContract, run_dir: Path | None) -> VerifierGrader:
        return VerifierGrader(
            client=self.client, contract=contract,
            root=declaration_name(goal.statement), preamble=self.preamble,
            policy=self.policy,
            ledger=VerificationLedger(Path(run_dir) / "verifier" if run_dir else None),
            allow_helpers=self.allow_helpers,
        )


__all__ = [
    "ASSEMBLY_LIMITS",
    "VerifierBinding",
    "LEAF_LIMITS",
    "GoalUnresolvable",
    "VerificationLedger",
    "VerifierClient",
    "VerifierGrader",
    "VerifierRejected",
    "VerifierUnavailable",
    "ensure_contracts",
    "extra_declarations",
    "verification_request",
    "verifier_environment",
]
