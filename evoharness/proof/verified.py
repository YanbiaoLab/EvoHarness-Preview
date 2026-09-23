"""Final re-verification and publication through the verifier.

`assembly.py` renders the finished proof and compiles it locally. With a
verifier, the rendering is the same -- one assembler, not two -- and the
compile is the verifier's, in `assembly` mode, against the root goal's key and
environment. What comes back is a certification the verifier holds and can be
asked about, instead of a local exit code nobody else can check.

Then publication: the certified root goes into the verifier's managed graph,
where a later task can find it and, once the verifier folds it into a new
environment, use it. The request carries the exact body and context that were
certified -- read back from the stored envelope, never re-rendered -- because
the verifier compares them byte for byte and dict for dict.

Three things the assembled file needs that a leaf candidate does not:

- **The preamble comes off.** Sketches carry the graph's preamble and
  `assemble` hoists it to the top; the verifier takes it as imports and
  context instead. A sketch whose preamble is not the graph's would change
  what the file means, so it is refused.
- **Every lemma is a declared root.** The file declares the root and every
  subgoal lemma. Roots are declared, never inferred on the other side: the
  verifier checks the set it gets against the set it was told.
- **Names are reported qualified.** The same namespace prefix the root goal
  was resolved under applies to every declaration in the file.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .assembly import AssemblyError, AssemblyResult, _assemble_body, _pin_map, _top_route
from .envelope import SourceEnvelope, preamble_context
from .policy import AxiomPolicy
from .verifier import (
    ASSEMBLY_LIMITS,
    GoalUnresolvable,
    VerifierClient,
    VerifierRejected,
    VerifierUnavailable,
    _DECLARATION,
    ensure_contracts,
    verification_request,
)
from .run_solver import declaration_name

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .graph import Certification
    from .sketch import Validation
    from .store import ProofGraphStore


def declared_roots(body: str, prefix: str) -> tuple[str, ...]:
    """The name-minimal declarations in `body`, qualified, in source order.

    Name-minimal because that is what the verifier reports: `foo.aux` beside
    `foo` belongs to `foo`. Component-wise, not string-wise: `foo_bar` is not
    under `foo`.
    """

    names = []
    for match in _DECLARATION.finditer(body):
        name = declaration_name(body[match.start(1):].split(":=", 1)[0])
        if name and name not in names:
            names.append(name)
    minimal = [n for n in names
               if not any(o != n and n.startswith(o + ".") for o in names)]
    return tuple(prefix + n for n in minimal)


def assembly_envelope(
    store: "ProofGraphStore", goal_id: str, *, preamble: str, name_prefix: str,
    routes: Iterable[str] = (),
) -> SourceEnvelope:
    """The assembled proof of `goal_id`, split for the verifier. Rendered once."""

    pins = _pin_map(store, routes)
    hoisted: list[str] = []
    body, root = _assemble_body(store, goal_id, pins, hoisted, set())
    graph_lines = [line for line in preamble.strip().splitlines()]
    foreign = [line for line in hoisted if line.strip() and line not in graph_lines]
    if foreign:
        raise AssemblyError(
            "a sketch on this route carries preamble lines the graph does not: "
            f"{foreign[:3]}; the verifier checks the file under the graph's own"
        )
    body = body.strip() + "\n"
    context = preamble_context(preamble)
    roots = declared_roots(body, name_prefix)
    primary = name_prefix + root
    if primary not in roots:
        raise AssemblyError(f"the assembled file does not declare its root `{root}`")
    full_text = (preamble.strip() + "\n\n" + body) if preamble.strip() else body
    return SourceEnvelope(
        candidate_file_sha256=hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
        imports=context.imports,
        ctx=dict(context.ctx),
        body=body,
        primary_root=primary,
        expected_roots=(primary, *[r for r in roots if r != primary]),
    )


def verify_with_verifier(
    store: "ProofGraphStore",
    goal_id: str,
    *,
    client: VerifierClient,
    base: Mapping[str, Any],
    preamble: str,
    policy: AxiomPolicy | None = None,
    routes: Iterable[str] = (),
) -> tuple[AssemblyResult, dict | None]:
    """Assemble and ask the verifier. (result, the verifier's side or None).

    Raises `AssemblyError` when nothing was measured -- no answer, or no key
    for the goal -- exactly as the local path does: a verdict of "fails" would
    be a much stronger claim than "could not check".
    """

    policy = policy or AxiomPolicy()
    goal = store.goal(goal_id)
    try:
        contracts = ensure_contracts(store, client, [goal], base=base, preamble=preamble)
    except VerifierUnavailable as exc:
        raise AssemblyError(f"could not reach the verifier: {exc}") from exc
    contract = contracts[goal.id]
    if isinstance(contract, GoalUnresolvable):
        raise AssemblyError(f"the root goal does not elaborate in the verifier: {contract}")

    envelope = assembly_envelope(store, goal_id, preamble=preamble,
                                 name_prefix=contract.name_prefix, routes=routes)
    store.record_envelope(envelope)
    decomposition = _top_route(store, goal_id, _pin_map(store, routes))
    request = verification_request(
        envelope, contract, minimum_trust=policy.minimum_trust, mode="assembly",
        limits=ASSEMBLY_LIMITS, route_id=decomposition.id if decomposition else "")
    try:
        # Asynchronous: an assembled tree can take far longer than one HTTP
        # call should stay open, and a lost response must not start it twice.
        answer = client.verify_async(request)
    except VerifierUnavailable as exc:
        raise AssemblyError(f"could not run the final check: {exc}") from exc
    except VerifierRejected as exc:
        raise AssemblyError(f"the verifier refused the request: {exc}") from exc

    status, reason = answer.get("status"), answer.get("reason")
    cert = answer.get("certification") or {}
    text = envelope.body
    if status != "verified" or not cert:
        messages = "\n".join(answer.get("messages") or [])
        return AssemblyResult(
            ok=False,
            reason=(f"the verifier did not certify the assembled proof: {status}/{reason}\n"
                    f"{messages}")[:4000],
            text=text,
        ), None
    axioms = frozenset(cert.get("axiom_set") or ())
    external = {
        "certification_id": cert["certification_id"],
        "goal_key": cert["goal_key"],
        "base_fingerprint": base.get("fingerprint"),
        "verification_job_id": answer.get("job_id"),
        "source_sha256": cert["source_sha256"],
        "runtime_fingerprint": cert["verifier_runtime_fingerprint"],
        "evidence_hash": cert["evidence_hash"],
        "envelope_hash": envelope.envelope_hash,
        "trust": cert["trust"],
    }
    return AssemblyResult(ok=True, text=text, axioms=axioms, trust=cert["trust"]), external


def certify_with_verifier(
    store: "ProofGraphStore",
    goal_id: str,
    **kwargs,
) -> "tuple[AssemblyResult, Certification]":
    """`verify_with_verifier`, then record the verdict -- pass or fail -- on the goal."""

    routes = kwargs.get("routes", ())
    result, external = verify_with_verifier(store, goal_id, **kwargs)
    decomposition = _top_route(store, goal_id, _pin_map(store, routes))
    certification = store.record_certification(
        goal_id,
        ok=result.ok,
        axioms=result.axioms,
        reason=result.reason,
        text_sha256=hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
        decomposition_id=decomposition.id if decomposition else "",
        trust=result.trust,
        external=external,
    )
    return result, certification


@dataclass
class VerifierSketchValidator:
    """Validate a decomposition in the verifier's `sketch` mode.

    Same two questions as `LeanSketchValidator` -- does the parent typecheck
    given the lemma statements, and is `sorry` confined to the lemmas -- asked
    of the verifier instead of a local compile, plus the verifier's third:
    the parent must actually use every lemma. A sketch check that could not
    run raises `SketchUnavailable`, which leaves the route PROPOSED rather than
    rejecting it for good.
    """

    client: VerifierClient
    preamble: str
    policy: AxiomPolicy
    contract_for: "Callable[[Any], Any]"

    def __call__(self, goal, sketch) -> "Validation":
        from .sketch import SketchUnavailable, Validation, render

        if sketch.parent_signature.strip() != goal.statement.strip():
            return Validation(ok=False,
                              reason="the sketch closes a different proposition than the goal")
        if sketch.preamble.strip() and sketch.preamble.strip() != self.preamble.strip():
            return Validation(ok=False,
                              reason="the sketch's preamble is not the graph's")
        try:
            contract = self.contract_for(goal)
        except VerifierUnavailable as exc:
            raise SketchUnavailable(f"the verifier did not answer: {exc}") from exc
        except GoalUnresolvable as exc:
            raise SketchUnavailable(f"the goal does not elaborate in the verifier: {exc}") from exc
        body = render(sketch, preamble=False).text
        context = preamble_context(self.preamble)
        prefix = contract.name_prefix
        primary = prefix + sketch.parent_name
        full = (self.preamble.strip() + "\n\n" + body) if self.preamble.strip() else body
        envelope = SourceEnvelope(
            candidate_file_sha256=hashlib.sha256(full.encode("utf-8")).hexdigest(),
            imports=context.imports, ctx=dict(context.ctx), body=body,
            primary_root=primary,
            expected_roots=(primary, *[prefix + spec.name for spec in sketch.subgoals]),
        )
        request = verification_request(envelope, contract,
                                       minimum_trust=self.policy.minimum_trust, mode="sketch")
        try:
            answer = self.client.verify(request)
        except (VerifierUnavailable, VerifierRejected) as exc:
            raise SketchUnavailable(f"the sketch check could not run: {exc}") from exc
        status, reason = answer.get("status"), answer.get("reason")
        if status == "verified":
            return Validation(ok=True, reason=f"{len(sketch.subgoals)} lemmas, parent closed")
        from .verdict import classify
        from .graph import NO_VERDICT_OUTCOMES

        if classify(status, reason) in NO_VERDICT_OUTCOMES:
            raise SketchUnavailable(f"the verifier gave no verdict on the sketch: {status}/{reason}")
        messages = "\n".join(answer.get("messages") or [])
        return Validation(ok=False, reason=f"{status}/{reason}: {messages}"[:2000])


class PublicationError(RuntimeError):
    """Nothing to publish, or the verifier would not take it."""


def publish(
    store: "ProofGraphStore",
    certification_id: str,
    *,
    client: VerifierClient,
    provenance: Mapping[str, Any] | None = None,
) -> dict:
    """Publish one verifier-certified root. Idempotent by certification.

    The key is derived from the certification, so a retry reuses it: the
    verifier refuses a second key for the same certification, and a fresh key
    per attempt would turn every retry after a lost response into an error.
    """

    cert = store.certification(certification_id)
    if not cert.ok or not cert.external:
        raise PublicationError(
            f"certification {certification_id} is not a verifier certification of a "
            "passing proof; only those can be published")
    envelope = store.envelope(cert.external["envelope_hash"])
    contract = store.goal_contract(cert.goal_id)
    if contract is None:
        raise PublicationError(f"goal {cert.goal_id} has no verifier contract")
    key = f"publish:{cert.external['certification_id']}"
    request = {
        "schema_version": 1,
        "idempotency_key": key,
        "certification_id": cert.external["certification_id"],
        "source": envelope.body,
        "context": dict(envelope.ctx),
        "base": dict(contract.base),
        "provenance": {
            "candidate_file_sha256": envelope.candidate_file_sha256,
            "envelope_hash": envelope.envelope_hash,
            "goal_id": cert.goal_id,
            "route_id": cert.decomposition_id,
            **dict(provenance or {}),
        },
    }
    try:
        answer = client.publish(request)
    except (VerifierUnavailable, VerifierRejected) as exc:
        raise PublicationError(str(exc)) from exc
    store.record_publication(certification_id, idempotency_key=key, answer=answer)
    return answer


__all__ = [
    "PublicationError",
    "VerifierSketchValidator",
    "assembly_envelope",
    "certify_with_verifier",
    "declared_roots",
    "publish",
    "verify_with_verifier",
]
