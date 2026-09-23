"""One narrow view of the proof graph, for callers that are not Python.

    uv run python -m evoharness.proof.cli open   --statement '...'
    uv run python -m evoharness.proof.cli status
    uv run python -m evoharness.proof.cli sketch --goal G --proposal '{...}'
    uv run python -m evoharness.proof.cli attack --goal G
    uv run python -m evoharness.proof.cli assemble --goal G

Every invocation prints exactly one JSON object and exits 0 on success, or
prints `{"error": ...}` and exits 1. Nothing is written to stdout but that
object, so a caller can parse it without stripping logs.

**Why a CLI rather than a TypeScript reimplementation.** The dsh tools that
front this shell out to it for the same reason `peer.ts` already does: the
schema lives in Python, and a second reader written on the other side would
drift from it silently, because both would keep answering. This is the one
narrow surface, built from a fixed set of keys rather than filtered down from
internal state.

**What this does NOT hand over.** A model driving these tools proposes and
inspects; it never adjudicates. `sketch` returns Lean's verdict on a
decomposition, `attack` returns a run's outcome derived from that run's own
report, and `assemble` returns the compiler's answer about the finished proof.
None of the three can be talked into agreeing.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from .assembly import AmbiguousRoute, AssemblyError, certify
from .controller import ProofController
from .envelope import preamble_context
from .grade import make_grader
from .graph import DecompositionStatus, GoalStatus
from .identity import ExactTextHasher, LeanExprHasher
from .inspect import InspectError, attempt_view
from .propose import ProposalError, parse_proposal
from .policy import AxiomPolicy
from .run_solver import ApiRunSolver, declaration_name
from .scope import GraphScope, local_environment, preamble_sha256
from .sketch import LeanRunner, LeanSketchValidator, SketchUnavailable
from .store import ProofGraphStore
from .verified import (
    PublicationError,
    VerifierSketchValidator,
    certify_with_verifier,
    publish,
)
from .verifier import (
    GoalUnresolvable,
    VerifierBinding,
    VerifierClient,
    VerifierUnavailable,
    ensure_contracts,
    verifier_environment,
)

#: Where the graph and every run directory live. One workspace is one line of
#: enquiry; pointing two sessions at the same one is how a conversation and a
#: long run share a graph on purpose.
WORK_ENV = "EVO_PROOF_WORK"
#: The lake project that puts Mathlib on the search path. Absent means bare
#: `lean`, which is right for core-Lean goals and wrong for every real
#: benchmark problem.
PROJECT_ENV = "EVO_LEAN_PROJECT"

#: The verifier's URL. Set, every candidate, sketch and assembled proof is
#: judged there and nothing is compiled locally; the graph's environment is the
#: verifier's (`verifier:<fingerprint>`), a scope setting like any other.
VERIFIER_ENV = "EVO_VERIFIER_URL"
#: Which of the verifier's environments: a base key it can look up, or
#: `@file.json` holding the whole environment record (for a verifier running
#: without its database).
VERIFIER_BASE_ENV = "EVO_VERIFIER_BASE"

#: How far a goal's lease outlives the attack it covers. It has to absorb
#: everything around the solver's own limit -- starting an interpreter, and
#: writing the attempt down afterwards -- plus the margin the caller keeps
#: back, because a caller that waits longer than the solver is exactly the
#: arrangement this layer is built for.
_LEASE_MARGIN_S = 300.0


def _runner(args) -> LeanRunner:
    project = args.lean_project or os.environ.get(PROJECT_ENV)
    if project:
        return LeanRunner.mathlib(project, timeout_s=args.lean_timeout)
    return LeanRunner(timeout_s=args.lean_timeout)


def _work(args) -> Path:
    return Path(args.work or os.environ.get(WORK_ENV) or ".proof")


def _store(args) -> ProofGraphStore:
    """Open the graph under the scope this command works in, or refuse.

    The scope is resolved before opening: what the command will actually use
    (the Lean it compiles against, the preamble it reads) plus the settings it
    states by flag, with the unstated ones inherited from the graph. The
    resolved scope is left on `args` so the hasher and the axiom policy come
    from the graph rather than from whichever flags this invocation remembered.
    """

    work = _work(args)
    work.mkdir(parents=True, exist_ok=True)
    path = work / "graph.db"
    if getattr(args, "force_new_graph", False):
        args._retired = _retire_graph(work)
    preamble = _read_preamble(args)
    # Checked before any scope is written: a preamble the verifier cannot use
    # would otherwise be fixed into the graph, and every candidate under it
    # would fail as the candidate's fault.
    preamble_context(preamble)
    verifier = _verifier(args)
    scope = GraphScope.resolve(
        stored=ProofGraphStore.read_scope(path),
        environment=(
            verifier_environment(verifier[1]) if verifier
            else local_environment(_project(args))
        ),
        preamble_sha256=preamble_sha256(preamble),
        identity_hasher=_requested_hasher(args),
        minimum_trust=getattr(args, "minimum_trust", None),
        base_id=verifier[1].get("base_id", "") if verifier else "",
    )
    store = ProofGraphStore(
        path, scope, adopt_scope=getattr(args, "adopt_scope", False)
    )
    args._scope = store.scope
    return store


def _verifier(args) -> "tuple[VerifierClient, dict] | None":
    """(client, the environment record) when a verifier is configured, else None.

    Resolved once per command and kept on `args`: the graph's scope, the
    grader, the sketch check and assembly must all talk about the same
    environment.
    """

    if hasattr(args, "_verifier"):
        return args._verifier
    url = getattr(args, "verifier", "") or os.environ.get(VERIFIER_ENV, "")
    if not url:
        args._verifier = None
        return None
    spec = getattr(args, "verifier_base", "") or os.environ.get(VERIFIER_BASE_ENV, "")
    if not spec:
        raise ValueError(
            f"--verifier needs --verifier-base (or ${VERIFIER_BASE_ENV}): a base key, "
            "or @file.json with the environment record")
    client = VerifierClient(url)
    if spec.startswith("@"):
        base = json.loads(Path(spec[1:]).read_text(encoding="utf-8"))
    else:
        base = client.base(int(spec))
    if "fingerprint" not in base:
        raise ValueError("the verifier's environment record has no fingerprint")
    args._verifier = (client, base)
    return args._verifier


def _contract_for(args, store: ProofGraphStore):
    """goal -> the verifier's contract for it, resolved once and kept in the graph."""

    client, base = _verifier(args)
    preamble = _read_preamble(args)

    def contract_for(goal):
        found = ensure_contracts(store, client, [goal], base=base, preamble=preamble)[goal.id]
        if isinstance(found, GoalUnresolvable):
            raise found
        return found

    return contract_for


def _retire_graph(work: Path) -> list[str]:
    """Move the graph and its preamble aside. Renamed, never deleted.

    The graph runs in WAL mode, so committed pages can still sit in
    `graph.db-wal`: checkpoint first, then move the database and both sidecar
    files together. A rename onto an existing path would overwrite it
    silently, so the suffix is made unique rather than trusted to be.
    """

    db = work / "graph.db"
    if db.is_file():
        conn = sqlite3.connect(str(db))
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    names = ("graph.db", "graph.db-wal", "graph.db-shm", "preamble.lean")
    suffix, n = stamp, 0
    while any((work / f"{name}.{suffix}.bak").exists() for name in names):
        n += 1
        suffix = f"{stamp}.{n}"
    moved = []
    for name in names:
        src = work / name
        if src.exists():
            dst = work / f"{name}.{suffix}.bak"
            src.rename(dst)
            moved.append(str(dst))
    return moved


def _project(args) -> str:
    return getattr(args, "lean_project", "") or os.environ.get(PROJECT_ENV, "")


def _requested_hasher(args) -> str | None:
    """The hasher this command names, or None to inherit the graph's."""

    named = getattr(args, "identity", None)
    if getattr(args, "lean_identity", False):
        if named not in (None, "lean-expr"):
            raise ValueError("--lean-identity contradicts --identity " + named)
        return "lean-expr"
    return named


def _hasher(args):
    scope = getattr(args, "_scope", None)
    name = scope.identity_hasher if scope else (_requested_hasher(args) or "exact-text")
    if name == "lean-expr":
        return LeanExprHasher()
    return ExactTextHasher()


def _policy(args) -> AxiomPolicy:
    scope = getattr(args, "_scope", None)
    return scope.policy if scope else AxiomPolicy()


def _certification_view(store: ProofGraphStore, goal_id: str) -> dict | None:
    """The latest compile of the assembled proof, or None when there was none.

    Shown beside `status` because the two answer different questions: PROVED
    says the route closed, and this says the finished file compiled. A reader
    given only the first will take it for the second.
    """

    cert = store.latest_certification(goal_id)
    if cert is None:
        return None
    return {
        "ok": cert.ok,
        "axioms": sorted(cert.axioms),
        "reason": cert.reason[:400],
        # Which route was compiled. `""` means the goal had none -- a solver
        # closed it directly -- and null means the record predates this being
        # kept, when whichever route was oldest won silently.
        "decomposition_id": cert.decomposition_id,
        # The trust level of the finished proof. Shown because accepting
        # native_decide means an `audited` proof counts as proved, and a
        # reader must be able to tell it from a `trusted` one.
        "trust": cert.trust,
        "at": cert.created_at,
    }


def _goal_view(store: ProofGraphStore, goal) -> dict:
    attempts = store.attempts_of(goal.id)
    decompositions = store.decompositions_of(goal.id)
    return {
        "goal_id": goal.id,
        "name": declaration_name(goal.statement),
        "status": goal.status.value,
        "certified": _certification_view(store, goal.id),
        "statement": goal.statement,
        # The id is here so `attempt` can be asked about one of these rather
        # than only about the most recent: without it the board describes
        # attempts the caller has no way to name.
        "attempts": [
            {"id": a.id, "outcome": a.outcome.value, "note": a.note[:200]}
            for a in attempts
        ],
        "decompositions": [
            {
                "id": d.id,
                "status": d.status.value,
                "reason": d.rejected_reason[:400],
                "subgoals": [
                    {
                        "goal_id": sid,
                        "name": declaration_name(store.goal(sid).statement),
                        "status": store.goal(sid).status.value,
                    }
                    for sid in d.subgoal_ids
                ],
            }
            for d in decompositions
        ],
        "exhausted_at_budget": goal.exhausted_at_budget,
        "exhausted_at_solver": goal.exhausted_at_solver,
    }


# --- commands ----------------------------------------------------------------

def cmd_open(args) -> dict:
    """Create or reopen a root goal. Idempotent by identity."""

    statement = args.statement.strip()
    if not statement:
        raise ValueError("--statement is empty")
    if ":=" in statement:
        raise ValueError(
            "--statement is a declaration SIGNATURE: everything up to but not "
            "including `:=`"
        )
    store = _store(args)
    try:
        identity = _hasher(args).hash_many([statement])[0]
        goal = store.upsert_goal(identity, statement)
        store.add_root(goal.id, args.label or declaration_name(statement))
        if args.preamble:
            _write_preamble(args, args.preamble)
        return {"opened": _goal_view(store, goal)}
    finally:
        store.close()


def cmd_status(args) -> dict:
    store = _store(args)
    try:
        if args.goal:
            return {"goal": _goal_view(store, store.goal(args.goal))}
        roots = store._conn.execute(
            "SELECT goal_id, label FROM roots ORDER BY label"
        ).fetchall()
        return {
            "roots": [
                {"label": row["label"], **_goal_view(store, store.goal(row["goal_id"]))}
                for row in roots
            ],
            "open_goals": [
                {"goal_id": g.id, "name": declaration_name(g.statement)}
                for g in store.open_goals()
            ],
            "spent": store.total_cost(),
        }
    finally:
        store.close()


def cmd_sketch(args) -> dict:
    """Validate a proposed decomposition, and record it if Lean accepts.

    The proposal arrives as the same json a model is asked for. Parsing is
    strict: a malformed one is refused rather than repaired, because a
    half-understood sketch that happens to compile enters the graph carrying an
    implication nobody checked.
    """

    store = _store(args)
    try:
        goal = store.goal(args.goal)
        payload = _read_proposal(args)
        parent_name = declaration_name(goal.statement)
        try:
            sketch = parse_proposal(
                json.dumps(payload),
                goal=goal,
                parent_name=parent_name,
                preamble=_read_preamble(args),
            )
        except ProposalError as exc:
            return {"accepted": False, "stage": "parse", "reason": str(exc)}

        identities = _hasher(args).hash_many(
            [spec.signature for spec in sketch.subgoals]
        )
        from .sketch import Sketch, SubgoalSpec

        sketch = Sketch(
            parent_name=sketch.parent_name,
            parent_signature=sketch.parent_signature,
            parent_body=sketch.parent_body,
            subgoals=tuple(
                SubgoalSpec(name=s.name, identity=i, signature=s.signature)
                for s, i in zip(sketch.subgoals, identities)
            ),
            preamble=sketch.preamble,
        )

        controller = _controller(args, store, solver_needed=False)
        outcome = controller.record_decomposition(goal, sketch)
        view = _goal_view(store, store.goal(goal.id))
        if outcome.cycles:
            return {
                "accepted": False,
                "stage": "acyclicity",
                "reason": (
                    "a proposed lemma restates one of this goal's own "
                    "ancestors, so it cannot make progress"
                ),
                "goal": view,
            }
        if outcome.deferred:
            # Its own stage, never `lean`. The checker did not run, so nothing
            # has been said about this route: a caller told "rejected" would
            # abandon a decomposition that may be perfectly sound, and would
            # read a Lean outage as evidence about the mathematics.
            return {
                "accepted": False,
                "stage": "deferred",
                "reason": (
                    "the Lean check could not run, so this route has NOT been "
                    "judged -- retry it rather than proposing differently"
                ),
                "goal": view,
            }
        latest = view["decompositions"][-1] if view["decompositions"] else {}
        return {
            "accepted": bool(outcome.accepted),
            "stage": "lean",
            "reason": latest.get("reason", ""),
            "goal": view,
        }
    finally:
        store.close()


def _attack_warnings(store, goal) -> list[str]:
    """What the caller should know before spending on this goal.

    Warnings rather than refusals, on purpose. A proved lemma is an asset
    whatever asked for it -- the graph is memoized by identity, so it can serve
    another route or another problem entirely -- and refusing to prove one
    because of a bookkeeping state upstream would throw that away. It is the
    same category error as letting an infra fault decide a goal is hard.

    But silence is worse: an unaccepted route is exactly where budget goes in
    and nothing ever closes.
    """

    warnings: list[str] = []
    if goal.status is GoalStatus.PROVED:
        warnings.append("this goal is already proved; attacking it spends for nothing")
    if goal.status is GoalStatus.EXHAUSTED:
        warnings.append(
            "this goal was ruled exhausted at budget "
            f"{goal.exhausted_at_budget} with solver "
            f"{goal.exhausted_at_solver!r}; attacking it again only helps if "
            "something about that has changed"
        )
    warnings.extend(_route_note(store, goal)[0])
    return warnings


def _route_note(store, goal) -> tuple[list[str], str | None]:
    """(warnings, refusal) about the routes this goal serves.

    The two ways a route can be unaccepted carry very different risk, and
    saying the same thing about both was too blunt:

    - PROPOSED means nobody has judged it yet, usually because the checker was
      down. The sketch may be perfectly good and its subgoals perfectly real.
    - REJECTED_BY_VERIFIER means Lean has judged it and refused. Its subgoals
      may be statements the model invented that do not hold at all.

    A goal with NO routes is not covered here: nothing referencing it is not
    the same as only bad things referencing it. Roots live there.
    """

    routes = store.decompositions_containing(goal.id)
    if not routes or any(
        route.status is DecompositionStatus.ACCEPTED for route in routes
    ):
        return [], None

    statuses = {route.status.value for route in routes}
    if statuses == {DecompositionStatus.REJECTED_BY_VERIFIER.value}:
        return [], (
            "every route that uses this goal was rejected by Lean, so this "
            "lemma is the leftover of a sketch that does not hold together. "
            "It may not even be true. Propose a route Lean accepts first, or "
            "pass --allow-unaccepted-route if you want it proved for its own "
            "sake."
        )
    return [], (
        "no route using this goal has been judged yet (routes are: "
        + ", ".join(sorted(statuses))
        + "). Proving it cannot close its parent -- only an ACCEPTED "
        "decomposition completes -- and the checker is not answering, so the "
        "route cannot be settled right now. Retry when it is, or pass "
        "--allow-unaccepted-route."
    )


def cmd_attack(args) -> dict:
    """Dispatch one goal to EvoHarness. One `api.run()`, one run directory."""

    store = _store(args)
    try:
        goal = store.goal(args.goal)
        controller = _controller(args, store)

        # Settle any route awaiting a verdict BEFORE deciding to refuse. A
        # route sits at PROPOSED because the checker was down, not because
        # anything is wrong with it, so refusing without trying again would
        # let one old timeout block work that is fine now. After this, a
        # refusal means the checker is down at this moment, not that it once
        # was.
        controller.revalidate_proposed()
        goal = store.goal(goal.id)

        warnings = _attack_warnings(store, goal)
        _, refusal = _route_note(store, goal)
        if refusal and not args.allow_unaccepted_route:
            return {
                "refused": True,
                "reason": refusal,
                "goal": _goal_view(store, goal),
                "warnings": warnings,
            }

        with _pinned(args, store, controller) as pin_note:
            report = controller.solve(
                goal.id, budget=args.budget, max_iterations=args.max_iterations
            )
        if pin_note:
            warnings.append(pin_note)
        return {
            "report": {
                "root_proved": report.root_proved,
                "stopped_reason": report.stopped_reason,
                "attempts": report.attempts,
                "spent": report.spent,
                "decompositions_accepted": report.decompositions_accepted,
                "cycles_refused": report.cycles_refused,
            },
            "goal": _goal_view(store, store.goal(goal.id)),
            "warnings": warnings,
            "refused": False,
        }
    finally:
        store.close()


def cmd_attempt(args) -> dict:
    """Read one finished attempt back. Costs nothing and starts nothing."""

    store = _store(args)
    try:
        return attempt_view(
            store,
            args.goal,
            args.attempt or None,
            include_code=args.code,
        )
    finally:
        store.close()


def cmd_assemble(args) -> dict:
    """The only thing that certifies a root: compile the finished article."""

    store = _store(args)
    try:
        goal = store.goal(args.goal)
        if goal.status is not GoalStatus.PROVED:
            return {
                "ok": False,
                "reason": f"goal is {goal.status.value}; nothing to assemble",
            }
        try:
            verifier = _verifier(args)
            if verifier:
                result, certification = certify_with_verifier(
                    store, goal.id, client=verifier[0], base=verifier[1],
                    preamble=_read_preamble(args), policy=_policy(args), routes=args.route,
                )
            else:
                result, certification = certify(
                    store, goal.id, runner=_runner(args), routes=args.route,
                    policy=_policy(args),
                )
        except AmbiguousRoute as exc:
            # A question, not a fault: answered by naming a route, so it comes
            # back as a payload the caller can act on rather than as an error
            # that reads like the graph is broken. Nothing is compiled and
            # nothing is recorded -- there is no verdict to record yet.
            return {
                "ok": False,
                "refused": True,
                "reason": str(exc),
                "goal": exc.goal_id,
                "routes": [
                    {"decomposition_id": route_id, "subgoals": list(names)}
                    for route_id, names in exc.candidates
                ],
            }
        out = Path(args.out) if args.out else None
        if result.ok and out:
            out.write_text(result.text, encoding="utf-8")
        return {
            "ok": result.ok,
            "reason": result.reason[:2000],
            "axioms": sorted(result.axioms),
            "certification_id": certification.id,
            # The verifier's own record, when it was the verifier that checked.
            "verifier": certification.external,
            "decomposition_id": certification.decomposition_id,
            "written_to": str(out) if (result.ok and out) else None,
            "text": result.text if args.show_text else None,
        }
    finally:
        store.close()


def cmd_publish(args) -> dict:
    """Publish a goal's latest verifier certification into the verifier's library."""

    if not _verifier(args):
        raise ValueError("publish needs --verifier: a local compile is not publishable")
    store = _store(args)
    try:
        cert = store.latest_certification(args.goal)
        if cert is None:
            raise ValueError("this goal has no certification; run `assemble` first")
        try:
            answer = publish(store, cert.id, client=_verifier(args)[0],
                             provenance={"label": args.label} if args.label else None)
        except PublicationError as exc:
            return {"published": False, "reason": str(exc)}
        return {"published": True, "certification_id": cert.id, "publication": answer}
    finally:
        store.close()


# --- wiring ------------------------------------------------------------------

class _RefusesToAttack:
    """Stands in where a solver is structurally required but must not be used.

    Validating a decomposition costs one Lean compile. Building a real solver
    for it would demand API credentials, so checking whether a route is sound
    would require the ability to spend money on it -- and a person reviewing a
    proposed decomposition should not need that.
    """

    level = "none"

    def attack(self, goal, *, budget):
        raise RuntimeError("this controller was built to validate, not to attack")


def _controller(
    args, store: ProofGraphStore, *, solver_needed: bool = True
) -> ProofController:
    from evoharness import BasicSearchProfile, ComponentSpec, RunSpec
    from evoharness.contracts.run import ProposalLimits

    runner = _runner(args)
    work = Path(args.work or os.environ.get(WORK_ENV) or ".proof")
    mode = "agentic" if args.level == "L2" else "single_shot"
    verifier = _verifier(args)
    binding = None
    validator = LeanSketchValidator(runner=runner)
    if verifier:
        from .retrieval import VerifierRetrievalProvider

        policy = _policy(args)
        contract_for = _contract_for(args, store)
        binding = VerifierBinding(
            client=verifier[0], preamble=_read_preamble(args), policy=policy,
            contract_for=contract_for,
            allow_helpers=getattr(args, "allow_helpers", False),
            retrieval=VerifierRetrievalProvider(verifier[0], policy.minimum_trust),
            on_retrieval=lambda goal, request_id, contract, hits: store.record_retrieval(
                goal.id, request_id, base=dict(contract.base), hits=hits),
        )
        validator = VerifierSketchValidator(
            client=verifier[0], preamble=_read_preamble(args), policy=policy,
            contract_for=contract_for)

    solver = _RefusesToAttack() if not solver_needed else ApiRunSolver(
        work_root=work / "runs",
        run_spec_factory=lambda out: RunSpec(
            models=(args.model,),
            proposer_backend=ComponentSpec.create(
                "proposer_backend", "proof.cli", version="v1"
            ),
            output_dir=str(out),
            seed=args.seed,
            # Set rather than defaulted. `ProposalLimits.timeout_s` defaults to
            # 5400s, which was chosen for an evolution run that owns its own
            # process; here the caller is a tool call with a ceiling of its
            # own, and whichever fires first decides what comes back. The
            # caller's ceiling produces `interrupted` -- a non-verdict, no
            # evidence about the goal, nothing recorded about the run -- while
            # this one produces `timeout`, which is a capability outcome. So
            # this must be the lower of the two, and the caller passes down
            # what it can actually wait for.
            proposal_limits=ProposalLimits(timeout_s=args.attack_timeout),
        ),
        search_profile_factory=lambda: BasicSearchProfile(
            num_trajectories=args.trajectories, proposal_mode=mode
        ),
        # With a verifier this is never called: one judge, no local precheck.
        grade_func=make_grader(runner, _policy(args)),
        transport=_transport(args),
        preamble=_read_preamble(args),
        level=args.level,
        on_attempt_dir=store.note_attempt_dir,
        verifier=binding,
    )
    return ProofController(
        store,
        solver,
        # `attack` never proposes a route: decomposition is `sketch`, and it is
        # the caller's move. Keeping them apart is what lets a person see the
        # decomposition before any budget is spent on it.
        decompositions=_NoDecompositions(),
        validate_sketch=validator,
        max_capability_attempts=args.max_attempts,
        # Longer than one attack can possibly last, and derived from it rather
        # than chosen. An expired lease means one thing -- the holder is dead
        # -- and recovery acts on it: force-releases the goal and records an
        # INTERRUPTED attempt. A lease shorter than the work it covers makes
        # that statement false halfway through every long attack, and then a
        # second solver starts on the same lemma while the first is still
        # proving it, with a fabricated interruption written against the one
        # that was working.
        lease_ttl_s=args.attack_timeout + _LEASE_MARGIN_S,
    )


class _pinned:
    """Pin the verifier's environment for the length of a run, and watch the view.

    Folding published declarations into a new environment moves them out of
    the old one, and a run pinned to the old one would stop seeing what it
    retrieved without being told. The pin makes the verifier refuse to fold
    while the run lives; the heartbeat renews it before every iteration, and
    stops the run if renewal fails or a retrieved declaration has vanished.

    A verifier without its database cannot pin; that is said once, as a
    warning, and the run goes on -- there is nothing to fold there either.
    """

    def __init__(self, args, store: ProofGraphStore, controller):
        self.args, self.store, self.controller = args, store, controller
        self.pin_id = None
        self.note = ""

    def __enter__(self) -> str:
        verifier = _verifier(self.args)
        if not verifier or "base_key" not in verifier[1]:
            return ""
        client, base = verifier
        ttl = self.args.attack_timeout + _LEASE_MARGIN_S
        try:
            self.pin_id = client.pin(int(base["base_key"]), holder=self.controller.owner,
                                     ttl_s=ttl)["pin_id"]
        except (VerifierUnavailable, ValueError) as exc:
            self.note = f"the verifier could not pin its environment: {exc}"
            return self.note
        minimum = _policy(self.args).minimum_trust

        def heartbeat():
            from .retrieval import vanished

            try:
                client.renew_pin(self.pin_id, ttl_s=ttl)
            except (VerifierUnavailable, ValueError):
                return "environment_pin_lost"
            try:
                if vanished(client, self.store.retrieved(), minimum_trust=minimum):
                    return "retrieval_view_shrank"
            except VerifierUnavailable:
                return None
            return None

        self.controller.heartbeat = heartbeat
        return ""

    def __exit__(self, *exc) -> None:
        if self.pin_id:
            try:
                _verifier(self.args)[0].release_pin(self.pin_id)
            except (VerifierUnavailable, ValueError):
                pass  # it expires on its own


class _NoDecompositions:
    def propose(self, goal):
        return None


def _transport(args):
    from evoharness.core.llm import make_openai_responses_transport

    base = os.environ.get("EVOHARNESS_API_BASE")
    key = os.environ.get("EVOHARNESS_API_KEY")
    if not base or not key:
        raise RuntimeError(
            "EVOHARNESS_API_BASE and EVOHARNESS_API_KEY must be set to attack "
            "a goal"
        )
    return make_openai_responses_transport(base, key, timeout_s=600.0)


def _preamble_path(args) -> Path:
    work = Path(args.work or os.environ.get(WORK_ENV) or ".proof")
    return work / "preamble.lean"


def _write_preamble(args, text: str) -> None:
    _preamble_path(args).write_text(text.strip() + "\n", encoding="utf-8")


def _read_preamble(args) -> str:
    if getattr(args, "preamble", None):
        return args.preamble.strip()
    path = _preamble_path(args)
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def _read_proposal(args) -> dict:
    raw = args.proposal
    if raw == "-":
        raw = sys.stdin.read()
    elif raw.startswith("@"):
        raw = Path(raw[1:]).read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--proposal is not json: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evoharness.proof.cli")
    parser.add_argument("--work", default="", help=f"default ${WORK_ENV}")
    parser.add_argument("--lean-project", default="", help=f"default ${PROJECT_ENV}")
    parser.add_argument("--lean-timeout", type=float, default=300.0)
    parser.add_argument("--lean-identity", action="store_true",
                        help="merge alpha-equivalent lemmas (asks Lean); "
                             "same as --identity lean-expr")
    # Scope settings. Left out, each is inherited from the graph; stated, it
    # must match the graph or the command is refused. See scope.py.
    parser.add_argument("--identity", choices=("exact-text", "lean-expr"),
                        default=None, help="identity hasher (a graph setting)")
    parser.add_argument("--minimum-trust", choices=("trusted", "audited", "claimed"),
                        default=None,
                        help="lowest trust a proof may have (a graph setting; "
                             "default for a new graph: audited)")
    parser.add_argument("--adopt-scope", action="store_true",
                        help="record this command's scope on a graph built "
                             "before scopes existed")
    parser.add_argument("--force-new-graph", action="store_true",
                        help="move the current graph and preamble aside "
                             "(renamed *.bak, never deleted) and start a new one")
    parser.add_argument("--verifier", default="",
                        help=f"the verifier's URL (default ${VERIFIER_ENV}); set, nothing "
                             "is compiled locally")
    parser.add_argument("--verifier-base", default="",
                        help=f"the verifier's environment: a base key, or @file.json "
                             f"(default ${VERIFIER_BASE_ENV})")
    parser.add_argument("--model", default=os.environ.get("DSH_MODEL", "gpt-5.6-sol"))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("open")
    p.add_argument("--statement", required=True)
    p.add_argument("--preamble", default="")
    p.add_argument("--label", default="")
    p.set_defaults(func=cmd_open)

    p = sub.add_parser("status")
    p.add_argument("--goal", default="")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("sketch")
    p.add_argument("--goal", required=True)
    p.add_argument("--proposal", required=True,
                   help='json, or @file, or - for stdin')
    p.add_argument("--preamble", default="")
    _solver_flags(p)
    p.set_defaults(func=cmd_sketch)

    p = sub.add_parser("attack")
    p.add_argument("--goal", required=True)
    p.add_argument("--budget", type=float, default=10.0)
    p.add_argument("--max-iterations", type=int, default=4)
    p.add_argument("--preamble", default="")
    # Refusing is the default. Budget spent on a goal no accepted route uses
    # buys a lemma that cannot close anything, and in a long autonomous run
    # nobody reads the warning that used to be all this said.
    p.add_argument("--allow-unaccepted-route", action="store_true")
    _solver_flags(p)
    p.set_defaults(func=cmd_attack)

    p = sub.add_parser("attempt")
    p.add_argument("--goal", required=True)
    p.add_argument("--attempt", default="",
                   help="one attempt id; omit for the most recent")
    p.add_argument("--code", action="store_true",
                   help="include the Lean file of the latest candidate")
    p.set_defaults(func=cmd_attempt)

    p = sub.add_parser("assemble")
    p.add_argument("--goal", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--show-text", action="store_true")
    # Repeatable rather than one value: a goal deep in the tree can have
    # several completed routes too, and pinning only the top would leave the
    # caller unable to answer the refusal it gets for the one below.
    p.add_argument("--route", action="append", default=[],
                   help="a completed decomposition to assemble through; "
                        "repeat for goals further down the tree")
    p.set_defaults(func=cmd_assemble)

    p = sub.add_parser("publish")
    p.add_argument("--goal", required=True)
    p.add_argument("--label", default="")
    p.set_defaults(func=cmd_publish)
    return parser


def _solver_flags(p) -> None:
    p.add_argument("--level", choices=("L1", "L2"), default="L2",
                   help="L1 one-shot; L2 an agent session that can call Lean")
    # Deliberately below anything that calls this. A caller with a shorter
    # ceiling gets `interrupted` and learns nothing about the goal; the whole
    # point of naming this is that the solver stops first, with a verdict.
    p.add_argument("--attack-timeout", type=float, default=840.0,
                   metavar="SECONDS",
                   help="how long one attempt may run before it stops itself; "
                        "must be under the caller's own ceiling")
    p.add_argument("--trajectories", type=int, default=1)
    p.add_argument("--allow-helpers", action="store_true",
                   help="with a verifier: let a proof declare helpers named under "
                        "its goal (`goal.step`)")
    p.add_argument("--max-attempts", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = args.func(args)
    except (
        ValueError, KeyError, RuntimeError, SketchUnavailable, AssemblyError,
        InspectError,
    ) as exc:
        json.dump({"error": f"{type(exc).__name__}: {exc}"}, sys.stdout)
        sys.stdout.write("\n")
        return 1
    retired = getattr(args, "_retired", None)
    if retired and isinstance(payload, dict):
        # Where the old graph went. Said once, here, because nothing else will.
        payload = {**payload, "retired_graph": retired}
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
