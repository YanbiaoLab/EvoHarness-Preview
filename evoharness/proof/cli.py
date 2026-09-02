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
import sys
from pathlib import Path

from .assembly import AssemblyError, certify
from .controller import ProofController
from .grade import make_grader
from .graph import DecompositionStatus, GoalStatus
from .identity import ExactTextHasher, LeanExprHasher
from .propose import ProposalError, parse_proposal
from .run_solver import ApiRunSolver, declaration_name
from .sketch import LeanRunner, LeanSketchValidator, SketchUnavailable
from .store import ProofGraphStore

#: Where the graph and every run directory live. One workspace is one line of
#: enquiry; pointing two sessions at the same one is how a conversation and a
#: long run share a graph on purpose.
WORK_ENV = "EVO_PROOF_WORK"
#: The lake project that puts Mathlib on the search path. Absent means bare
#: `lean`, which is right for core-Lean goals and wrong for every real
#: benchmark problem.
PROJECT_ENV = "EVO_LEAN_PROJECT"


def _runner(args) -> LeanRunner:
    project = args.lean_project or os.environ.get(PROJECT_ENV)
    if project:
        return LeanRunner.mathlib(project, timeout_s=args.lean_timeout)
    return LeanRunner(timeout_s=args.lean_timeout)


def _store(args) -> ProofGraphStore:
    work = Path(args.work or os.environ.get(WORK_ENV) or ".proof")
    work.mkdir(parents=True, exist_ok=True)
    return ProofGraphStore(work / "graph.db")


def _hasher(args):
    if args.lean_identity:
        return LeanExprHasher()
    return ExactTextHasher()


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
        "attempts": [
            {"outcome": a.outcome.value, "note": a.note[:200]} for a in attempts
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

        report = controller.solve(
            goal.id, budget=args.budget, max_iterations=args.max_iterations
        )
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
        result, certification = certify(store, goal.id, runner=_runner(args))
        out = Path(args.out) if args.out else None
        if result.ok and out:
            out.write_text(result.text, encoding="utf-8")
        return {
            "ok": result.ok,
            "reason": result.reason[:2000],
            "axioms": sorted(result.axioms),
            "certification_id": certification.id,
            "written_to": str(out) if (result.ok and out) else None,
            "text": result.text if args.show_text else None,
        }
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

    runner = _runner(args)
    work = Path(args.work or os.environ.get(WORK_ENV) or ".proof")
    mode = "agentic" if args.level == "L2" else "single_shot"

    solver = _RefusesToAttack() if not solver_needed else ApiRunSolver(
        work_root=work / "runs",
        run_spec_factory=lambda out: RunSpec(
            models=(args.model,),
            proposer_backend=ComponentSpec.create(
                "proposer_backend", "proof.cli", version="v1"
            ),
            output_dir=str(out),
            seed=args.seed,
        ),
        search_profile_factory=lambda: BasicSearchProfile(
            num_trajectories=args.trajectories, proposal_mode=mode
        ),
        grade_func=make_grader(runner),
        transport=_transport(args),
        preamble=_read_preamble(args),
        level=args.level,
    )
    return ProofController(
        store,
        solver,
        # `attack` never proposes a route: decomposition is `sketch`, and it is
        # the caller's move. Keeping them apart is what lets a person see the
        # decomposition before any budget is spent on it.
        decompositions=_NoDecompositions(),
        validate_sketch=LeanSketchValidator(runner=runner),
        max_capability_attempts=args.max_attempts,
    )


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
                        help="merge alpha-equivalent lemmas (asks Lean)")
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

    p = sub.add_parser("assemble")
    p.add_argument("--goal", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--show-text", action="store_true")
    p.set_defaults(func=cmd_assemble)
    return parser


def _solver_flags(p) -> None:
    p.add_argument("--level", choices=("L1", "L2"), default="L2",
                   help="L1 one-shot; L2 an agent session that can call Lean")
    p.add_argument("--trajectories", type=int, default=1)
    p.add_argument("--max-attempts", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = args.func(args)
    except (
        ValueError, KeyError, RuntimeError, SketchUnavailable, AssemblyError
    ) as exc:
        json.dump({"error": f"{type(exc).__name__}: {exc}"}, sys.stdout)
        sys.stdout.write("\n")
        return 1
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
