"""One proof problem, end to end: a model decomposes, Lean judges, EvoHarness
solves each lemma, and the assembled result is compiled again.

    # offline -- scripted replies, real Lean, real runs, no model calls
    uv run python scripts/proof_demo.py

    # live
    export EVOHARNESS_API_BASE=...   EVOHARNESS_API_KEY=...
    uv run python scripts/proof_demo.py --live --model gpt-5.6-sol

The offline mode is not a mock of the pipeline. Lean really compiles the
sketch, `api.run()` really runs, and the final assembled proof is really
checked. Only the model's two jobs -- proposing a decomposition and writing a
lemma body -- come from a script. That is the honest boundary: everything the
framework does is exercised; only what the model would say is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evoharness import BasicSearchProfile, ComponentSpec, RunSpec  # noqa: E402
from evoharness.core import LLMResponse, LLMStopReason  # noqa: E402
from evoharness.proof.assembly import certify  # noqa: E402
from evoharness.proof.controller import ProofController  # noqa: E402
from evoharness.proof.identity import ExactTextHasher, LeanExprHasher  # noqa: E402
from evoharness.proof.propose import ModelDecompositionSource  # noqa: E402
from evoharness.proof.run_solver import ApiRunSolver, declaration_name  # noqa: E402
from evoharness.proof.sketch import LeanSketchValidator  # noqa: E402
from evoharness.proof.store import ProofGraphStore  # noqa: E402

GOAL = (
    "theorem goal (a b c : Nat) :\n"
    "    (a + b) * c = a * c + b * c\n"
    "    ∧ (a + b) + c = a + (b + c)\n"
    "    ∧ a * 0 = 0"
)

#: What a model would reply to the decomposition prompt. Used offline.
SCRIPTED_DECOMPOSITION = """```json
{
  "lemmas": [
    {"name": "lemma_distrib",
     "signature": "theorem lemma_distrib (a b c : Nat) : (a + b) * c = a * c + b * c"},
    {"name": "lemma_assoc",
     "signature": "theorem lemma_assoc (a b c : Nat) : (a + b) + c = a + (b + c)"},
    {"name": "lemma_mul_zero",
     "signature": "theorem lemma_mul_zero (a : Nat) : a * 0 = 0"}
  ],
  "parent_body": "⟨lemma_distrib a b c, lemma_assoc a b c, lemma_mul_zero a⟩"
}
```"""

#: What a model would write for each lemma body, keyed by declaration name.
SCRIPTED_BODIES = {
    "lemma_distrib": "Nat.add_mul a b c",
    "lemma_assoc": "Nat.add_assoc a b c",
    "lemma_mul_zero": "Nat.mul_zero a",
}

ALLOWED_AXIOMS = ("propext", "Quot.sound", "Classical.choice")


# --- the grader every subgoal task is scored by -------------------------------

def grade_subgoal(candidate_dir, ctx):
    """Compile the candidate and read Lean's axiom report.

    `passed` is "it compiles", not "it is proved" -- see the P-0a fixture
    grader. Fitness is graded so search has something to climb; 1.0 comes only
    from the axiom report.
    """

    from evoharness.serve import InfraError

    path = Path(candidate_dir) / "subgoal.lean"
    if not path.is_file():
        return {"fitness": 0.0, "passed": False, "fault_kind": "invalid_candidate",
                "fault": "no subgoal.lean"}
    binary = shutil.which("lean")
    if not binary:
        raise InfraError("no `lean` on PATH")

    source = path.read_text(encoding="utf-8")
    name = declaration_name(
        source.split("EDIT-REGION-BEGIN", 1)[-1].split(":=", 1)[0]
    )
    try:
        done = subprocess.run(
            [binary, path.name], cwd=path.parent, capture_output=True,
            text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {"fitness": 0.0, "passed": False, "fault_kind": "timeout",
                "fault": "compile exceeded 120s"}
    if done.returncode < 0:
        raise InfraError(f"lean killed by signal {-done.returncode}")

    output = (done.stdout or "") + (done.stderr or "")
    if done.returncode != 0:
        return {"fitness": 0.0, "passed": False, "fault_kind": "task_failure",
                "fault": "does not compile", "notes": output[:2000]}

    axioms = _axioms(output, name)
    forbidden = axioms - set(ALLOWED_AXIOMS) - {"sorryAx"}
    if forbidden:
        return {"fitness": 0.0, "passed": False, "fault_kind": "task_failure",
                "fault": f"forbidden axioms: {sorted(forbidden)}"}
    if "sorryAx" in axioms:
        # Compiles, still leans on `sorry`. Partial credit, and passed=True so
        # it can be a parent.
        return {"fitness": 0.3, "passed": True, "notes": output[:2000],
                "visible_metrics": {"proved": 0}}
    return {"fitness": 1.0, "passed": True,
            "visible_metrics": {"proved": 1, "axioms": ",".join(sorted(axioms))}}


def _axioms(output: str, name: str) -> set[str]:
    import re

    if re.search(rf"'{re.escape(name)}' does not depend on any axioms", output):
        return set()
    match = re.search(rf"'{re.escape(name)}' depends on axioms: \[([^\]]*)\]", output)
    if not match:
        return {"sorryAx"}  # no report: treat as unproved rather than proved
    return {item.strip() for item in match.group(1).split(",") if item.strip()}


# --- model plumbing -----------------------------------------------------------

def live_transport(model: str):
    from evoharness.core.llm import make_openai_responses_transport

    base = os.environ.get("EVOHARNESS_API_BASE")
    key = os.environ.get("EVOHARNESS_API_KEY")
    if not base or not key:
        raise SystemExit(
            "--live needs EVOHARNESS_API_BASE and EVOHARNESS_API_KEY"
        )
    return make_openai_responses_transport(base, key, timeout_s=300.0)


def scripted_transport():
    """Replies for the solver lane: fill the `sorry` with the known body."""

    def transport(*, messages, model, **kwargs):
        text = "\n".join(message.content for message in messages)
        name = next((n for n in SCRIPTED_BODIES if n in text), None)
        if name is None:
            # The root goal. A scripted model has nothing for it, and that is
            # the case the demo is about: direct proving fails, so the
            # controller goes looking for a decomposition.
            return LLMResponse(
                text="I cannot prove this directly.", model=model, cost=0.0,
                prompt_tokens=10, completion_tokens=10,
                stop_reason=LLMStopReason.COMPLETED,
            )
        signature = next(
            line.strip()
            for line in text.splitlines()
            if line.strip().startswith(f"theorem {name} ")
        )
        code = (
            "-- EDIT-REGION-BEGIN\n"
            f"{signature} := {SCRIPTED_BODIES[name]}\n"
            "-- EDIT-REGION-END\n\n"
            f"#print axioms {name}\n"
        )
        return LLMResponse(
            text=f"```lean\n{code}\n```", model=model, cost=0.0,
            prompt_tokens=10, completion_tokens=10,
            stop_reason=LLMStopReason.COMPLETED,
        )

    return transport


def ask_via_transport(transport, model: str):
    from evoharness.core.llm import LLMMessage, LLMToolChoice

    def ask(prompt: str) -> str:
        response = transport(
            messages=(LLMMessage(role="user", content=prompt),),
            model=model, temperature=0.2, max_tokens=8000,
            tools=(), tool_choice=LLMToolChoice(),
            parallel_tool_calls=False, timeout_s=300.0,
        )
        return response.text

    return ask


# --- the run ------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--work", default="")
    parser.add_argument(
        "--decompose-first", action="store_true",
        help="decompose the root before trying to prove it directly; the "
             "'with graph' arm, and the only way to exercise the whole chain "
             "on a goal the model could have closed in one shot",
    )
    parser.add_argument(
        "--lean-identity", action="store_true",
        help="merge alpha-equivalent lemmas (asks Lean); default is the "
             "conservative text hasher",
    )
    args = parser.parse_args()

    if not shutil.which("lean"):
        raise SystemExit("no `lean` on PATH; this demo is judged by Lean")

    work = Path(args.work or tempfile.mkdtemp(prefix="proof_demo_"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"work: {work}\n")

    transport = live_transport(args.model) if args.live else scripted_transport()
    if args.live:
        ask = ask_via_transport(transport, args.model)
    else:
        ask = lambda prompt: SCRIPTED_DECOMPOSITION  # noqa: E731

    hasher = LeanExprHasher() if args.lean_identity else ExactTextHasher()
    store = ProofGraphStore(work / "graph.db")
    try:
        root_identity = hasher.hash_many([GOAL])[0]
        root = store.upsert_goal(root_identity, GOAL)
        store.add_root(root.id, "demo")

        solver = ApiRunSolver(
            work_root=work / "runs",
            run_spec_factory=lambda out: RunSpec(
                models=(args.model,),
                proposer_backend=ComponentSpec.create(
                    "proposer_backend", "proof.demo", version="v1"
                ),
                output_dir=str(out),
                seed=1,
            ),
            search_profile_factory=lambda: BasicSearchProfile(
                num_trajectories=2, proposal_mode="single_shot"
            ),
            grade_func=grade_subgoal,
            transport=transport,
            level="L1",
        )
        source = ModelDecompositionSource(
            ask=ask, hasher=hasher, parent_name="goal"
        )
        controller = ProofController(
            store,
            solver,
            decompositions=source,
            validate_sketch=LeanSketchValidator(),
            max_capability_attempts=1,
            decompose_root_first=args.decompose_first,
        )

        print("solving...")
        report = controller.solve(root.id, budget=1000.0, max_iterations=40)
        print(json.dumps(report.__dict__, indent=2, ensure_ascii=False))
        if source.errors:
            print("proposal errors:", source.errors)
        for goal in [store.goal(root.id), *(
            store.goal(i) for d in store.decompositions_of(root.id)
            for i in d.subgoal_ids
        )]:
            for attempt in store.attempts_of(goal.id):
                print(f"  {attempt.outcome.value:16s} "
                      f"{declaration_name(goal.statement):16s} {attempt.note[:60]}")

        if not report.root_proved:
            print("\nthe graph did not close the root; nothing to assemble")
            return 1

        print("\nfinal re-verification (the only thing that certifies the root)")
        result, _ = certify(store, root.id)
        print("ok:", result.ok)
        print("axioms:", sorted(result.axioms))
        if not result.ok:
            print(result.reason)
            return 1
        (work / "assembled.lean").write_text(result.text, encoding="utf-8")
        print(f"assembled proof: {work / 'assembled.lean'}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
