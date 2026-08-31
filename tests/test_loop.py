"""A2 acceptance test: the full loop runs 20 generations on a mock grader
(no real LLM; fitness = a simple function of the code)."""

from pathlib import Path

import numpy as np

from evoharness.core import (
    EvalReport,
    InspirationSelector,
    LLMClient,
    LLMResponse,
    PopulationConfig,
    PopulationStore,
    PromptBuilder,
    SearchConfig,
    SearchLoop,
    StaticRouter,
    make_parent_selector,
)
from evoharness.core.workspace import GitWorkspace
from evoharness.core.loop import RunReport
from evoharness.core.population import Candidate

INITIAL = """# EDIT-REGION-BEGIN
x = 0
x += 1
# EDIT-REGION-END
print(x)
"""


class MockGrader:
    """fitness = number of `x += 1` lines; always passes."""

    def grade(self, cand, workdir: Path) -> EvalReport:
        return EvalReport(
            fitness=float(cand.code.count("x += 1")),
            passed=True,
            visible_metrics={"increments": cand.code.count("x += 1")},
            eval_cost_usd=0.001,
        )


def make_rewrite_transport():
    """Each call returns a rewrite adding one more increment line."""
    counter = {"n": 1}

    def transport(messages, model, **kw):
        counter["n"] += 1
        body = "x = 0\n" + "x += 1\n" * counter["n"]
        code = f"# EDIT-REGION-BEGIN\n{body}# EDIT-REGION-END\nprint(x)\n"
        text = (
            f"TITLE: add increment {counter['n']}\n"
            f"SUMMARY: one more increment\n"
            f"```python\n{code}```"
        )
        return LLMResponse(text=text, model=model, cost=0.002)

    return transport


def build_loop(transport, operators, probs, tmp_path, generations=20):
    cfg = SearchConfig(
        num_generations=generations,
        operators=operators,
        operator_probs=probs,
        seed=7,
    )
    pop_cfg = PopulationConfig(num_islands=2)
    store = PopulationStore(pop_cfg)
    return (
        SearchLoop(
            cfg=cfg,
            pop_cfg=pop_cfg,
            store=store,
            grader=MockGrader(),
            llm=LLMClient(transport=transport, sleep=lambda s: None),
            prompt_builder=PromptBuilder("maximize increments"),
            parent_selector=make_parent_selector(pop_cfg),
            inspiration_selector=InspirationSelector(pop_cfg),
            model_router=StaticRouter(["mock-model"]),
            workdir=tmp_path,
        ),
        store,
    )


def test_loop_20_generations_rewrite(tmp_path):
    loop, store = build_loop(
        make_rewrite_transport(), ["rewrite"], [1.0], tmp_path
    )
    report = loop.run(INITIAL)

    assert report.generations_completed == 20
    assert report.evaluations == 21  # seed + 20 children
    assert report.stopped_reason == "completed"
    assert report.best_fitness > 1.0  # improved over the seed
    assert store.count() == 22  # 2 seed copies (one per island) + 20 children
    assert report.total_llm_cost > 0 and report.total_eval_cost > 0

    # lineage integrity: every child's parent exists
    for cand in store.all_candidates():
        if cand.parent_id:
            assert store.get(cand.parent_id) is not None
    ok = [h for h in report.history if h["status"] == "ok"]
    assert len(ok) == 20


def test_loop_revise_path(tmp_path):
    def revise_transport(messages, model, **kw):
        text = (
            "TITLE: touch\nSUMMARY: rewrite same line\n"
            "<<<<<<< ORIGINAL\nx = 0\n=======\nx = 0\n>>>>>>> UPDATED\n"
        )
        return LLMResponse(text=text, model=model, cost=0.001)

    loop, store = build_loop(
        revise_transport, ["revise"], [1.0], tmp_path, generations=5
    )
    report = loop.run(INITIAL)
    assert report.generations_completed == 5
    revised = [c for c in store.all_candidates() if c.operator == "revise"]
    assert len(revised) == 5


def test_loop_stops_on_budget(tmp_path):
    class TinyBudget:
        def __init__(self):
            self.spent = 0.0

        def charge(self, usd):
            self.spent += usd

        def should_stop(self):
            return self.spent >= 0.01

    loop, _ = build_loop(make_rewrite_transport(), ["rewrite"], [1.0], tmp_path)
    loop.budget = TinyBudget()
    report = loop.run(INITIAL)
    assert report.stopped_reason == "budget"
    assert report.generations_completed < 20


def test_loop_survives_malformed_llm_output(tmp_path):
    calls = {"n": 0}

    def flaky_transport(messages, model, **kw):
        calls["n"] += 1
        if calls["n"] % 2 == 1:
            return LLMResponse(text="garbage with no code", model=model)
        return make_rewrite_transport()(
            messages=messages,
            model=model,
            **kw,
        )

    loop, store = build_loop(
        flaky_transport, ["rewrite"], [1.0], tmp_path, generations=6
    )
    report = loop.run(INITIAL)
    # malformed outputs are retried within max_op_resamples, loop never crashes
    assert report.generations_completed == 6
    assert store.count() >= 2


def test_loop_repair_flow(tmp_path):
    """A failing candidate triggers the repair operator next generation."""

    class PickyGrader:
        def grade(self, cand, workdir):
            broken = "# broken" in cand.code
            return EvalReport(
                fitness=float(cand.code.count("x += 1")),
                passed=not broken,
                fault="marker found" if broken else None,
                stderr_log="Traceback: broken" if broken else "",
            )

    calls = {"n": 0}

    def transport(messages, model, **kw):
        calls["n"] += 1
        system = messages[0].content
        if "FAILED" in system:  # repair prompt
            code = "# EDIT-REGION-BEGIN\nx = 0\nx += 1\n# EDIT-REGION-END\n"
        elif calls["n"] == 1:  # first proposal is broken
            code = "# EDIT-REGION-BEGIN\nx = 0\n# broken\n# EDIT-REGION-END\n"
        else:
            code = "# EDIT-REGION-BEGIN\nx = 0\nx += 1\nx += 1\n# EDIT-REGION-END\n"
        return LLMResponse(
            text=f"TITLE: t\nSUMMARY: s\n```python\n{code}```",
            model=model,
        )

    loop, store = build_loop(transport, ["rewrite"], [1.0], tmp_path, generations=4)
    loop.grader = PickyGrader()
    report = loop.run(INITIAL)

    repairs = [c for c in store.all_candidates() if c.operator == "repair"]
    assert len(repairs) >= 1
    assert repairs[0].passed
    failed = [h for h in report.history if h["status"] == "failed"]
    assert len(failed) >= 1


def test_loop_novelty_rejection_path(tmp_path):
    """With a constant embedding every follow-up proposal is rejected."""
    from evoharness.core import NoveltyGate

    loop, store = build_loop(
        make_rewrite_transport(), ["rewrite"], [1.0], tmp_path, generations=4
    )
    loop.novelty_gate = NoveltyGate(
        embed_fn=lambda code: [1.0, 0.0], threshold=0.99, mode="similarity"
    )
    report = loop.run(INITIAL)

    assert report.novelty_rejections >= 1
    skipped = [h for h in report.history if h["status"] == "skipped"]
    accepted = [h for h in report.history if h["status"] == "ok"]
    # first child accepted (nothing embedded yet), later ones rejected
    assert len(accepted) >= 1 and len(skipped) >= 1


# 在文件顶部 imports 区补:
#   from evoharness.core import Candidate
#   from evoharness.core.loop import RunReport
#   from evoharness.core.workspace import GitWorkspace


def test_git_parent_child_inherits_workspace(tmp_path):
    """Collapse-bug vaccine: a git parent's child must stay a git genome and
    keep its sibling files (pre-fix: child silently became a single file)."""
    loop ,_ = build_loop(
        make_rewrite_transport(), ["rewrite"], [1.0], tmp_path, generations=1
    )
    ws = GitWorkspace(base_files={"main.py": INITIAL, "util.py": "HELPER = 1\n"})
    seed = Candidate(
        id="gitseed",
        code=ws.serialize(),
        generation=0,
        parent_id=None,
        island_idx=0,
        operator="seed",
        workspace_kind="git",
        report=EvalReport(fitness=1.0, passed=True),
    )
    loop.store.seed_all_islands(seed)

    child = loop._propose(1, RunReport())
    assert child is not None
    assert child.workspace_kind == "git"  # THE assertion: no collapse
    cw = child.workspace
    assert isinstance(cw, GitWorkspace)
    assert len(cw.patches) == 1  # exactly one lineage step
    rebuilt = cw.materialize(tmp_path / "check")
    assert (rebuilt / "util.py").read_text() == "HELPER = 1\n"  # sibling survives
    assert "x += 1" in cw.main_text()  # mutation landed in the main file

def test_proposal_workspace_lane_bypasses_lens(tmp_path):
    """Lane-2 vaccine: when a proposer delivers a complete workspace (the M3
    agent path), the loop must adopt it verbatim — kind, genome bytes and
    metadata all come from the proposal, and the single-file lens stays out."""
    from evoharness.core.proposer import Proposal, ProposeResult

    delivered = GitWorkspace(
        base_files={"main.py": "x = 1\n", "helper.py": "H = 2\n"}
    )

    class FakeProposer:
        def propose(self, operator, parent_code, system, user):
            return ProposeResult(
                Proposal(
                    "x = 1\n",
                    "t",
                    "s",
                    "m",
                    workspace=delivered,
                    metadata={
                        "proposal_id": "proposal-1",
                        "trace_path": "/run/agent_sessions/proposal-1",
                    },
                ),
                llm_cost=0.0,
                attempts=1,
            )

    loop, _ = build_loop(
        make_rewrite_transport(), ["rewrite"], [1.0], tmp_path, generations=1
    )
    loop.proposer = FakeProposer()
    seed = Candidate(
        id="fileseed",
        code=INITIAL,
        generation=0,
        parent_id=None,
        island_idx=0,
        operator="seed",
        report=EvalReport(fitness=1.0, passed=True),
    )
    loop.store.seed_all_islands(seed)

    child = loop._propose(1, RunReport())
    assert child is not None
    # a FILE parent produced a GIT child: the kind comes from what the
    # proposal delivered, not from the parent
    assert child.workspace_kind == "git"
    assert child.code == delivered.serialize()  # byte-for-byte, lens untouched
    assert child.change_title == "t"  # metadata flows from the proposal too
    # A subset, not the whole dict: this parent is a seed copy, so the loop
    # also stamps lineage_parent_id, and asserting equality here made an
    # unrelated test fail for a correct change.
    assert child.metadata.items() >= {
        "proposal_id": "proposal-1",
        "trace_path": "/run/agent_sessions/proposal-1",
    }.items()


def test_failed_proposal_trace_is_preserved_in_run_history(tmp_path):
    from evoharness.core.proposer import ProposeResult

    seen = {}

    class FailingProposer:
        def propose(self, operator, parent, system, user):
            seen["parent_id"] = parent.id
            return ProposeResult(
                proposal=None,
                llm_cost=0.25,
                attempts=2,
                failure_reason="backend-error",
                trace_path="/run/agent_sessions/proposal-1",
            )

    loop, _ = build_loop(
        make_rewrite_transport(),
        ["rewrite"],
        [1.0],
        tmp_path,
        generations=1,
    )
    loop.proposer = FailingProposer()
    seed = Candidate(
        id="seed",
        code=INITIAL,
        generation=0,
        parent_id=None,
        island_idx=0,
        operator="seed",
        report=EvalReport(fitness=1.0, passed=True),
    )
    loop.store.seed_all_islands(seed)
    report = RunReport()

    assert loop._propose(1, report) is None
    assert report.proposals_failed == 1
    assert report.total_llm_cost == 0.25
    assert report.history[-1] == {
        "generation": 1,
        "status": "proposal_failed",
        "parent_id": seen["parent_id"],
        "operator": "rewrite",
        "failure_reason": "backend-error",
        "trace_path": "/run/agent_sessions/proposal-1",
        "attempts": 2,
        "llm_cost": 0.25,
    }


def test_multi_file_evolution_smoke(tmp_path):
    """M2.5 maiden voyage: git seed -> real PromptBuilder renders the whole
    workspace as FILE blocks -> fake LLM answers in the same format -> the
    child genome gains an LLM-created file, siblings intact."""
    from evoharness.core import LLMResponse

    seen = {}

    def transport(messages, model, **kw):
        seen["user"] = messages[-1].content
        return LLMResponse(
            "TITLE: split\nSUMMARY: helper\n"
            "### FILE: main.py\n```python\n# EDIT-REGION-BEGIN\n"
            "import helper\nx = helper.n()\n# EDIT-REGION-END\nprint(x)\n```\n"
            "### FILE: helper.py\n```python\ndef n():\n    return 2\n```\n",
            model, cost=0.002,
        )

    loop, _ = build_loop(transport, ["rewrite"], [1.0], tmp_path, generations=1)
    ws = GitWorkspace(base_files={"main.py": INITIAL, "util.py": "HELPER = 1\n"})
    seed = Candidate(id="gitseed", code=ws.serialize(), generation=0,
                     parent_id=None, island_idx=0, operator="seed",
                     workspace_kind="git",
                     report=EvalReport(fitness=1.0, passed=True))
    loop.store.seed_all_islands(seed)

    child = loop._propose(1, RunReport())
    assert child is not None and child.workspace_kind == "git"
    # the input side worked: whole workspace + format spec reached the LLM
    assert "### FILE: util.py" in seen["user"]
    assert "Response format (multi-file workspace)" in seen["user"]
    # the output side worked: LLM-created file landed, sibling survived
    root = child.workspace.materialize(tmp_path / "check")
    assert (root / "helper.py").read_text() == "def n():\n    return 2\n"
    assert (root / "util.py").read_text() == "HELPER = 1\n"


def test_failed_attempts_ledger_reaches_the_prompt(tmp_path):
    """试过而没涨分的改动要进提案上下文。

    参考程序那两条通道(archive / top_k)都按**高分**选,所以一个分数原地踏步的
    run 里失败尝试对提案器完全不可见,它会一遍遍重提同一类改动,每次付一整轮评测。

    实测(ETP Austin 线,2026-08-31):八个候选、五个不同旋钮,全是 28/100,
    而且解出的是完全相同的 28 行。提案器当时看得见的只有种子和「top」,
    两者都是 0.28 —— 没有任何东西告诉它这五条路已经走过。

    判据是「没有严格超过父本」而不是「分数低」:持平才是原地踏步的 run 里的
    主要失败形态,那八个候选里一个「低分」的都没有。
    """
    from evoharness.core.interfaces import MutationContext
    from evoharness.core.operators import PromptBuilder

    from conftest import make_candidate

    parent = make_candidate("p", 0.28)
    ctx = MutationContext(
        parent=parent,
        archive_inspirations=[],
        top_k_inspirations=[],
        operator="revise",
        generation=3,
        failed_attempts=(
            ("Bounded 16-Rule CNF Saturation", 0.0),
            ("Retain larger CNF completion prefixes", 0.0),
            ("Widen v13 target-witness pool", -0.01),
        ),
    )
    builder = PromptBuilder(task_sys_msg="")
    text = builder._history(ctx)
    assert "Already tried, no gain" in text
    assert "Bounded 16-Rule CNF Saturation — +0.0000" in text
    assert "Widen v13 target-witness pool — -0.0100" in text
    # 没有失败记录时不留空节 —— 一个只有标题的小节是纯噪声。
    assert builder._history(replace_failed(ctx, ())) == ""


def replace_failed(ctx, attempts):
    import dataclasses
    return dataclasses.replace(ctx, failed_attempts=attempts)
