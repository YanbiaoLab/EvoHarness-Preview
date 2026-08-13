"""Research Layer I-0: named cross-task contract tests.

Each test pins ONE invariant from todo/research_layer.md §I-0. Unlike the
per-module suites, these are organized by contract, not by implementation:
if a refactor breaks a test here, it broke a research-governance guarantee,
not a code detail.

Invariants that only become expressible at I-2 (evidence envelope, fault
taxonomy, score namespace) are xfail(strict=True) placeholders: the moment
the module lands and they pass, pytest fails on the unexpected pass and
forces the marker off — the placeholder becomes a real contract test.
"""

from pathlib import Path

import pytest

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
from evoharness.core.checkpoint import config_fingerprint
from evoharness.core.remote import EvalInfraError
from evoharness.serve import InfraError
from evoharness.runtime import WorkspaceGradeFnGrader

INITIAL = """# EDIT-REGION-BEGIN
x = 0
x += 1
# EDIT-REGION-END
print(x)
"""


# ---------------------------------------------------------------------------
# 契约:基础设施失败不是任务失败(grader 边界,eval_protocol §5)
# ---------------------------------------------------------------------------


def _seed_candidate():
    from evoharness.core import Candidate

    return Candidate(
        id="c0", code=INITIAL, generation=0, parent_id=None,
        island_idx=0, operator="seed",
    )


def test_infra_error_is_raised_not_recorded_as_failure(tmp_path):
    """A dependency outage must escape as EvalInfraError — never be coerced
    into EvalReport(passed=False), which would enter the population as
    (fake) negative evidence about the candidate."""

    def grade_func(candidate_dir: Path, ctx):
        raise InfraError("judge backend 503")

    grader = WorkspaceGradeFnGrader(grade_func)
    with pytest.raises(EvalInfraError):
        grader.grade(_seed_candidate(), tmp_path)


def test_candidate_defect_is_a_verdict_not_an_infra_error(tmp_path):
    """The other side of the same boundary: an uncaught exception from the
    candidate's own behaviour IS an evaluation verdict (passed=False), so it
    must enter the population and trigger repair — not be retried as infra."""

    def grade_func(candidate_dir: Path, ctx):
        raise RuntimeError("candidate blew up")

    grader = WorkspaceGradeFnGrader(grade_func)
    report = grader.grade(_seed_candidate(), tmp_path)
    assert report.passed is False
    assert report.stage_reached == 0


# ---------------------------------------------------------------------------
# 契约:基础设施失败不形成假设的负面证据(loop 边界)
# ---------------------------------------------------------------------------


class OutageOnThirdChild:
    def __init__(self):
        self.children_seen = 0

    def grade(self, cand, workdir: Path) -> EvalReport:
        if cand.operator != "seed":
            self.children_seen += 1
            if self.children_seen == 3:
                raise EvalInfraError("eval service down")
        return EvalReport(fitness=1.0, passed=True)


def _rewrite_transport(messages, model, **kw):
    code = f"# EDIT-REGION-BEGIN\nx = 0\nx += 1\nx += 1\n# EDIT-REGION-END\nprint(x)\n"
    return LLMResponse(
        text=f"TITLE: t\nSUMMARY: s\n```python\n{code}```",
        model=model, cost=0.001,
    )


def test_infra_dropped_candidate_never_enters_population(tmp_path):
    """No verdict → no evidence: the dropped candidate is absent from the
    store, uncounted as an evaluation, and visible only in run history."""
    cfg = SearchConfig(
        num_generations=5, operators=["rewrite"], operator_probs=[1.0], seed=7,
    )
    pop_cfg = PopulationConfig(num_islands=1)
    store = PopulationStore(pop_cfg)
    loop = SearchLoop(
        cfg=cfg, pop_cfg=pop_cfg, store=store, grader=OutageOnThirdChild(),
        llm=LLMClient(transport=_rewrite_transport, sleep=lambda s: None),
        prompt_builder=PromptBuilder("goal"),
        parent_selector=make_parent_selector(pop_cfg),
        inspiration_selector=InspirationSelector(pop_cfg),
        model_router=StaticRouter(["mock-model"]),
        workdir=tmp_path,
    )
    report = loop.run(INITIAL)

    dropped = [h for h in report.history if h["status"] == "infra_error"]
    assert len(dropped) == 1
    assert store.count() == 1 + 4          # seed + 4 graded children
    assert report.evaluations == 1 + 4     # the outage was not an evaluation


# ---------------------------------------------------------------------------
# 契约:resume 不允许改变实验身份
# ---------------------------------------------------------------------------


def test_any_config_change_changes_the_fingerprint():
    """Experiment identity is the fingerprint over effective configuration;
    editing any frozen field must produce a new identity. (Loop-level
    refusal is pinned by test_checkpoint.py::test_resume_refuses_config_change;
    this pins the identity function itself.)"""
    base = SearchConfig(num_generations=5, seed=7)
    same = SearchConfig(num_generations=5, seed=7)
    changed = SearchConfig(num_generations=6, seed=7)

    assert config_fingerprint(base) == config_fingerprint(same)
    assert config_fingerprint(base) != config_fingerprint(changed)


# ---------------------------------------------------------------------------
# I-2 契约(原占位,evoharness.evaluation 落地后转正)
# ---------------------------------------------------------------------------


def test_missing_is_not_task_failure():
    """未运行/缺失与任务失败必须是不同状态,不能都编码为 passed=False。"""
    from evoharness.evaluation.faults import FaultKind

    assert FaultKind.MISSING is not FaultKind.TASK_FAILURE
    assert FaultKind.INFRA_ERROR is not FaultKind.TASK_FAILURE
    assert FaultKind.UNKNOWN is not FaultKind.TASK_FAILURE


def test_partial_coverage_cannot_pose_as_complete():
    """部分覆盖的证据不能生成冒充完整观察的可比较分数。"""
    from evoharness.evaluation.evidence import Coverage

    partial = Coverage(planned_units=100, executed_units=20, trustworthy_units=20)
    assert partial.complete is False


def test_cross_namespace_comparison_is_a_type_error():
    """不同评测协议产生的分数不可直接比较——比较动作本身必须报错,
    而不是静默返回一个数。(run 内版本冻结的现行保证由
    test_remote_grader.py::test_task_version_drift_raises 钉住。)"""
    from evoharness.evaluation.namespace import ScoreNamespace

    a = ScoreNamespace(criterion_hash="c1", measurement_hash="m1",
                       evaluator_hash="e1", universe_hash="u1")
    b = ScoreNamespace(criterion_hash="c1", measurement_hash="m2",
                       evaluator_hash="e1", universe_hash="u1")
    with pytest.raises(Exception):
        a.require_comparable(b)
