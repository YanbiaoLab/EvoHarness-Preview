"""控制器循环:直接证在先、分解在后,以及三条不许越过的线。

这一层把图、存储、求解器接起来,所以它是 P-3 验收真正落地的地方。用桩跑,
毫秒级,完全确定。

三条承重:
- **`infra-failed` 不许触发分解**,也不许把目标推向穷尽;
- **未经校验的分解不许被采纳**——只有 ACCEPTED 才能走到 COMPLETED;
- **循环必须停**:证完了、钱花光了、没有可攻的目标、连续基础设施故障。
"""

import pytest

from evoharness.proof.controller import FixedDecompositions, ProofController
from evoharness.proof.graph import DecompositionStatus, GoalStatus, Outcome
from evoharness.proof.sketch import Sketch, SubgoalSpec, Validation
from evoharness.proof.solver import StubSolver
from evoharness.proof.store import ProofGraphStore

ROOT = ("sha256:root", "theorem root (A B : Prop) (ha : A) (hb : B) : A ∧ B")
LEFT = ("sha256:left", "theorem left (A : Prop) (ha : A) : A")
RIGHT = ("sha256:right", "theorem right (B : Prop) (hb : B) : B")


def spec(pair, name):
    return SubgoalSpec(name=name, identity=pair[0], signature=pair[1])


def sketch_for(root_pair, subgoal_pairs, *, body="trivial"):
    """A route the controller can carry. These tests never compile anything --
    the validator is a stub -- so the Lean text only has to be well formed
    enough to store and read back."""

    return Sketch(
        parent_name="root",
        parent_signature=root_pair[1],
        parent_body=body,
        subgoals=tuple(
            spec(pair, f"sub_{index}")
            for index, pair in enumerate(subgoal_pairs)
        ),
    )


SPLIT = sketch_for(ROOT, [LEFT, RIGHT])


def accept_all(goal, subgoals):
    return Validation(ok=True)


def reject_all(goal, subgoals):
    return Validation(ok=False, reason="sketch left sorry in the main body")


@pytest.fixture
def store(tmp_path):
    graph = ProofGraphStore(tmp_path / "graph.db")
    yield graph
    graph.close()


def build(store, solver, *, decompositions=None, validate=accept_all, **kwargs):
    return ProofController(
        store,
        solver,
        decompositions=decompositions or FixedDecompositions({ROOT[0]: SPLIT}),
        validate_sketch=validate,
        **kwargs,
    )


def seeded_root(store):
    root = store.upsert_goal(*ROOT)
    store.add_root(root.id, "fixture")
    return root


# --- 直接证在先,分解在后 ----------------------------------------------------

def test_a_goal_the_solver_can_close_directly_is_never_decomposed(store):
    """LEAP 的顺序:先直接证,证不动才分解。能一次证完就不该拆。"""

    root = seeded_root(store)
    solver = StubSolver({ROOT[0]: [Outcome.PROVED]})
    report = build(store, solver).solve(root.id, budget=100)

    assert report.root_proved is True
    assert report.stopped_reason == "proved"
    assert report.decompositions_accepted == 0
    assert store.decompositions_of(root.id) == []


def test_a_failed_direct_attempt_brings_out_the_decomposition(store):
    """三个子目标全绿之后,根目标跟着绿——这是整条链的验收。"""

    root = seeded_root(store)
    solver = StubSolver({
        ROOT[0]: [Outcome.TASK_FAILED],
        LEFT[0]: [Outcome.PROVED],
        RIGHT[0]: [Outcome.PROVED],
    })
    report = build(store, solver).solve(root.id, budget=100)

    assert report.root_proved is True
    assert report.decompositions_accepted == 1
    assert store.goal(root.id).status is GoalStatus.PROVED


def test_the_parent_is_not_attacked_again_while_its_route_is_being_worked(store):
    """有活分解的目标由子目标扛着。再直接攻一遍是重复花钱,而且因为活分解
    让父目标永远 OPEN,这个循环停不下来。"""

    root = seeded_root(store)
    solver = StubSolver({
        ROOT[0]: [Outcome.TASK_FAILED],
        LEFT[0]: [Outcome.PROVED],
        RIGHT[0]: [Outcome.PROVED],
    })
    build(store, solver).solve(root.id, budget=100)

    assert solver.calls.count(ROOT[0]) == 1


# --- 承重:infra 不许改写图的形状 --------------------------------------------

def test_an_infra_failure_does_not_trigger_a_decomposition(store):
    """判题器宕机说的是这次运行,不是这个目标。

    拿它当「这条路走不通」去换分解,等于让基础设施故障重写整张图的形状——
    而那个分解本来根本不需要。
    """

    root = seeded_root(store)
    solver = StubSolver({ROOT[0]: [Outcome.INFRA_FAILED, Outcome.PROVED]})
    report = build(store, solver).solve(root.id, budget=100)

    assert report.root_proved is True
    assert report.decompositions_accepted == 0
    assert store.decompositions_of(root.id) == []


def test_an_infra_failure_does_not_exhaust_the_goal(store):
    root = seeded_root(store)
    solver = StubSolver(
        {ROOT[0]: [Outcome.INFRA_FAILED] * 4},
        default=Outcome.INFRA_FAILED,
    )
    build(store, solver, max_capability_attempts=2).solve(root.id, budget=100)

    assert store.goal(root.id).status is GoalStatus.OPEN


def test_consecutive_infra_failures_stop_the_run_not_the_goal(store):
    """对着一个死掉的判题器重试一整晚,产出是一行日志。"""

    root = seeded_root(store)
    solver = StubSolver({}, default=Outcome.INFRA_FAILED)
    report = build(store, solver, max_consecutive_infra=3).solve(
        root.id, budget=100
    )

    assert report.stopped_reason == "infra"
    assert report.attempts == 3
    assert store.goal(root.id).status is GoalStatus.OPEN


def test_one_infra_failure_does_not_poison_the_count(store):
    """连续计数要被一次真裁定清零,否则零星故障会假装成判题器挂了。"""

    root = seeded_root(store)
    solver = StubSolver({ROOT[0]: [
        Outcome.INFRA_FAILED, Outcome.TASK_FAILED,
        Outcome.INFRA_FAILED, Outcome.TASK_FAILED,
        Outcome.INFRA_FAILED, Outcome.PROVED,
    ]})
    report = build(
        store, solver, decompositions=FixedDecompositions({}),
        max_consecutive_infra=2, max_capability_attempts=9,
    ).solve(root.id, budget=100)

    assert report.stopped_reason == "proved"


# --- 承重:未经校验的分解不许被采纳 ------------------------------------------

def test_a_rejected_sketch_is_marked_rejected_by_verifier_not_accepted(store):
    """校验器说草图不合法是事实。它的子目标就算全绿,也不许让父目标变绿。"""

    root = seeded_root(store)
    solver = StubSolver({
        ROOT[0]: [Outcome.TASK_FAILED],
        LEFT[0]: [Outcome.PROVED],
        RIGHT[0]: [Outcome.PROVED],
    }, default=Outcome.TASK_FAILED)
    report = build(store, solver, validate=reject_all).solve(root.id, budget=100)

    assert report.decompositions_accepted == 0
    assert report.decompositions_rejected == 1
    assert report.root_proved is False

    decomposition = store.decompositions_of(root.id)[0]
    assert decomposition.status is DecompositionStatus.REJECTED_BY_VERIFIER
    assert "sorry" in decomposition.rejected_reason


def test_a_decomposition_that_restates_an_ancestor_is_counted_not_crashed(store):
    """LEAP 去掉审稿人之后的退化模式。被拒了,但要在报告里看得见——
    悄悄消失的话,预算烧光时没人知道钱去哪了。"""

    root = seeded_root(store)
    solver = StubSolver({}, default=Outcome.TASK_FAILED)
    controller = build(
        store, solver,
        decompositions=FixedDecompositions({ROOT[0]: sketch_for(ROOT, [ROOT])}),
        max_capability_attempts=2,
    )
    report = controller.solve(root.id, budget=100)

    assert report.cycles_refused == 1
    assert store.decompositions_of(root.id) == []


# --- 终止 --------------------------------------------------------------------

def test_the_loop_stops_when_the_budget_runs_out(store):
    root = seeded_root(store)
    solver = StubSolver({}, default=Outcome.TASK_FAILED, cost_per_attempt=4.0)
    report = build(
        store, solver, decompositions=FixedDecompositions({}),
        max_capability_attempts=0,
    ).solve(root.id, budget=10.0)

    assert report.stopped_reason == "budget"
    assert report.spent <= 12.0


def test_the_loop_stops_when_every_goal_is_exhausted(store):
    """穷尽之后没有可攻的目标了,不该空转到 max_iterations。"""

    root = seeded_root(store)
    solver = StubSolver({}, default=Outcome.TASK_FAILED)
    report = build(
        store, solver, decompositions=FixedDecompositions({}),
        max_capability_attempts=2,
    ).solve(root.id, budget=100)

    assert report.stopped_reason == "no_actionable_goal"
    assert store.goal(root.id).status is GoalStatus.EXHAUSTED


def test_exhaustion_records_the_solver_it_was_decided_under(store):
    """「做不到」必须带上「用什么做不到」,否则换求解器之后无从重开。"""

    root = seeded_root(store)
    solver = StubSolver({}, default=Outcome.TASK_FAILED)
    build(
        store, solver, decompositions=FixedDecompositions({}),
        max_capability_attempts=2,
    ).solve(root.id, budget=100)

    exhausted = store.goal(root.id)
    assert exhausted.exhausted_at_solver == "stub"
    assert exhausted.exhausted_at_budget == pytest.approx(2.0)


# --- 账本是算出来的 ----------------------------------------------------------

def test_a_resumed_controller_starts_from_the_spend_already_on_disk(store):
    """总账不另存。第二个控制器读到的是同一本账,因为它是 attempts 的和。"""

    root = seeded_root(store)
    first = StubSolver({}, default=Outcome.TASK_FAILED, cost_per_attempt=3.0)
    build(
        store, first, decompositions=FixedDecompositions({}),
        max_capability_attempts=0,
    ).solve(root.id, budget=9.0)

    spent_before = store.total_cost()
    assert spent_before >= 9.0

    second = StubSolver({}, default=Outcome.TASK_FAILED, cost_per_attempt=3.0)
    report = build(
        store, second, decompositions=FixedDecompositions({}),
        max_capability_attempts=0,
    ).solve(root.id, budget=9.0)

    # 预算已经花完了,第二个控制器一次都不该攻。
    assert report.stopped_reason == "budget"
    assert second.calls == []


# --- 作用域 ------------------------------------------------------------------

def test_solving_one_goal_does_not_spend_on_an_unrelated_problem(store):
    """一个工作区装着几道题是刻意的设计,所以「攻这个目标」必须只攻这个。

    实跑抓到的:`actionable_goals()` 原本返回整个工作区所有开着的目标,于是攻
    一条引理顺手把它的兄弟也证了,并且把账记在第一条引理的预算上。
    """

    root = seeded_root(store)
    other = store.upsert_goal("sha256:unrelated", "theorem unrelated : C")
    store.add_root(other.id, "另一道题")

    solver = StubSolver({}, default=Outcome.TASK_FAILED)
    build(
        store, solver, decompositions=FixedDecompositions({}),
        max_capability_attempts=1,
    ).solve(root.id, budget=100)

    assert solver.calls == [ROOT[0]]
    assert store.goal(other.id).status is GoalStatus.OPEN


def test_subgoals_of_a_fresh_decomposition_enter_the_scope(store):
    """作用域是分解之后才长出来的,所以它必须在分解落地后重算——否则新子目标
    永远进不了候选集,循环立刻报「没有可攻的目标」。"""

    root = seeded_root(store)
    solver = StubSolver({
        ROOT[0]: [Outcome.TASK_FAILED],
        LEFT[0]: [Outcome.PROVED],
        RIGHT[0]: [Outcome.PROVED],
    })
    report = build(store, solver).solve(root.id, budget=100)

    assert report.root_proved is True
    assert LEFT[0] in solver.calls and RIGHT[0] in solver.calls
