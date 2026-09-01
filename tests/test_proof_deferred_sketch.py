"""一次 Lean 超时不许把目标永久卡死。

草图检查跑不起来时,分解留在 `PROPOSED` 是对的——标成 `rejected-by-verifier`
是终态、永不重访,一次基础设施故障就会永久删掉一条可行的路。

**但只做到这里是个楔子。** `PROPOSED` 算「路正在走」,于是父目标不再被直接攻;
而在真跑里发现:没有任何东西会回来复验那条路。目标就此关死,无声无息,起因只是
一次超时。

这一组钉的是那个闭环:`recover()` 顺带复验,失败就留到下一轮,而不是留到永远。
"""

import pytest

from evoharness.proof.controller import (
    FixedDecompositions,
    ProofController,
)
from evoharness.proof.graph import DecompositionStatus, GoalStatus, Outcome
from evoharness.proof.sketch import (
    Sketch,
    SketchUnavailable,
    SubgoalSpec,
    Validation,
)
from evoharness.proof.solver import StubSolver
from evoharness.proof.store import ProofGraphStore

ROOT = ("id:root", "theorem root (A B : Prop) (ha : A) (hb : B) : A ∧ B")
LEFT = ("id:left", "theorem left (A : Prop) (ha : A) : A")
RIGHT = ("id:right", "theorem right (B : Prop) (hb : B) : B")

SPLIT = Sketch(
    parent_name="root",
    parent_signature=ROOT[1],
    parent_body="⟨left A ha, right B hb⟩",
    subgoals=(
        SubgoalSpec(name="left", identity=LEFT[0], signature=LEFT[1]),
        SubgoalSpec(name="right", identity=RIGHT[0], signature=RIGHT[1]),
    ),
)


class Flaky:
    """A validator that is down for the first `outages` calls, then answers."""

    def __init__(self, outages: int, ok: bool = True):
        self.outages = outages
        self.ok = ok
        self.calls = 0

    def __call__(self, goal, sketch):
        self.calls += 1
        if self.calls <= self.outages:
            raise SketchUnavailable("lean exceeded its timeout")
        return Validation(ok=self.ok, reason="checked on a later round")


@pytest.fixture
def store(tmp_path):
    graph = ProofGraphStore(tmp_path / "graph.db")
    yield graph
    graph.close()


def build(store, validate, **kwargs):
    return ProofController(
        store,
        StubSolver({}, default=Outcome.TASK_FAILED),
        decompositions=FixedDecompositions({ROOT[0]: SPLIT}),
        validate_sketch=validate,
        **kwargs,
    )


def wedged(store):
    """A root whose only route was proposed while the checker was down."""

    root = store.upsert_goal(*ROOT)
    store.add_root(root.id, "wedge")
    validator = Flaky(outages=1)
    outcome = build(store, validator).record_decomposition(store.goal(root.id), SPLIT)
    return root, validator, outcome


# --- 楔子本身 ----------------------------------------------------------------

def test_a_checker_outage_leaves_the_route_undecided_not_rejected(store):
    """跑不起来不是裁定。标成 rejected 会永久删掉一条可能可行的路。"""

    root, _, outcome = wedged(store)

    assert outcome.deferred == 1
    assert outcome.rejected == 0
    decomposition = store.decompositions_of(root.id)[0]
    assert decomposition.status is DecompositionStatus.PROPOSED


def test_an_undecided_route_wedges_the_parent_while_its_subgoals_stay_open(store):
    """楔子的准确形状,比「父目标不可攻」更糟。

    `PROPOSED` 算「路正在走」,父目标于是不再被直接攻——但子目标已经建好了,
    照样可以攻。所以预算会花在证那些引理上,而**只有 ACCEPTED 的分解能走到
    COMPLETED**:全部证完,父目标还是关不上。

    也就是说,一次 Lean 超时买来的不是「少一条路」,是「一条会把钱花光却
    永远不结账的路」。
    """

    root, _, _ = wedged(store)
    controller = build(store, Flaky(outages=99))

    actionable = {goal.identity for goal in controller.actionable_goals()}
    assert root.identity not in actionable          # 父目标被挡住
    assert actionable == {LEFT[0], RIGHT[0]}        # 子目标照花钱

    # 把两条引理都证掉,父目标依然关不上——分解还没被采纳。
    for identity in (LEFT[0], RIGHT[0]):
        subgoal = store.goal_by_identity(identity)
        store.record_attempt(subgoal.id, Outcome.PROVED, proof_text="h")
        store.propagate(subgoal.id, max_capability_attempts=3)
    assert store.goal(root.id).status is GoalStatus.OPEN


# --- 闭环 --------------------------------------------------------------------

def test_recovery_reaches_the_verdict_the_outage_prevented(store):
    """下一轮回来复验。这是「留到下一轮」原本缺的那一半。"""

    root, validator, _ = wedged(store)
    validator.outages = 0  # 检查器恢复了

    totals = build(store, validator).revalidate_proposed()

    assert totals.accepted == 1
    decomposition = store.decompositions_of(root.id)[0]
    assert decomposition.status is DecompositionStatus.ACCEPTED
    assert "later round" in decomposition.rejected_reason


def test_a_route_the_checker_refuses_is_finally_refused(store):
    """复验不是只会说好。恢复之后判定为不合法,就该落成终态。"""

    root = store.upsert_goal(*ROOT)
    validator = Flaky(outages=1, ok=False)
    build(store, validator).record_decomposition(store.goal(root.id), SPLIT)

    totals = build(store, validator).revalidate_proposed()

    assert totals.rejected == 1
    assert store.decompositions_of(root.id)[0].status is (
        DecompositionStatus.REJECTED_BY_VERIFIER
    )


def test_a_still_dead_checker_leaves_it_for_the_round_after(store):
    """连续故障不该把路判死,只是继续等——但这次是真的会再来一次。"""

    root, validator, _ = wedged(store)
    controller = build(store, Flaky(outages=99))

    assert controller.revalidate_proposed().deferred == 1
    assert store.decompositions_of(root.id)[0].status is (
        DecompositionStatus.PROPOSED
    )
    # 关键:它仍然在待办清单上,而不是被忘掉。
    assert len(store.proposed_decompositions()) == 1


def test_solve_unwedges_the_goal_before_it_starts(store):
    """恢复是循环第一件事,所以一次卡住的会话下一次跑就自己解开。"""

    root, validator, _ = wedged(store)
    validator.outages = 0

    solver = StubSolver({}, default=Outcome.TASK_FAILED)
    controller = ProofController(
        store, solver,
        decompositions=FixedDecompositions({}),   # 不再提新分解
        validate_sketch=validator,
    )
    controller.solve(root.id, budget=50)

    # 那条卡住的路在循环开始前就被复验并采纳了,而不是继续挡着。
    assert store.decompositions_of(root.id)[0].status is (
        DecompositionStatus.ACCEPTED
    )
    # 而且循环攻的是子目标,不是在父目标上原地打转。
    assert set(solver.calls) == {LEFT[0], RIGHT[0]}


def test_a_decomposition_with_no_sketch_cannot_wait_forever(store):
    """没有草图就永远复验不了。这不是裁定,但它也永远变不成裁定,
    所以说清楚,而不是每一轮都重试一次。"""

    root = store.upsert_goal(*ROOT)
    store.add_decomposition(
        root.id, [(LEFT[0], LEFT[1]), (RIGHT[0], RIGHT[1])], sketch=None
    )

    totals = build(store, Flaky(outages=0)).revalidate_proposed()

    assert totals.rejected == 1
    decomposition = store.decompositions_of(root.id)[0]
    assert decomposition.status is DecompositionStatus.REJECTED_BY_VERIFIER
    assert "without a sketch" in decomposition.rejected_reason


# --- 花钱之前的警告 ----------------------------------------------------------

def test_a_goal_only_a_rejected_route_uses_is_refused(store):
    """Lean 判过不合法的草图,它的子目标可能是模型编的、根本不成立的命题。

    这一档和「还没判」不是一回事,措辞和后果都该不同。
    """

    from evoharness.proof.cli import _route_note

    root = store.upsert_goal(*ROOT)
    build(store, Flaky(outages=0, ok=False)).record_decomposition(
        store.goal(root.id), SPLIT
    )

    _, refusal = _route_note(store, store.goal_by_identity(LEFT[0]))

    assert refusal is not None
    assert "rejected by Lean" in refusal
    assert "may not even be true" in refusal


def test_a_goal_whose_route_is_merely_unjudged_is_refused_differently(store):
    """还没判的路说的是「检查器现在不通」,不是「这条引理是垃圾」。"""

    from evoharness.proof.cli import _route_note

    root, _, _ = wedged(store)
    _, refusal = _route_note(store, store.goal_by_identity(LEFT[0]))

    assert refusal is not None
    assert "has been judged yet" in refusal
    assert "Retry when it is" in refusal


def test_a_goal_with_no_routes_at_all_is_not_refused(store):
    """没有路引用它,不等于只有坏路引用它。根目标就住在这一档。"""

    from evoharness.proof.cli import _route_note

    root = store.upsert_goal(*ROOT)
    assert _route_note(store, store.goal(root.id)) == ([], None)


def test_an_accepted_route_is_not_refused(store):
    from evoharness.proof.cli import _route_note

    root, validator, _ = wedged(store)
    validator.outages = 0
    build(store, validator).revalidate_proposed()

    subgoal = store.goal_by_identity(LEFT[0])
    assert _route_note(store, subgoal) == ([], None)


def test_a_root_with_no_route_at_all_draws_no_warning(store):
    """还没有分解的目标不该被警告:直接攻它正是该做的事。"""

    from evoharness.proof.cli import _attack_warnings

    root = store.upsert_goal(*ROOT)
    assert _attack_warnings(store, store.goal(root.id)) == []


def test_attacking_something_already_proved_is_called_out(store):
    from evoharness.proof.cli import _attack_warnings

    root = store.upsert_goal(*ROOT)
    store.record_attempt(root.id, Outcome.PROVED, proof_text="h")
    store.propagate(root.id, max_capability_attempts=3)

    warnings = _attack_warnings(store, store.goal(root.id))
    assert any("already proved" in w for w in warnings)


def test_an_exhausted_goal_says_under_what_it_was_exhausted(store):
    """「做不到」必须带上「用什么、花了多少做不到」,否则没法判断该不该重试。"""

    from evoharness.proof.cli import _attack_warnings

    root = store.upsert_goal(*ROOT)
    for _ in range(3):
        store.record_attempt(root.id, Outcome.TASK_FAILED, cost=2.0)
    store.propagate(
        root.id, max_capability_attempts=3, budget_spent=6.0, solver_level="L2"
    )

    warnings = _attack_warnings(store, store.goal(root.id))
    assert any("exhausted at budget 6.0" in w and "'L2'" in w for w in warnings)


def test_a_stale_timeout_does_not_refuse_once_the_checker_is_back(
    store, tmp_path, monkeypatch
):
    """默认拒绝之所以安全,全在这一条。

    `cmd_attack` 在决定拒绝之前先跑一次复验。所以一条只是「当初检查器没起来」
    的路会在此刻被判掉,拒绝根本不会发生——**被挡住只可能是因为检查器现在
    仍然不通**,而不是因为它曾经不通过。

    少了这一步,默认拒绝就会把一次旧超时变成一道要人手动解开的闸。
    """

    import argparse

    from evoharness.proof import cli
    from evoharness.proof.controller import ProofController

    root, validator, _ = wedged(store)              # 路卡在 PROPOSED
    subgoal_id = store.goal_by_identity(LEFT[0]).id
    store.close()
    validator.outages = 0                            # 检查器恢复了

    def controller(args, opened, solver_needed=True):
        return ProofController(
            opened,
            StubSolver({}, default=Outcome.TASK_FAILED),
            decompositions=FixedDecompositions({}),
            validate_sketch=validator,
        )

    monkeypatch.setattr(cli, "_controller", controller)
    out = cli.cmd_attack(argparse.Namespace(
        goal=subgoal_id, work=str(tmp_path), budget=1.0, max_iterations=1,
        preamble="", allow_unaccepted_route=False,
    ))

    assert out.get("refused") is not True


def test_a_checker_that_is_still_down_does_refuse(store, tmp_path, monkeypatch):
    """反过来也要成立,否则「默认拒绝」根本没生效。"""

    import argparse

    from evoharness.proof import cli
    from evoharness.proof.controller import ProofController

    root, _, _ = wedged(store)
    subgoal_id = store.goal_by_identity(LEFT[0]).id
    store.close()
    still_down = Flaky(outages=99)

    monkeypatch.setattr(cli, "_controller", lambda args, opened, solver_needed=True:
        ProofController(
            opened, StubSolver({}, default=Outcome.TASK_FAILED),
            decompositions=FixedDecompositions({}), validate_sketch=still_down,
        ))
    out = cli.cmd_attack(argparse.Namespace(
        goal=subgoal_id, work=str(tmp_path), budget=1.0, max_iterations=1,
        preamble="", allow_unaccepted_route=False,
    ))

    assert out["refused"] is True
    assert "has been judged yet" in out["reason"]
