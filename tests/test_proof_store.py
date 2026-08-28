"""图存储:记忆化、原子性、传播、租约。

这一层做对了图才是图。四组断言:

- **记忆化**——同一个身份只能是一个节点。这是树和图的全部差别。
- **原子性**——分解要么整个落盘,要么一点不落。
- **传播**——共享引理证完之后,每一个等它的父目标都要收到。
- **租约**——两个 worker 抢同一个目标,只能有一个赢;而且租约不许碰 status。
"""

import sqlite3

import pytest

from evoharness.proof.graph import (
    CycleError,
    DecompositionStatus,
    GoalStatus,
    Outcome,
)
from evoharness.proof import store as proof_store
from evoharness.proof.store import ProofGraphStore

ROOT = ("sha256:root", "theorem root : A ∧ B := sorry")
LEFT = ("sha256:left", "theorem left : A := sorry")
RIGHT = ("sha256:right", "theorem right : B := sorry")


@pytest.fixture
def store(tmp_path):
    graph = ProofGraphStore(tmp_path / "graph.db")
    yield graph
    graph.close()


def accepted_split(store, goal_id, subgoals=(LEFT, RIGHT)):
    """一个已通过草图校验的分解——只有 ACCEPTED 才允许走到 COMPLETED。"""

    decomposition = store.add_decomposition(goal_id, list(subgoals))
    store.set_decomposition_status(
        decomposition.id, DecompositionStatus.ACCEPTED
    )
    return store.decomposition(decomposition.id)


# --- 记忆化 ------------------------------------------------------------------

def test_one_identity_is_one_node(store):
    """身份相同的两个目标在物理上无法成为两行——这是 UNIQUE 约束在兜底。"""

    first = store.upsert_goal(*ROOT)
    second = store.upsert_goal(ROOT[0], "写法不同但身份相同的同一个命题")

    assert first.id == second.id
    # 先到的陈述留下。身份说它们是同一个命题,重写文本只会让谱系更难读。
    assert second.statement == ROOT[1]


def test_a_shared_lemma_is_reused_across_branches(store):
    """两个分解都要同一条引理时,第二个应该捡到第一个的成果,而不是重证。

    这条就是 LEAP 图记忆化消融买到的东西。少了它,那两个 40.0% -> 56.7%
    的百分点一个都拿不到。
    """

    root = store.upsert_goal(*ROOT)
    other = store.upsert_goal("sha256:other", "theorem other : C := sorry")
    accepted_split(store, root.id, (LEFT, RIGHT))
    accepted_split(store, other.id, (LEFT, ("sha256:third", "theorem t : D := sorry")))

    shared = store.goal_by_identity(LEFT[0])
    assert shared is not None
    assert sorted(store.parents_of(shared.id)) == sorted([root.id, other.id])


# --- 原子性与无环 ------------------------------------------------------------

def test_a_decomposition_that_would_cycle_never_lands(store):
    """被拒的分解不许留下半张图:子目标、分解行、边,一样都不该出现。"""

    root = store.upsert_goal(*ROOT)
    accepted_split(store, root.id)
    left = store.goal_by_identity(LEFT[0])

    with pytest.raises(CycleError):
        # 子目标复述了祖父目标——LEAP 去掉审稿人之后的退化模式。
        store.add_decomposition(left.id, [ROOT, ("sha256:new", "theorem n : E := sorry")])

    assert store.goal_by_identity("sha256:new") is None
    assert store.decompositions_of(left.id) == []


def test_a_failure_midway_through_writing_rolls_the_whole_thing_back(
    store, monkeypatch
):
    """上一个用例是在事务**开始前**就拒了,证明不了回滚。这个才是。

    注入点选在第二个子目标的 id 生成:那一刻第一个子目标已经写进事务了。
    如果不回滚,图上会留下一个孤儿目标,而且没有任何地方会说这件事。
    """

    root = store.upsert_goal(*ROOT)
    calls = {"n": 0}
    real = proof_store._new_id

    def explode(prefix: str) -> str:
        calls["n"] += 1
        if calls["n"] >= 3:  # dec, 第一个 goal, 然后炸
            raise sqlite3.OperationalError("disk went away")
        return real(prefix)

    monkeypatch.setattr(proof_store, "_new_id", explode)

    with pytest.raises(sqlite3.OperationalError):
        store.add_decomposition(root.id, [LEFT, RIGHT])

    monkeypatch.undo()
    assert store.decompositions_of(root.id) == []
    assert store.goal_by_identity(LEFT[0]) is None
    assert store.goal_by_identity(RIGHT[0]) is None


def test_an_empty_decomposition_is_refused(store):
    """`all([])` 是 True。空分解要是能落盘,它就是一台凭空造证明的机器。"""

    root = store.upsert_goal(*ROOT)
    with pytest.raises(ValueError, match="vacuously"):
        store.add_decomposition(root.id, [])


def test_ancestors_are_transitive_not_just_the_parent(store):
    root = store.upsert_goal(*ROOT)
    accepted_split(store, root.id)
    left = store.goal_by_identity(LEFT[0])
    accepted_split(store, left.id, (("sha256:deep", "theorem d : F := sorry"),))
    deep = store.goal_by_identity("sha256:deep")

    assert store.ancestor_identities(deep.id) == {ROOT[0], LEFT[0]}


# --- 传播 --------------------------------------------------------------------

def test_proving_every_subgoal_closes_the_parent(store):
    root = store.upsert_goal(*ROOT)
    accepted_split(store, root.id)

    for identity, _ in (LEFT, RIGHT):
        subgoal = store.goal_by_identity(identity)
        store.record_attempt(subgoal.id, Outcome.PROVED, proof_text="by trivial")
        store.propagate(subgoal.id, max_capability_attempts=3)

    assert store.goal(root.id).status is GoalStatus.PROVED


def test_one_open_subgoal_keeps_the_parent_open(store):
    root = store.upsert_goal(*ROOT)
    accepted_split(store, root.id)

    left = store.goal_by_identity(LEFT[0])
    store.record_attempt(left.id, Outcome.PROVED, proof_text="by trivial")
    store.propagate(left.id, max_capability_attempts=3)

    assert store.goal(root.id).status is GoalStatus.OPEN


def test_a_shared_lemma_closing_reaches_every_parent_waiting_on_it(store):
    """DAG 而不是树:传播必须往上走到所有父目标,不是一条链。"""

    root = store.upsert_goal(*ROOT)
    other = store.upsert_goal("sha256:other", "theorem other : C := sorry")
    accepted_split(store, root.id, (LEFT,))
    accepted_split(store, other.id, (LEFT,))

    left = store.goal_by_identity(LEFT[0])
    store.record_attempt(left.id, Outcome.PROVED, proof_text="by trivial")
    store.propagate(left.id, max_capability_attempts=3)

    assert store.goal(root.id).status is GoalStatus.PROVED
    assert store.goal(other.id).status is GoalStatus.PROVED


def test_an_infra_failure_never_exhausts_a_goal(store):
    """判题器宕机重复二十次,也不是「这题做不出来」。"""

    root = store.upsert_goal(*ROOT)
    for _ in range(20):
        store.record_attempt(root.id, Outcome.INFRA_FAILED)
    store.propagate(root.id, max_capability_attempts=3)

    assert store.goal(root.id).status is GoalStatus.OPEN


def test_exhaustion_records_the_budget_it_was_decided_under(store):
    """不记下来,加预算 resume 之后没有依据重开,图会永远绕着这个节点走。"""

    root = store.upsert_goal(*ROOT)
    for _ in range(3):
        store.record_attempt(root.id, Outcome.TASK_FAILED)
    store.propagate(
        root.id, max_capability_attempts=3, budget_spent=12.5, solver_level="L2"
    )

    exhausted = store.goal(root.id)
    assert exhausted.status is GoalStatus.EXHAUSTED
    assert exhausted.exhausted_at_budget == 12.5
    assert exhausted.exhausted_at_solver == "L2"


def test_reopening_clears_the_exhaustion_context(store):
    """穷尽的理由随着穷尽本身一起作废,否则读到的是过期的裁定。"""

    root = store.upsert_goal(*ROOT)
    for _ in range(3):
        store.record_attempt(root.id, Outcome.TASK_FAILED)
    store.propagate(root.id, max_capability_attempts=3, budget_spent=12.5)

    store.propagate(root.id, max_capability_attempts=10)

    reopened = store.goal(root.id)
    assert reopened.status is GoalStatus.OPEN
    assert reopened.exhausted_at_budget is None


# --- 租约 --------------------------------------------------------------------

def test_two_workers_cannot_hold_one_goal(store):
    root = store.upsert_goal(*ROOT)

    assert store.claim(root.id, "worker-a", ttl_s=60, now=1000.0) is True
    assert store.claim(root.id, "worker-b", ttl_s=60, now=1000.0) is False


def test_an_expired_lease_can_be_taken_over(store):
    """持有者崩了之后,目标不能永远卡住。"""

    root = store.upsert_goal(*ROOT)
    store.claim(root.id, "worker-a", ttl_s=60, now=1000.0)

    assert store.claim(root.id, "worker-b", ttl_s=60, now=1100.0) is True


def test_a_lease_does_not_touch_status(store):
    """租约是调度状态。它一旦影响语义状态,两者就再也分不开了。"""

    root = store.upsert_goal(*ROOT)
    store.claim(root.id, "worker-a", ttl_s=60)

    assert store.goal(root.id).status is GoalStatus.OPEN
    assert store.open_goals()[0].id == root.id


# --- 崩溃恢复 ----------------------------------------------------------------

def test_the_graph_survives_being_reopened(tmp_path):
    """P-3 的验收之一:进程没了,图还在,而且状态一致。"""

    path = tmp_path / "graph.db"
    first = ProofGraphStore(path)
    root = first.upsert_goal(*ROOT)
    first.add_root(root.id, "fixture")
    accepted_split(first, root.id)
    left = first.goal_by_identity(LEFT[0])
    first.record_attempt(left.id, Outcome.PROVED, proof_text="by trivial")
    first.propagate(left.id, max_capability_attempts=3)
    first.close()

    second = ProofGraphStore(path)
    try:
        assert second.goal_by_identity(LEFT[0]).status is GoalStatus.PROVED
        assert second.goal(root.id).status is GoalStatus.OPEN
        assert len(second.decompositions_of(root.id)) == 1
    finally:
        second.close()
