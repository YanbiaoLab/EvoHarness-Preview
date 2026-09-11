"""崩溃恢复:进程死了,图还在,而且「试过但被打断」不会丢。

P-3 验收里最后一条。它测的不是 SQLite 会不会坏——那是 SQLite 的事——而是
**语义能不能续上**:

- 已经证完的子目标不必重证;
- 死在半路的那次尝试要被记成 `INTERRUPTED`,而不是无声消失;
- `INTERRUPTED` 在 NO_VERDICT 里,所以**崩溃不算「这条引理难」**;
- 第二个控制器接着跑,能把根目标关掉。

其中一条用真 SIGKILL 测。用异常模拟杀进程测不出来:异常会走 `finally`,把租约
释放掉,而真正被 kill 的进程什么都不会做——留下的正是那把没人释放的租约。
"""

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from evoharness.proof.controller import FixedDecompositions, ProofController
from evoharness.proof.graph import GoalStatus, Outcome
from evoharness.proof.sketch import Sketch, SubgoalSpec, Validation
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


def accept_all(goal, sketch):
    return Validation(ok=True)


@pytest.fixture
def store(tmp_path):
    graph = ProofGraphStore(tmp_path / "graph.db")
    yield graph
    graph.close()


def controller_for(store, solver, **kwargs):
    return ProofController(
        store,
        solver,
        decompositions=FixedDecompositions({ROOT[0]: SPLIT}),
        validate_sketch=accept_all,
        **kwargs,
    )


# --- 恢复扫描 ----------------------------------------------------------------

def test_an_abandoned_lease_becomes_a_recorded_interruption(store):
    """死掉的 worker 什么都没记。不补这一条,「试过」这个事实就没了。"""

    root = store.upsert_goal(*ROOT)
    store.claim(root.id, "worker-that-died", ttl_s=60, now=1000.0)

    controller = controller_for(store, StubSolver({}))
    recovered = controller.recover(now=2000.0)

    assert recovered == [root.id]
    outcomes = [a.outcome for a in store.attempts_of(root.id)]
    assert outcomes == [Outcome.INTERRUPTED]
    assert "worker-that-died" in store.attempts_of(root.id)[0].note


def test_the_interruption_says_where_the_dead_attempt_was_working(store):
    """承重项。死掉的 worker 什么都没记,而它的轨迹还在磁盘上——一条指不出
    目录的记录,读起来和「什么都没试过」一模一样,轨迹就此成为孤儿。"""

    root = store.upsert_goal(*ROOT)
    store.claim(root.id, "worker-that-died", ttl_s=60, now=1000.0)
    store.note_attempt_dir(root.id, "/runs/id_root/attempt_0007")

    controller_for(store, StubSolver({})).recover(now=2000.0)

    attempt = store.attempts_of(root.id)[0]
    assert attempt.outcome is Outcome.INTERRUPTED
    assert attempt.run_dir == "/runs/id_root/attempt_0007"


def test_recovery_frees_the_goal_for_the_next_worker(store):
    root = store.upsert_goal(*ROOT)
    store.claim(root.id, "worker-that-died", ttl_s=60, now=1000.0)
    store.note_attempt_dir(root.id, "/runs/id_root/attempt_0007")

    controller_for(store, StubSolver({})).recover(now=2000.0)

    assert store.goal(root.id).lease_owner is None
    assert store.goal(root.id).lease_run_dir is None
    assert store.claim(root.id, "worker-b", ttl_s=60, now=2000.0) is True


def test_a_new_holder_inherits_no_directory_from_the_last_one(store):
    """指针跟着租约走,而**接手过期租约**是它唯一会出错的那条路。

    worker A 记下目录后死掉,worker B 直接抢走过期的租约(恢复扫描还没跑到)。
    B 要是也死了,而指针还是 A 的,那次中断就会指着**别人的**轨迹——
    一个指错地方的指针比没有指针难发现得多,它一路读起来都成立。

    刻意不先 `release`:那条路上指针本来就被清掉了,测它等于什么都没测。
    """

    root = store.upsert_goal(*ROOT)
    store.claim(root.id, "worker-a", ttl_s=60, now=1000.0)
    store.note_attempt_dir(root.id, "/runs/id_root/attempt_0000")

    assert store.claim(root.id, "worker-b", ttl_s=60, now=2000.0) is True
    assert store.goal(root.id).lease_run_dir is None

    controller_for(store, StubSolver({})).recover(now=9000.0)

    attempt = store.attempts_of(root.id)[0]
    assert "worker-b" in attempt.note
    assert attempt.run_dir is None


def test_two_controllers_do_not_share_a_name(store):
    """承重项,而且是并行的前提。

    租约是**按名字**释放的(`WHERE lease_owner = ?`)。两个控制器都叫
    `controller`,先收工的那个就会把另一个还攥着的目标放出来,第三个随即在
    同一条引理上起跑——**正是租约要防的那个失败,从标识符这一侧绕了回来。**
    """

    root = store.upsert_goal(*ROOT)
    a = controller_for(store, StubSolver({}))
    b = controller_for(store, StubSolver({}))

    assert a.owner != b.owner

    store.claim(root.id, a.owner, ttl_s=600, now=1000.0)
    store.release(root.id, b.owner)

    assert store.goal(root.id).lease_owner == a.owner
    assert store.claim(root.id, b.owner, ttl_s=600, now=1100.0) is False


def test_the_interruption_names_a_worker_rather_than_a_role(store):
    """`lease held by controller expired` 在 N 个 worker 之下什么也没说。"""

    root = store.upsert_goal(*ROOT)
    a = controller_for(store, StubSolver({}))
    store.claim(root.id, a.owner, ttl_s=60, now=1000.0)

    controller_for(store, StubSolver({})).recover(now=2000.0)

    assert a.owner in store.attempts_of(root.id)[0].note


def test_an_older_graph_opens_without_the_column_and_reads(tmp_path):
    """图比这个字段活得久。`CREATE TABLE IF NOT EXISTS` 加得了表、加不了列,
    所以旧图会在第一次读的时候炸——而它们正是要恢复的那些。"""

    import sqlite3

    path = tmp_path / "old.db"
    ProofGraphStore(path).close()
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE goals DROP COLUMN lease_run_dir")

    store = ProofGraphStore(path)
    try:
        goal = store.upsert_goal(*ROOT)
        assert store.goal(goal.id).lease_run_dir is None
    finally:
        store.close()


def test_a_crash_is_not_evidence_that_a_lemma_is_hard(store):
    """`INTERRUPTED` 在 NO_VERDICT 里。崩溃二十次也不该把目标判成穷尽——
    否则一次不稳的机器就能把一条本来可证的引理从图上抹掉。"""

    root = store.upsert_goal(*ROOT)
    controller = controller_for(store, StubSolver({}), max_capability_attempts=2)

    for index in range(20):
        store.claim(root.id, f"worker-{index}", ttl_s=1, now=1000.0 + index)
        controller.recover(now=5000.0)

    assert store.goal(root.id).status is GoalStatus.OPEN


def test_a_live_lease_is_left_alone(store):
    """还没到期的租约是别人正在干活,不是尸体。"""

    root = store.upsert_goal(*ROOT)
    store.claim(root.id, "worker-a", ttl_s=600, now=1000.0)

    assert controller_for(store, StubSolver({})).recover(now=1100.0) == []
    assert store.goal(root.id).lease_owner == "worker-a"


def test_solve_recovers_before_it_starts(store):
    """恢复是循环的第一件事,不用调用方记得做。"""

    root = store.upsert_goal(*ROOT)
    store.claim(root.id, "worker-that-died", ttl_s=1, now=time.time() - 100)

    solver = StubSolver({ROOT[0]: [Outcome.PROVED]})
    report = controller_for(store, solver).solve(root.id, budget=100)

    assert report.root_proved is True
    outcomes = [a.outcome for a in store.attempts_of(root.id)]
    assert outcomes == [Outcome.INTERRUPTED, Outcome.PROVED]


# --- 真的杀进程 --------------------------------------------------------------

_CRASH_SCRIPT = textwrap.dedent(
    '''
    import os, sys
    sys.path.insert(0, {repo!r})
    from evoharness.proof.controller import FixedDecompositions, ProofController
    from evoharness.proof.graph import Goal, Outcome
    from evoharness.proof.solver import AttemptResult, StubSolver
    from evoharness.proof.store import ProofGraphStore
    sys.path.insert(0, {tests!r})
    from test_proof_recovery import ROOT, LEFT, RIGHT, SPLIT, accept_all

    class DiesAfterFirstLemma:
        level = "stub"
        def attack(self, goal, *, budget):
            if goal.identity == ROOT[0]:
                return AttemptResult(Outcome.TASK_FAILED, cost=1.0)
            if goal.identity == LEFT[0]:
                return AttemptResult(
                    Outcome.PROVED, proof_text="ha", cost=1.0
                )
            # 第二条引理攻到一半:租约已经拿在手里,尝试还没记。
            # 用 SIGKILL 而不是抛异常——异常会走 finally 把租约释放掉,
            # 而那把没人释放的租约正是要测的东西。
            os.kill(os.getpid(), 9)

    store = ProofGraphStore({db!r})
    root = store.upsert_goal(*ROOT)
    store.add_root(root.id, "crash")
    ProofController(
        store, DiesAfterFirstLemma(),
        decompositions=FixedDecompositions({{ROOT[0]: SPLIT}}),
        validate_sketch=accept_all,
        lease_ttl_s=1,
    ).solve(root.id, budget=100)
    '''
)


def test_a_killed_process_leaves_a_graph_the_next_one_can_finish(tmp_path):
    """P-3 的验收原文:杀掉进程再恢复,图状态完整。"""

    repo = str(Path(__file__).resolve().parent.parent)
    database = tmp_path / "graph.db"
    script = _CRASH_SCRIPT.format(
        repo=repo, tests=str(Path(__file__).resolve().parent), db=str(database)
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    # -9: 进程确实是被杀的,不是自己干净退出的。
    assert done.returncode == -9, done.stderr[-2000:]

    store = ProofGraphStore(database)
    try:
        root = store.goal_by_identity(ROOT[0])
        left = store.goal_by_identity(LEFT[0])
        right = store.goal_by_identity(RIGHT[0])

        # 崩溃之前的进展留下了:分解在,第一条引理已证。
        assert left.status is GoalStatus.PROVED
        assert right.status is GoalStatus.OPEN
        assert root.status is GoalStatus.OPEN
        assert len(store.decompositions_of(root.id)) == 1

        # 等那把死租约到期。这不是测试的权宜之计,是真实约束:恢复要等 TTL,
        # 所以长跑的 TTL 定多长就是「崩溃后多久能接手」。
        time.sleep(1.2)

        # 第二个控制器接手:恢复那把死租约,只攻剩下的那条,收尾。
        solver = StubSolver(
            {RIGHT[0]: [Outcome.PROVED]}, proofs={RIGHT[0]: "hb"}
        )
        report = ProofController(
            store, solver,
            decompositions=FixedDecompositions({}),
            validate_sketch=accept_all,
        ).solve(root.id, budget=100)

        assert report.root_proved is True
        # 已证的那条一次都没有被重攻——记忆化的意义就在这里。
        assert LEFT[0] not in solver.calls
        assert Outcome.INTERRUPTED in [
            a.outcome for a in store.attempts_of(right.id)
        ]
    finally:
        store.close()
