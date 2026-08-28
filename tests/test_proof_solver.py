"""求解器接缝:outcome 只在一个地方派生,桩能按剧本走完每一条路径。

两组断言。

**派生规则**——`stopped_reason` 到 `Outcome` 的映射是承重的,因为它是「这次到底
成没成」的唯一真相源。图控制器不许有第二份看法。其中最容易写反的是顺序:先问
「证明了没有」,再问「怎么停的」。

**桩**——它存在的全部理由是让那些没法用真求解器测的路径可测:判题器按需宕机、
跑到一半被打断、预算正好在某个节点用完。
"""

import pytest

from evoharness.proof.graph import Goal, Outcome
from evoharness.proof.solver import AttemptResult, SolverError, StubSolver

GOAL = Goal(id="goal_1", identity="sha256:left", statement="theorem left : A := sorry")


class FakeReport:
    """`RunReport` 里派生只用到的三个字段。"""

    def __init__(self, stopped_reason, best_fitness=None, total_llm_cost=0.0):
        self.stopped_reason = stopped_reason
        self.best_fitness = best_fitness
        self.total_llm_cost = total_llm_cost


# --- PROVED 的不变式 ---------------------------------------------------------

def test_a_proved_attempt_must_carry_its_proof():
    """否则图上会有一个「已证但没东西可组装」的节点。

    这个矛盾如果不在构造时挡住,要等到最终整体复验才炸——离出错点很远,而那时
    看到的现象是「组装不起来」,不是「某个节点当初就不该被标成已证」。
    """

    with pytest.raises(ValueError, match="nothing to assemble"):
        AttemptResult(Outcome.PROVED)


def test_a_failed_attempt_cannot_carry_proof_text():
    """反过来也要挡:失败却带着证明,说明派生逻辑串了。"""

    with pytest.raises(ValueError, match="cannot carry proof text"):
        AttemptResult(Outcome.INFRA_FAILED, proof_text="by trivial")


# --- 派生规则 ----------------------------------------------------------------

def test_reaching_the_threshold_is_proved():
    result = AttemptResult.from_run_report(
        FakeReport("target_reached", best_fitness=1.0), proof_text="by trivial"
    )
    assert result.outcome is Outcome.PROVED


def test_a_proof_already_in_hand_survives_the_run_falling_over_afterwards():
    """顺序断言:先问证明了没有,再问怎么停的。

    判题器在拿到证明**之后**宕机,不该让这次尝试变成 infra-failed——那等于把
    已经到手的证明扔掉,然后让控制器重证一遍。
    """

    result = AttemptResult.from_run_report(
        FakeReport("eval_infra", best_fitness=1.0), proof_text="by trivial"
    )
    assert result.outcome is Outcome.PROVED


def test_an_infra_stop_without_a_proof_is_no_verdict():
    result = AttemptResult.from_run_report(FakeReport("eval_infra", best_fitness=0.4))
    assert result.outcome is Outcome.INFRA_FAILED


@pytest.mark.parametrize("reason", ["budget", "llm_billing"])
def test_running_out_of_money_is_not_running_out_of_ideas(reason):
    """`budget-exhausted` 混进 `task-failed` 会污染能力统计:前者加钱可能就成,
    后者才是唯一说明难度的信号。"""

    result = AttemptResult.from_run_report(FakeReport(reason, best_fitness=0.4))
    assert result.outcome is Outcome.BUDGET_EXHAUSTED


def test_finishing_without_a_proof_is_a_capability_failure():
    result = AttemptResult.from_run_report(FakeReport("completed", best_fitness=0.6))
    assert result.outcome is Outcome.TASK_FAILED


def test_the_solved_threshold_is_the_tasks_to_set():
    """判据阈值归任务,不是硬编码的 1.0。"""

    report = FakeReport("completed", best_fitness=0.9)
    assert AttemptResult.from_run_report(report).outcome is Outcome.TASK_FAILED
    assert AttemptResult.from_run_report(
        report, solved_at=0.9, proof_text="by trivial"
    ).outcome is Outcome.PROVED


def test_reaching_the_threshold_without_extractable_proof_text_is_an_error():
    """静默降级成 TASK_FAILED 会让一次成功变成一次失败,而且没人会知道。"""

    with pytest.raises(SolverError, match="no proof text"):
        AttemptResult.from_run_report(FakeReport("target_reached", best_fitness=1.0))


def test_an_interrupted_run_does_not_come_from_a_report():
    """能返回报告就说明没被打断。被打断是「它根本没返回」。"""

    result = AttemptResult.interrupted(run_dir="/runs/att_1")
    assert result.outcome is Outcome.INTERRUPTED
    assert result.run_dir == "/runs/att_1"


def test_the_run_cost_is_carried_through():
    result = AttemptResult.from_run_report(
        FakeReport("completed", best_fitness=0.1, total_llm_cost=3.25)
    )
    assert result.cost == 3.25


# --- 桩 ----------------------------------------------------------------------

def test_the_script_is_followed_in_order():
    """判题器按需宕机两次再恢复——真求解器上没法造出来的场景。"""

    solver = StubSolver({GOAL.identity: [
        Outcome.INFRA_FAILED, Outcome.INFRA_FAILED, Outcome.PROVED
    ]})

    outcomes = [solver.attack(GOAL, budget=100).outcome for _ in range(3)]

    assert outcomes == [
        Outcome.INFRA_FAILED, Outcome.INFRA_FAILED, Outcome.PROVED
    ]
    assert solver.calls == [GOAL.identity] * 3


def test_a_goal_the_script_never_mentions_fails_rather_than_succeeds():
    """桩的默认值不能是成功。默认成功会把「控制器压根没问过这个目标」伪装成
    「一切正常」,而那正是这一层最该抓的错。"""

    solver = StubSolver({})
    assert solver.attack(GOAL, budget=100).outcome is Outcome.TASK_FAILED


def test_the_script_running_out_falls_back_to_the_default():
    solver = StubSolver({GOAL.identity: [Outcome.TIMEOUT]})
    solver.attack(GOAL, budget=100)

    assert solver.attack(GOAL, budget=100).outcome is Outcome.TASK_FAILED


def test_a_proved_stub_attempt_supplies_proof_text():
    solver = StubSolver(
        {GOAL.identity: [Outcome.PROVED]},
        proofs={GOAL.identity: "theorem left : A := trivial"},
    )
    result = solver.attack(GOAL, budget=100)

    assert result.proof_text == "theorem left : A := trivial"


def test_running_out_of_budget_is_not_a_verdict_on_the_goal():
    """预算不够时目标根本没被试,它没有对自己的难度说过任何话。"""

    solver = StubSolver({GOAL.identity: [Outcome.PROVED]}, cost_per_attempt=5.0)
    result = solver.attack(GOAL, budget=1.0)

    assert result.outcome is Outcome.BUDGET_EXHAUSTED
    assert result.cost == 0.0
    # 剧本没有被消耗:钱到位之后这次尝试还在。
    assert solver.attack(GOAL, budget=100).outcome is Outcome.PROVED
