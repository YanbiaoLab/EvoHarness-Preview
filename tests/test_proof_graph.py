"""状态机的传播规则:哪些事让目标变绿,哪些事绝不许。

这些断言全是纯函数上的,没有数据库、没有模型、没有编译器——**先把语义验对,
再接别的**。P-3 后面每一层的失败都会先经过这里,如果这层是错的,后面分不清是
分解质量差还是传播规则本来就写错了。

三条是承重的,单独成组:
- `infra-failed` 不许把目标推向 EXHAUSTED,也不许算进能力统计;
- 只有 ACCEPTED(草图过了 Lean)的分解才允许走到 COMPLETED;
- 无环检查跑在**身份**上,不是节点 id 上。
"""

import pytest

from evoharness.proof.graph import (
    CAPABILITY_OUTCOMES,
    NO_VERDICT_OUTCOMES,
    CycleError,
    DecompositionStatus,
    GoalStatus,
    Outcome,
    check_acyclic,
    decomposition_status_from,
    goal_status_from,
)


def status_of(attempts=(), decompositions=(), max_capability_attempts=3):
    return goal_status_from(
        attempt_outcomes=list(attempts),
        decomposition_statuses=list(decompositions),
        max_capability_attempts=max_capability_attempts,
    )


# --- 目标怎么变绿 ------------------------------------------------------------

def test_one_proved_attempt_closes_the_goal():
    """OR 节点:任何一条路走通就够了。"""

    assert status_of([Outcome.TASK_FAILED, Outcome.PROVED]) is GoalStatus.PROVED


def test_a_completed_decomposition_closes_the_goal():
    """AND 节点全绿之后,父目标也绿——这是分解的全部意义。"""

    assert status_of(
        decompositions=[DecompositionStatus.COMPLETED]
    ) is GoalStatus.PROVED


def test_a_live_decomposition_keeps_the_goal_open_however_the_attempts_went():
    """还有路可走,就不该因为直接证失败若干次而宣告穷尽。"""

    exhausting = [Outcome.TASK_FAILED] * 9
    assert status_of(
        exhausting, [DecompositionStatus.ACCEPTED], max_capability_attempts=3
    ) is GoalStatus.OPEN


# --- 承重条款:infra 不是能力信号 --------------------------------------------

@pytest.mark.parametrize("outcome", sorted(NO_VERDICT_OUTCOMES, key=str))
def test_no_verdict_outcomes_never_exhaust_a_goal(outcome):
    """判题器宕机、跑到一半被打断、钱花完了——没有一样是「这题做不出来」。

    把它们算进能力统计,就是让基础设施故障重写整张图的形状:目标被判穷尽,
    控制器转去找别的分解,而那个分解本来根本不需要。
    """

    assert outcome not in CAPABILITY_OUTCOMES
    assert status_of([outcome] * 20, max_capability_attempts=3) is GoalStatus.OPEN


def test_only_capability_failures_count_toward_exhaustion():
    """穷尽必须是「试过做不到」堆出来的,不能被无裁定的结果掺进去。"""

    mixed = [Outcome.INFRA_FAILED, Outcome.TASK_FAILED, Outcome.INTERRUPTED]
    assert status_of(mixed, max_capability_attempts=3) is GoalStatus.OPEN

    assert status_of(
        mixed + [Outcome.TIMEOUT, Outcome.TASK_FAILED], max_capability_attempts=3
    ) is GoalStatus.EXHAUSTED


def test_exhaustion_can_be_switched_off_entirely():
    """预算相对:上限为 0 表示这一轮不做穷尽判定。"""

    assert status_of(
        [Outcome.TASK_FAILED] * 50, max_capability_attempts=0
    ) is GoalStatus.OPEN


# --- 承重条款:只有验证过的分解才能完成 --------------------------------------

def test_an_unvalidated_decomposition_cannot_complete():
    """草图还没过 Lean,子目标全绿也不算数。

    否则一个未经验证的分解可以凭「子目标都绿了」把父目标标成已证,而那些
    引理未必拼得回原命题——装配漂移从状态机这一侧进来。
    """

    assert decomposition_status_from(
        current=DecompositionStatus.PROPOSED,
        subgoal_statuses=[GoalStatus.PROVED, GoalStatus.PROVED],
    ) is DecompositionStatus.PROPOSED


def test_an_accepted_decomposition_completes_when_every_subgoal_is_proved():
    assert decomposition_status_from(
        current=DecompositionStatus.ACCEPTED,
        subgoal_statuses=[GoalStatus.PROVED, GoalStatus.PROVED],
    ) is DecompositionStatus.COMPLETED


def test_one_open_subgoal_is_enough_to_hold_a_decomposition_back():
    assert decomposition_status_from(
        current=DecompositionStatus.ACCEPTED,
        subgoal_statuses=[GoalStatus.PROVED, GoalStatus.OPEN],
    ) is DecompositionStatus.ACCEPTED


def test_a_decomposition_with_no_subgoals_does_not_complete_vacuously():
    """`all([])` 是 True。空分解要是能完成,它就是一台凭空造证明的机器。"""

    assert decomposition_status_from(
        current=DecompositionStatus.ACCEPTED, subgoal_statuses=[]
    ) is DecompositionStatus.ACCEPTED


def test_a_reviewer_rejection_does_not_drift_back_to_completed():
    """审稿人否决可以重访,但重访是显式动作,不是靠子目标状态自己漂回来。"""

    assert decomposition_status_from(
        current=DecompositionStatus.REJECTED_BY_REVIEWER,
        subgoal_statuses=[GoalStatus.PROVED],
    ) is DecompositionStatus.REJECTED_BY_REVIEWER


def test_a_verifier_rejection_is_terminal():
    """Lean 说草图不合法是事实,不是可以再商量的判断。"""

    assert decomposition_status_from(
        current=DecompositionStatus.REJECTED_BY_VERIFIER,
        subgoal_statuses=[GoalStatus.PROVED],
    ) is DecompositionStatus.REJECTED_BY_VERIFIER


# --- 承重条款:无环检查跑在身份上 --------------------------------------------

def test_proposing_an_ancestor_as_a_subgoal_is_refused():
    """LEAP 去掉审稿人之后的退化模式:提出一个与祖父目标等价的子目标。"""

    with pytest.raises(CycleError, match="sha256:root"):
        check_acyclic(["sha256:root", "sha256:mid"], ["sha256:root"])


def test_the_check_runs_against_transitive_ancestors_not_just_the_parent():
    """那个退化模式正好跨了一层:先展开定义,再折叠回去。"""

    with pytest.raises(CycleError):
        check_acyclic(["sha256:grandparent", "sha256:parent"], ["sha256:grandparent"])


def test_genuinely_new_subgoals_pass():
    check_acyclic(["sha256:root"], ["sha256:a", "sha256:b"])
