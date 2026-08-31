"""单题 ETP 任务:分数只能来自 Lean,判题器不在就不许出分。

这个任务是给演示用的,而演示最怕的不是跑不起来,是**跑起来了但里面没有
Lean**。`experiments/etp_stage2/grade.py` 在判题器不可达时自动退到纯 Python
的离线代理——对长跑是对的(判题器每份几秒),对这里是错的:退化之后的输出
和正常输出长得一模一样,同样的形状、同样的数字、同样的绿。

所以这里的用例分两类:**判题器在**的时候分数要对,**判题器不在**的时候要
炸而不是给个零分。后者没有对照就测不出来——一个永远返回 0 的实现,在只测
"种子得 0" 的套件里是全绿的。
"""

import importlib.util
import json
import pathlib

import pytest

TASK = pathlib.Path(__file__).resolve().parent.parent / "tasks" / "authored" / "etp_one"


def load_grade():
    spec = importlib.util.spec_from_file_location("etp_one_grade", TASK / "grade.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def grader():
    return load_grade()


@pytest.fixture
def workspace(tmp_path):
    return tmp_path


def test_a_missing_judge_stops_the_run_rather_than_scoring_zero(
    grader, workspace, monkeypatch
):
    """一个"Lean 悄悄没跑"的结果,读起来和"模型没做出来"完全一样。"""

    monkeypatch.delenv("ETP_JUDGE_URL", raising=False)
    (workspace / "submission.lean").write_text("import JudgeProblem\n")

    with pytest.raises(grader.JudgeUnavailable, match="no offline mode"):
        grader.grade(workspace, None)


def test_an_unreachable_judge_is_also_an_error_not_a_wrong_answer(
    grader, workspace, monkeypatch
):
    """判题器宕机不是候选答错。给它记 0 分,等于把一个捏造的数据点放进种群。"""

    monkeypatch.setenv("ETP_JUDGE_URL", "http://127.0.0.1:1")
    (workspace / "submission.lean").write_text("import JudgeProblem\n")

    with pytest.raises(grader.JudgeUnavailable, match="did not answer"):
        grader.grade(workspace, None)


def test_a_missing_submission_is_a_wrong_answer_not_an_error(grader, workspace):
    """这一个反过来:候选没交文件是候选的问题,不该惊动判题器。

    也是上面两条的对照——只测"缺判题器要炸"的话,一个凡事都炸的实现也全绿。
    """

    result = grader.grade(workspace, None)
    assert result["fitness"] == 0.0
    assert "submission.lean" in result["notes"]


def test_the_expected_verdict_follows_the_problem_label(grader):
    """label 决定要的是证明还是反例。送错种类,判题器会正确地拒——但那是在

    回答另一个问题,不是这道题答错了。"""

    task = json.loads((TASK / "task.json").read_text())
    row = json.loads((TASK / "problem.json").read_text())[task["problem_id"]]
    assert row["label"] in (True, False)
    # 现在在用的这道是反例题;换题时这条会跟着变,提醒改 prompt 的措辞。
    assert row["label"] is False


def test_the_official_axiom_policy_is_sent_every_time(grader):
    """判题器自己的缺省是零公理,比官方严得多。不显式传,好证书会被判不合格
    ——而且是静悄悄地判,看起来就像模型写错了。"""

    assert set(grader.OFFICIAL_POLICY["allowed_axioms"]) == {
        "propext", "Quot.sound", "Classical.choice",
    }


def test_the_task_declares_one_unit_and_names_its_problem(grader):
    """planned_units 必须是 1:单题、无进化,覆盖率是"这一道跑过了"。"""

    task = json.loads((TASK / "task.json").read_text())
    assert task["measurement"]["planned_units"] == 1
    # 题目 id 写在 task.json 里而不是启动参数里,所以它进任务哈希:换题就是
    # 换实验,两次跑不会被当成同一个比。
    assert task["problem_id"] in json.loads((TASK / "problem.json").read_text())
    assert task["measurement"]["universe_hash"] == task["problem_id"]


def test_the_seed_scores_zero_rather_than_erroring(grader):
    """种子交的是 `sorry`。它存在的意义是:什么都不改的跑得 0 分,而不是崩。"""

    seed = (TASK / "seed" / "submission.lean").read_text()
    assert "sorry" in seed


def test_the_worked_example_is_not_the_problem_being_asked(grader):
    """知识文件里放一份被接受过的证书是为了教格式,不是给答案。"""

    task = json.loads((TASK / "task.json").read_text())
    example = (TASK / "knowledge" / "accepted_example.md").read_text()
    assert "```lean" in example
    assert task["problem_id"] not in example
