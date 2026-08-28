"""P-0a 的验收:Lean 判题接对了,而且接错的那几种方式都会被抓住。

这个 fixture 任务本身很平凡——三个连接词,每个一行就能证完。**它的难度不是
重点,接线才是。** `todo/proof_vertical.md` 把四件事列为承重,这里逐条钉死:

1. `passed` = 编译通过,不是「完整证明」。`Candidate.archive_eligible` 就是
   `return self.passed`,而种子是一份 `sorry` 骨架。把 `passed` 定成「证完了」,
   种子自己就不是 passed,岛里没有 passed 父本,`_pick_island` 跳过整个岛,
   **一个提案都产不出来**——而这在只测「种子得低分」的套件里是全绿的。
2. 适应度分级,否则 0/1 让搜索没有东西可爬。
3. 命题由 Lean 钉住:削弱定理不是更省力的高分路线,是编译错误。
4. 工具链坏了不是候选答错——那要炸,不是给零分。

第 1 和第 4 条都需要**对照**才测得出来:一个永远返回 0.0 的实现,和一个永远
返回 passed=False 的实现,在只测「种子分数低」的套件里同样全绿。
"""

import importlib.util
import pathlib
import shutil

import pytest

from evoharness.serve import InfraError

TASK = (
    pathlib.Path(__file__).resolve().parent.parent
    / "tasks" / "authored" / "lean_fixture"
)
SEED = (TASK / "seed" / "fixture.lean").read_text(encoding="utf-8")

FULL_PROOF = "  refine ⟨Nat.add_mul a b c, Nat.add_assoc a b c, Nat.mul_zero a⟩"

pytestmark = pytest.mark.skipif(
    shutil.which("lean") is None,
    reason="P-0a 是 Lean 判题的验收,没有 lean 就没有什么可验的",
)


def load_grade():
    spec = importlib.util.spec_from_file_location(
        "lean_fixture_grade", TASK / "grade.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def grader():
    return load_grade()


def write(tmp_path, source: str) -> pathlib.Path:
    (tmp_path / "fixture.lean").write_text(source, encoding="utf-8")
    return tmp_path


# --- 承重条款一:passed 的语义 ------------------------------------------------

def test_the_sorry_seed_is_passed_so_it_can_be_a_parent(grader, tmp_path):
    """种子必须能当父本,否则整个 run 产不出提案。

    这不是风格问题。`SeedOnlySelector` 从 `passed_candidates` 里取
    generation 0 的那一个,而种子是**被判过分的**;它不 passed,岛里就没有可用
    父本,`_pick_island` 直接跳过。L1、L2 也一样受影响,不只 L3/L4。
    """

    report = grader.grade(write(tmp_path, SEED), None)

    assert report["passed"] is True
    assert report.get("fault_kind") is None
    assert report["visible_metrics"]["compile_ok"] == 1
    assert report["visible_metrics"]["axioms"] == "sorryAx"


# --- 承重条款二:适应度分级 --------------------------------------------------

def test_a_compiling_sorry_file_scores_strictly_between_zero_and_one(
    grader, tmp_path
):
    """带 sorry 但能编译,比编译不过更近——分数必须看得出这个差别。"""

    report = grader.grade(write(tmp_path, SEED), None)

    assert 0.0 < report["fitness"] < 1.0


def test_closing_helper_lemmas_raises_the_score(grader, tmp_path):
    """软适应度要随进展单调上升,否则父本选择排的是噪声。"""

    seed_score = grader.grade(write(tmp_path, SEED), None)["fitness"]

    partial = SEED.replace(
        "theorem fixture_main",
        "theorem h1 (a b c : Nat) : (a+b)*c = a*c+b*c := Nat.add_mul a b c\n"
        "theorem h2 (a b c : Nat) : (a+b)+c = a+(b+c) := Nat.add_assoc a b c\n"
        "theorem fixture_main",
    )
    partial_score = grader.grade(write(tmp_path, partial), None)["fitness"]

    assert seed_score < partial_score < 1.0


def test_only_the_axiom_report_awards_one(grader, tmp_path):
    """1.0 只能来自 Lean 的公理报告,不能来自任何软信号。"""

    report = grader.grade(write(tmp_path, SEED.replace("  sorry", FULL_PROOF)), None)

    assert report["fitness"] == 1.0
    assert report["passed"] is True
    assert "sorryAx" not in report["visible_metrics"]["axioms"]


# --- 承重条款三:命题由 Lean 钉住 --------------------------------------------

def test_weakening_the_theorem_is_a_compile_error_not_a_cheap_win(
    grader, tmp_path
):
    """「证一个更容易的命题」必须是编译错误,而不是刷软分最省力的路。

    守卫不是比对字节,是锁定尾部那句 `example ... := fixture_main`:定理一被
    削弱,它就类型不匹配。
    """

    weakened = SEED.replace(
        "theorem fixture_main (a b c : Nat) :\n"
        "    (a + b) * c = a * c + b * c\n"
        "    ∧ (a + b) + c = a + (b + c)\n"
        "    ∧ a * 0 = 0 := by\n"
        "  sorry",
        "theorem fixture_main (a b c : Nat) : a * 0 = 0 := Nat.mul_zero a",
    )
    report = grader.grade(write(tmp_path, weakened), None)

    assert report["fitness"] == 0.0
    assert report["passed"] is False
    assert report["visible_metrics"]["compile_ok"] == 0


def test_deleting_the_locked_footer_is_malformed_not_merely_wrong(
    grader, tmp_path
):
    """删掉尾部,剩下的文件照样编译、照样打印公理——但命题不再被钉住。

    这是类型检查唯一抓不到的攻击,所以得单独查一次。
    """

    report = grader.grade(
        write(tmp_path, SEED.split("-- LOCKED FOOTER")[0]), None
    )

    assert report["passed"] is False
    assert report["fault_kind"] == "invalid_candidate"


def test_native_decide_is_not_a_proof_here(grader, tmp_path):
    """编译通过、公理报告也打出来了,但依赖表里多了东西,就不算证明。"""

    cheating = SEED.replace(
        "theorem fixture_main",
        "theorem cheat : (2:Nat) + 2 = 4 := by native_decide\n\n"
        "theorem fixture_main",
    ).replace(
        "  sorry",
        "  have h := cheat\n"
        "  refine ⟨Nat.add_mul a b c, Nat.add_assoc a b c, ?_⟩\n"
        "  cases h\n"
        "  exact Nat.mul_zero a",
    )
    report = grader.grade(write(tmp_path, cheating), None)

    assert report["fitness"] == 0.0
    assert report["passed"] is False
    assert "forbidden axioms" in report["fault"]


# --- 承重条款四:坏的工具链不是答错 ------------------------------------------

def test_a_missing_lean_stops_the_run_rather_than_scoring_zero(
    grader, tmp_path, monkeypatch
):
    """没有 Lean 的一次运行,读起来和「模型没做出来」一模一样。

    给它记 0 分等于把一个捏造的数据点放进种群,并且把基础设施故障说成能力
    退步——ETP 那边为这个错误付过整轮结论作废的代价。
    """

    monkeypatch.setattr(grader.shutil, "which", lambda name: None)

    with pytest.raises(InfraError, match="no offline mode"):
        grader.grade(write(tmp_path, SEED), None)


def test_a_slow_proof_is_the_candidates_fault_not_infrastructure(
    grader, tmp_path, monkeypatch
):
    """候选自己写的证明跑超时,是**有裁定**的失败,不是无裁定的基础设施故障。

    两者在本仓库的分类里后果不同:`timeout` 是 VERDICT fault(记录在案、可修复、
    不当父本),`infra_error` 是 NO_VERDICT(整条丢弃、不计入评估次数)。把候选
    写崩的 `decide` 记成基础设施故障,会让真实的能力失败从统计里消失。
    """

    monkeypatch.setattr(grader, "TIMEOUT_S", 0.001)
    report = grader.grade(write(tmp_path, SEED), None)

    assert report["passed"] is False
    assert report["fault_kind"] == "timeout"
    assert report["visible_metrics"]["timed_out"] == 1


# --- 判题的可重复性 ----------------------------------------------------------

def test_the_same_file_grades_the_same_twice(grader, tmp_path):
    """判两次不一致的评估器,后面所有的比较都建立在沙子上。"""

    first = grader.grade(write(tmp_path, SEED), None)
    second = grader.grade(write(tmp_path, SEED), None)

    assert first["fitness"] == second["fitness"]
    assert first["passed"] == second["passed"]
    assert first["visible_metrics"]["axioms"] == second["visible_metrics"]["axioms"]


# --- 端到端 ------------------------------------------------------------------

def test_the_fixture_runs_end_to_end_through_api_run(tmp_path):
    """种子 → 提案 → preflight → Lean → 证据,整条要真的通。

    这条抓到过一个真问题:种子原本用的是自造的 `-- EVOLVE-BEGIN` 标记,而
    `validate_edit_markers` 认的是 `EDIT-REGION-BEGIN`,于是每个提案都在
    preflight 就被拒,`best_fitness` 停在种子的 0.1。**单测全绿,曲线也不难看,
    只是一次变异都没成功过。**
    """

    from evoharness import BasicSearchProfile, ComponentSpec, RunSpec, run
    from evoharness.authoring import load_task_from_dir
    from evoharness.core import LLMResponse, LLMStopReason

    proved = SEED.replace("  sorry", FULL_PROOF)

    def transport(*, messages, model, **kw):
        return LLMResponse(
            text="Close all three conjuncts.\n\n```lean\n" + proved + "\n```",
            model=model, cost=0.0, prompt_tokens=10, completion_tokens=10,
            stop_reason=LLMStopReason.COMPLETED,
        )

    report = run(
        load_task_from_dir(TASK),
        RunSpec(
            models=("fake-model",),
            proposer_backend=ComponentSpec.create(
                "proposer_backend", "tests.fake", version="v1"
            ),
            output_dir=str(tmp_path / "run"),
            seed=1,
        ),
        BasicSearchProfile(num_trajectories=2, proposal_mode="single_shot"),
        transport=transport,
    )

    assert report.stopped_reason == "completed"
    assert report.best_fitness == 1.0
    # 种子 + 两条轨迹。少于三次意味着提案被拒而不是被评估。
    assert report.evaluations == 3


# --- 任务声明本身 ------------------------------------------------------------

def test_the_authored_directory_loads_and_declares_what_it_should():
    """判据、种子主文件、preflight 三样接错任何一样,后面的验收都测不到点上。"""

    from evoharness.authoring import load_task_from_dir

    task = load_task_from_dir(TASK)

    assert task.spec.criterion.solved_at == 1.0
    assert task.spec.criterion.direction == "maximize"
    assert task.initial_code == SEED
    assert {check.name for check in task.preflight_validators} == {
        "fixture_present",
        "locked_footer_present",
    }
