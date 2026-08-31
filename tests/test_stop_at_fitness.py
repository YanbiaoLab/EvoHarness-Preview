"""停在"已经解开了"这个点上,而不是把剩下的代数花在重解一遍。

演示跑 etp_one_dsv4 第 1 代就拿到 fitness 1.0,判题器收了证书,然后循环
照常往下跑 —— 最后是人手动 kill 的。这里钉的是三件事:声明了上限的任务会
停,**没声明的任务照常跑满**,以及那个数字住在任务里而不是启动命令里。

中间那条是对照。只测"该停的停了"的话,一个每代都停的实现全绿 —— 而它会
把每一次进化跑都砍成一代。
"""

import pathlib

import pytest

from evoharness.contracts.task import CriterionSpec
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

ROOT = pathlib.Path(__file__).resolve().parent.parent

INITIAL = """# EDIT-REGION-BEGIN
x = 0
x += 1
# EDIT-REGION-END
print(x)
"""


class CountingGrader:
    """Fitness is the number of increments, so it climbs one per generation."""

    def grade(self, cand, workdir) -> EvalReport:
        return EvalReport(
            fitness=float(cand.code.count("x += 1")),
            passed=True,
            eval_cost_usd=0.0,
        )


def climbing_transport():
    counter = {"n": 1}

    def transport(messages, model, **kw):
        counter["n"] += 1
        body = "x = 0\n" + "x += 1\n" * counter["n"]
        code = f"# EDIT-REGION-BEGIN\n{body}# EDIT-REGION-END\nprint(x)\n"
        return LLMResponse(
            text=(
                f"TITLE: add increment {counter['n']}\n"
                "SUMMARY: one more increment\n"
                f"```python\n{code}```"
            ),
            model=model,
            cost=0.0,
        )

    return transport


def build_loop(tmp_path, *, generations=10, stop_at=None, batch=1):
    cfg = SearchConfig(
        num_generations=generations,
        stop_at_fitness=stop_at,
        operators=["rewrite"],
        operator_probs=[1.0],
        seed=7,
        eval_batch_size=batch,
    )
    pop_cfg = PopulationConfig(num_islands=1)
    store = PopulationStore(pop_cfg)
    loop = SearchLoop(
        cfg=cfg,
        pop_cfg=pop_cfg,
        store=store,
        grader=CountingGrader(),
        llm=LLMClient(transport=climbing_transport(), sleep=lambda s: None),
        prompt_builder=PromptBuilder("maximize increments"),
        parent_selector=make_parent_selector(pop_cfg),
        inspiration_selector=InspirationSelector(pop_cfg),
        model_router=StaticRouter(["mock-model"]),
        workdir=tmp_path,
    )
    return loop, store


# -- the loop ---------------------------------------------------------------


def test_a_solved_run_stops_instead_of_spending_the_rest_of_its_schedule(tmp_path):
    """种子 1 分,之后每代 +1。上限设 4,应该在第 3 代停。"""

    loop, store = build_loop(tmp_path, generations=10, stop_at=4.0)
    report = loop.run(INITIAL)

    assert report.stopped_reason == "target_reached"
    assert report.generations_completed == 3
    assert report.best_fitness == 4.0
    # 停下来是不再提议,不是把已经跑出来的丢掉。
    assert store.count() == 1 + 3


def test_a_run_with_no_declared_ceiling_uses_its_whole_schedule(tmp_path):
    """上面那条的对照。

    没有这一条,一个"每代都停"的实现照样全绿 —— 而它会把每一次开放式搜索
    砍成一代,并且报 `target_reached`,读起来像是成功。
    """

    loop, store = build_loop(tmp_path, generations=6, stop_at=None)
    report = loop.run(INITIAL)

    assert report.stopped_reason == "completed"
    assert report.generations_completed == 6
    assert store.count() == 1 + 6


def test_a_ceiling_out_of_reach_does_not_end_the_run_early(tmp_path):
    """第二条对照,反着来:旋钮开着但没够到,不该改变任何行为。"""

    loop, _ = build_loop(tmp_path, generations=5, stop_at=999.0)
    report = loop.run(INITIAL)

    assert report.stopped_reason == "completed"
    assert report.generations_completed == 5


def test_the_batch_lane_stops_on_the_same_condition(tmp_path):
    """批量和串行是两条独立的路径。

    早停写在其中一条上、另一条漏掉,是那种"测过了"的缺陷:套件默认走串行,
    而真正的长跑几乎都开着 eval_batch_size。
    """

    loop, _ = build_loop(tmp_path, generations=10, stop_at=3.0, batch=2)
    report = loop.run(INITIAL)

    assert report.stopped_reason == "target_reached"
    assert report.generations_completed < 10


# -- where the number lives -------------------------------------------------


def test_a_criterion_may_declare_the_point_at_which_it_is_solved():
    criterion = CriterionSpec(name="judge_accepts", solved_at=1.0)
    assert criterion.solved_at == 1.0
    assert criterion.to_payload()["solved_at"] == 1.0


def test_a_minimizing_criterion_may_not_declare_one():
    """写在这里的数字会被拿去和 fitness 比,而 fitness 永远是越大越好。

    一个 minimize 的判据,人写下来的是判据值 —— 照单全收会让运行停在与
    任务声明**相反**的条件上,而且下游没有任何东西能看出来。
    """

    with pytest.raises(ValueError, match="maximizing criterion"):
        CriterionSpec(name="bytes", direction="minimize", solved_at=100.0)


def test_not_declaring_one_leaves_the_criterion_hash_untouched():
    """加这个字段没有让任何既有任务换掉身份。

    判据的 hash 进任务 hash 进运行身份:如果无条件写进 payload,每一个既有
    的 run 目录都会因为一个它没用到的字段而拒绝续跑。
    """

    plain = CriterionSpec(name="judge_accepts")
    assert "solved_at" not in plain.to_payload()
    # 声明了的那个必须换 hash —— 否则上面那条就是靠"这个字段谁也不影响"
    # 混过去的。
    assert plain.hash != CriterionSpec(name="judge_accepts", solved_at=1.0).hash


# -- the handoff from task to search ----------------------------------------


def _built(tmp_path, task_name, *overrides):
    from evoharness.launch.build import build
    from evoharness.launch.config import LaunchConfig

    return build(LaunchConfig(
        recipe="e0",
        run_dir=tmp_path / "run",
        task_dir=ROOT / "tasks" / "authored" / task_name,
        overrides=("search.num_generations=1", *overrides),
    ))


def test_an_authored_task_carries_its_own_ceiling_into_the_search(tmp_path):
    """数字住在 task.json 里,所以它进任务哈希:换掉它就是换了个实验,而不是
    换了个启动参数。启动器不知道一个任务有没有天花板,任务知道。
    """

    assert _built(tmp_path, "etp_one").search.stop_at_fitness == 1.0


def test_a_task_that_declares_none_leaves_the_search_open_ended(tmp_path):
    """对照。digit_power 的 fitness 没有"到此为止"的那个点,一个无条件填 1.0
    的实现会在第一代把它停掉。
    """

    assert _built(tmp_path, "digit_power").search.stop_at_fitness is None


def test_an_explicit_setting_outranks_the_task(tmp_path):
    """和上面 task_sys_msg 那一行同样的交接:任务给默认值,显式设置说了算。"""

    built = _built(tmp_path, "etp_one", "search.stop_at_fitness=0.5")
    assert built.search.stop_at_fitness == 0.5
