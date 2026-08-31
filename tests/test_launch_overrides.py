"""启动命令怎么把设置交给运行 —— 以及它怎么在半路把设置弄丢。

两个缺陷,一个模式:**设置消失得毫无声息**。丢掉的那些悄悄取了默认值,
而运行的记录会把那些默认值写成"这就是当初要的"。

  * `--set` 用的是 `nargs="*"`,第二次出现整体替换第一次。为了可读把设置
    分几行写的调用者,实际只有最后一组生效。

  * 没有 `--model` 的时候,EvoHarness 问的是它自己的默认模型,而候选运行时
    的 catalog 里只有另一个名字。运行 etp_one_run1 就这样在五秒内烧掉五代、
    零后代,最后报 `proposer_dead` —— 一个关于提议器的判决,而提议器是好的。
"""

import json

import pytest

from evoharness.launch import start as start_module
from evoharness.launch.start import EXIT_OK, build_parser, main
from recipes import load_experiment_config


class _Started:
    def to_json(self) -> dict:
        return {"ok": True}


@pytest.fixture
def launched(monkeypatch):
    """Run `main` with the launch itself stubbed out; keep what it was told."""

    seen: dict = {}

    def fake_start_run(run_dir, argv, *, cwd, launch, force):
        seen["launch"] = launch
        return _Started()

    monkeypatch.setattr(start_module, "start_run", fake_start_run)

    def run(*extra: str) -> list[str]:
        code = main([
            "--recipe", "e0", "--run-dir", "/tmp/does-not-need-to-exist", *extra
        ])
        assert code == EXIT_OK, seen
        return seen["launch"]["overrides"]

    return run


# -- --set ------------------------------------------------------------------


def test_two_set_flags_both_survive(launched):
    """这条是缺陷本身。之前 `search.seed=1` 会被整段丢掉。"""

    overrides = launched("--set", "proposal.mode=agentic", "--set", "search.seed=1")

    assert overrides == ["proposal.mode=agentic", "search.seed=1"]


def test_one_set_flag_with_several_values_still_works(launched):
    """上面那条的对照,也是 README 里写的用法。

    改成 `action="append"` 同样能让上面那条变绿,但会把这里的两个值套进一层
    列表,再让 `load_experiment_config` 在一个 list 上做 partition。
    """

    overrides = launched("--set", "proposal.mode=agentic", "search.seed=1")

    assert overrides == ["proposal.mode=agentic", "search.seed=1"]


def test_no_set_flag_leaves_the_defaults_alone(launched):
    assert launched() == []


def test_the_parser_does_not_accumulate_across_calls():
    """argparse 的可变默认值陷阱:`extend` 若就地改 default,第二次解析会带上
    第一次的东西 —— 在一个进程里解析两次命令行的地方(测试、resume)才看得见。
    """

    first = build_parser().parse_args(["--recipe", "e0", "--run-dir", "/x",
                                       "--set", "search.seed=1"])
    second = build_parser().parse_args(["--recipe", "e0", "--run-dir", "/x"])

    assert first.overrides == ["search.seed=1"]
    assert second.overrides == []


# -- --model ----------------------------------------------------------------


def test_model_names_itself_on_both_keys_that_decide_it(launched):
    """两个地方都要说,因为两个地方都会被读:`proposal.model` 是单次提议用的,
    `search.llm_models[0]` 是它为空时的退路,也是运行记录里写下来的那个。
    """

    overrides = launched("--model", "deepseek-v4-pro")

    assert overrides == [
        "proposal.model=deepseek-v4-pro",
        'search.llm_models=["deepseek-v4-pro"]',
    ]
    # 而且要真能解析出来 —— 上面比的是字符串,这里比的是它的意思。
    search, _, _, proposal = load_experiment_config(None, overrides)
    assert proposal.model == "deepseek-v4-pro"
    assert search.llm_models == ["deepseek-v4-pro"]


def test_an_explicit_set_beats_the_flag(launched):
    """`--model` 是个便利写法,不是个上锁的门。显式写下来的必须赢。"""

    overrides = launched(
        "--model", "deepseek-v4-pro", "--set", "proposal.model=some-other",
    )

    search, _, _, proposal = load_experiment_config(None, overrides)
    assert proposal.model == "some-other"


def test_without_the_flag_nothing_is_said_about_the_model(launched):
    """对照:一个无条件写死模型的实现,在只测上面两条的套件里是全绿的,
    而它会让 `--set proposal.model=...` 和配置文件都失效。
    """

    overrides = launched("--set", "search.seed=1")

    assert not any("model" in item for item in overrides)


def test_the_flag_is_described_as_having_to_match_the_route(launched):
    """帮助文本是这里唯一会被人读到的东西 —— 名字对不上的代价就写在它里面。"""

    help_text = build_parser().format_help()
    assert "--model" in help_text
    assert "catalog" in help_text


def test_a_model_name_with_a_quote_survives_as_one_value(launched):
    """展开走的是 JSON,不是字符串拼接。手写引号的话,一个带引号的名字会把
    override 切成两半,而 `_parse_value` 会安静地退回成字符串。
    """

    overrides = launched("--model", 'weird"name')

    search, _, _, _ = load_experiment_config(None, overrides)
    assert search.llm_models == ['weird"name']
    assert json.loads(overrides[1].partition("=")[2]) == ['weird"name']
