"""新增的顶层函数必须被接上,否则它永远不会执行。

2026-08-28 实测:一个候选写了 179 行 `infinite_shifted_order_offset_model` ——
一个新的 Presburger 可判定模型族,想法对、代码质量不低,连"别和已有族重复"的
守卫都写了。然后 `strict_search`(唯一的调度器)与种子**逐字节相同**,整个
工作区里那个函数只出现一次:它自己的 def。

那 179 行永远不会执行。八个档位有七个与种子逐位相同,110 轮、21.4 万输出
token、19 次编辑,换来一个测不出差别的候选 —— 而完整评测要半小时才说出来。
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

os.environ.setdefault(
    "ETP_SEED_STATE_DIR",
    str(Path(__file__).resolve().parents[1]
        / "experiments/etp_stage2/seeds/solo_v3_1"))

from evoharness.core.preflight import PreflightContext  # noqa: E402
from experiments.etp_stage2.preflight import (  # noqa: E402
    Stage2SubmissionValidator,
    _orphan_definitions,
)

SEED = Path("/Users/zhangkang/Documents/Projects/etp-work/seed_solo_v3_1")

NEW_ENGINE = '''

def infinite_shifted_order_offset_model(source, target):
    """A genuinely new family; compares x < y + d instead of x < y."""
    return None
'''


@pytest.fixture()
def ws(tmp_path):
    if not (SEED / "layout.json").exists():
        pytest.skip("gemma 谱系种子不在")
    out = tmp_path / "ws"
    shutil.copytree(SEED, out)
    return out


class _Parent:
    """父本只需要给出 `code`(序列化后的工作区文本)。"""

    def __init__(self, code):
        self.code = code


def _seed_code():
    import json as _json

    order = _json.loads((SEED / "layout.json").read_text(encoding="utf-8"))["order"]
    return "\n".join((SEED / n).read_text(encoding="utf-8")
                     for n in order if n.endswith(".py"))


def _codes(ws, parent=None):
    if parent is None:
        parent = _Parent(_seed_code())
    r = Stage2SubmissionValidator().validate(
        PreflightContext(parent=parent, operator="revise", workdir=str(ws)))
    return {i.code for i in r.issues}, r


def _order(ws):
    return list(json.loads((ws / "layout.json").read_text(encoding="utf-8"))["order"])


def test_untouched_seed_still_passes(ws):
    """种子里本来就有的死代码不是候选的责任 —— 拦它等于第一次提案必然失败。"""
    codes, _ = _codes(ws)
    assert "orphan-definition" not in codes


def test_no_parent_means_the_check_is_skipped(ws):
    """没有父本就没有"新增"可言(种子自己的 preflight 就是这种情况)。"""
    f = ws / "infinite_addition.py"
    f.write_text(f.read_text(encoding="utf-8") + NEW_ENGINE, encoding="utf-8")
    codes, _ = _codes(ws, parent=_Parent(None))
    assert "orphan-definition" not in codes


def test_a_string_keyed_dispatch_counts_as_a_reference(ws):
    """反例侧的调度器就是这么被调用的:launcher.tmpl 里
    `namespace["strict_search"](...)`。只扫 .py 会把种子自己的调度器判成孤儿。"""
    order = _order(ws)
    assert any(not n.endswith(".py") for n in order)   # 确实有非 .py 分段
    assert _orphan_definitions(ws, order, base="") == []


def test_a_new_function_nobody_calls_is_rejected(ws):
    f = ws / "infinite_addition.py"
    f.write_text(f.read_text(encoding="utf-8") + NEW_ENGINE, encoding="utf-8")
    codes, report = _codes(ws)
    assert "orphan-definition" in codes
    msg = [i.message for i in report.issues if i.code == "orphan-definition"][0]
    assert "infinite_shifted_order_offset_model" in msg
    assert "strict_search" in msg          # 告诉它该接到哪


def test_wiring_it_in_makes_it_pass(ws):
    """闸的出口必须存在,而且是一行的事 —— 否则它只是个障碍。"""
    f = ws / "infinite_addition.py"
    src = f.read_text(encoding="utf-8") + NEW_ENGINE
    src = src.replace(
        "def strict_search(source, target, seconds, profile):",
        "def strict_search(source, target, seconds, profile):\n"
        "    _ = infinite_shifted_order_offset_model", 1)
    f.write_text(src, encoding="utf-8")
    codes, _ = _codes(ws)
    assert "orphan-definition" not in codes


def test_a_cross_file_reference_counts(ws):
    """接线可以落在另一个分段里 —— 只在同一个文件里找会误杀。"""
    (ws / "infinite_addition.py").write_text(
        (ws / "infinite_addition.py").read_text(encoding="utf-8") + NEW_ENGINE,
        encoding="utf-8")
    fa = ws / "false.py"
    fa.write_text(fa.read_text(encoding="utf-8")
                  + "\n_HOOK = 'infinite_shifted_order_offset_model'\n",
                  encoding="utf-8")
    codes, _ = _codes(ws)
    assert "orphan-definition" not in codes


def test_the_issue_is_repairable(ws):
    """agent 要能在同一次会话里补一行接线就过,而不是整个提案作废。"""
    f = ws / "infinite_addition.py"
    f.write_text(f.read_text(encoding="utf-8") + NEW_ENGINE, encoding="utf-8")
    _, report = _codes(ws)
    issue = [i for i in report.issues if i.code == "orphan-definition"][0]
    assert issue.repairable is True


def test_names_already_in_the_parent_are_not_new(tmp_path):
    """只查**新增**的。父本里就有的名字不算这个候选写的。"""
    ws = tmp_path / "w"
    ws.mkdir()
    (ws / "a.py").write_text("def lonely():\n    return 1\n", encoding="utf-8")
    order = ["a.py"]
    assert _orphan_definitions(ws, order, base=None) == [("lonely", "a.py")]
    assert _orphan_definitions(
        ws, order, base="def lonely():\n    pass\n") == []
