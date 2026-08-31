"""字节上限必须量**提交产物**,不是源码之和。

2026-08-28 run18 实测:preflight 按源码之和判上限。对 v97 那种纯拼接体这两个数
恰好相等,对 gemma 谱系差 3.5 倍 —— 源码 1,452,297 装配 + LZMA 之后是 410,865。
于是每个候选都被判「超出上限 952,297 字节」,连**原封不动的种子**都过不了;而且
那条 issue 标着 repairable,agent 会照着去删 952 KB 根本不用删的代码。

一整轮里 5 次提案有 2 次死在这上面,而它们的改动本身是合法的。
"""

from __future__ import annotations

import os
import random
import shutil
import string
from pathlib import Path

import pytest

os.environ.setdefault(
    "ETP_SEED_STATE_DIR",
    str(Path(__file__).resolve().parents[1]
        / "experiments/etp_stage2/seeds/solo_v3_1"),
)

from evoharness.core.preflight import PreflightContext  # noqa: E402
from experiments.etp_stage2 import bundle  # noqa: E402
from experiments.etp_stage2.preflight import (  # noqa: E402
    Stage2SubmissionValidator,
    component_table,
)

SEED = Path("/Users/zhangkang/Documents/Projects/etp-work/seed_solo_v3_1")


@pytest.fixture()
def ws(tmp_path):
    if not (SEED / "layout.json").exists():
        pytest.skip("gemma 谱系种子不在")
    out = tmp_path / "ws"
    shutil.copytree(SEED, out)
    return out


def _validate(ws):
    return Stage2SubmissionValidator().validate(
        PreflightContext(parent=None, operator="revise", workdir=str(ws))
    )


def test_untouched_seed_passes(ws):
    """最基本的一条:什么都没改的种子必须能过。修复前它过不了。"""
    assert not _validate(ws).issues


def test_source_is_far_larger_than_the_submission(ws):
    """这个反差就是那个 bug 的全部内容 —— 拿源码判上限必然误杀。"""
    layout = __import__("json").loads(
        (ws / "layout.json").read_text(encoding="utf-8"))
    source = sum((ws / n).stat().st_size for n in layout["order"])
    produced = len(bundle.assemble(ws))
    assert source > bundle.HARD_CAP_BYTES     # 源码本身就超上限
    assert produced < bundle.HARD_CAP_BYTES   # 产物远在上限之内
    assert source / produced > 3


def test_incompressible_payload_is_still_rejected(ws):
    """放宽口径不等于放行:真的塞不下就要拒。"""
    rnd = random.Random(0)
    blob = "".join(rnd.choice(string.ascii_letters + string.digits)
                   for _ in range(400_000))
    f = ws / "infinite_addition.py"
    f.write_text(f.read_text(encoding="utf-8") + f"\n_PAD = '{blob}'\n",
                 encoding="utf-8")
    issues = _validate(ws).issues
    assert issues and issues[0].code == "over-byte-cap"
    assert "提交产物" in str(issues[0].message)


def test_compressible_payload_is_not_rejected(ws):
    """反过来也要成立:高度可压的内容不该被源码大小误杀。

    3.6 MB 重复注释压完几乎为零 —— 按源码判会拒,按产物判该放行。
    """
    f = ws / "infinite_addition.py"
    f.write_text(f.read_text(encoding="utf-8") + ("\n# " + "x" * 90) * 40000,
                 encoding="utf-8")
    codes = {i.code for i in _validate(ws).issues}
    assert "over-byte-cap" not in codes


def test_component_table_reports_submission_headroom(ws):
    """清单页脚是 agent 撞线时读的东西,它必须给产物余量而不是源码余量。"""
    table = component_table(ws)
    assert "提交产物" in table
    assert "源码合计" in table
    produced = len(bundle.assemble(ws))
    assert str(bundle.HARD_CAP_BYTES - produced) in table


def test_assemble_and_bundle_agree_below_the_cap(ws):
    assert len(bundle.assemble(ws)) == len(bundle.bundle(ws))
