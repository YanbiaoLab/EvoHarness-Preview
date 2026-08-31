"""改了代码就必须给出守卫探针的读数。

2026-08-28 实测,同一轮里两个候选:

  岛 1  跑了 `selfcheck.py`(尺寸+语法,0.5 秒)就提交,**从没跑过 start**。
        它丢掉了整个证明侧 —— 195 道题,fitness 0.8108 → 0.0699。
  岛 0  跑了探针,40 秒内看到 12 道守卫题全灭,继续修了 70 轮。

差别不在模型,在于这件事当时是**可选的**。

口径:只要求"测过并且没掉行",不要求"有增益"。没用但没害的候选该放行,
由 fitness 去淘汰它 —— 这条闸管的是安全,不是价值。
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import pytest

os.environ.setdefault(
    "ETP_SEED_STATE_DIR",
    str(Path(__file__).resolve().parents[1]
        / "experiments/etp_stage2/seeds/solo_v3_1"))

from experiments.etp_stage2.preflight import (  # noqa: E402
    _guard_evidence_issues,
    _probe_state_path,
)

SEED = Path("/Users/zhangkang/Documents/Projects/etp-work/seed_solo_v3_1")


@pytest.fixture()
def ws(tmp_path):
    if not (SEED / "layout.json").exists():
        pytest.skip("gemma 谱系种子不在")
    out = tmp_path / "ws"
    shutil.copytree(SEED, out)
    _clear_probe_state_for(out)          # 进来时先清,别继承别人的残留
    yield out
    _clear_probe_state_for(out)


def _clear_probe_state_for(ws):
    """清掉**测过这份内容**的所有探针状态,不只是这个路径下的。

    闸是内容寻址的,所以污染也是内容级的:手工在别处跑一次未改动种子的探针,
    就会在 /tmp 留下一份 .meta,用例再断言"没跑过"就会失败 —— 而失败的原因
    和用例要测的东西毫无关系。按路径清不够,得按内容清。
    """
    import hashlib
    from experiments.etp_stage2.bundle import assemble

    try:
        want = hashlib.sha256(assemble(ws)).hexdigest()
    except Exception:                                          # noqa: BLE001
        want = None
    prefixes = {_probe_state_path(ws)}
    if want:
        for meta in Path("/tmp").glob("selfcheck_*.meta"):
            try:
                if json.loads(meta.read_text(encoding="utf-8"))[
                        "solver_sha256"] == want:
                    prefixes.add(meta.with_suffix(""))
            except (OSError, ValueError, KeyError):
                pass
    for prefix in prefixes:
        for suffix in (".jobs", ".guard.jsonl", ".gain.jsonl", ".log", ".pid",
                       ".done", ".meta", ".solver.py"):
            prefix.with_suffix(suffix).unlink(missing_ok=True)


def _guard_ids():
    probe = Path(__file__).resolve().parents[1] / "experiments/etp_stage2/probe"
    return json.loads((probe / "baseline.json").read_text(encoding="utf-8"))[
        "guard"]["solved"]


def _write_probe(ws, rows, *, started=True, stamp=True):
    """伪造一次探针跑完的状态。

    `.meta` 里的产物哈希是新鲜度的判据 —— 探针测的必须就是**这份**代码。
    `stamp=False` 用来模拟"改完代码没重跑探针"。
    """
    import hashlib

    from experiments.etp_stage2.bundle import assemble

    state = _probe_state_path(ws)
    if started:
        state.with_suffix(".jobs").write_text("guard")
    state.with_suffix(".guard.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    if stamp:
        state.with_suffix(".meta").write_text(json.dumps({
            "solver_sha256": hashlib.sha256(assemble(ws)).hexdigest(),
            "solver_bytes": len(assemble(ws)),
        }), encoding="utf-8")
    return state


def _codes(ws):
    return [i.code for i in _guard_evidence_issues(ws, "t")]


def test_never_running_the_probe_is_blocked(ws):
    codes = _codes(ws)
    assert codes == ["guard-not-run"]


def test_the_message_says_which_command_to_run(ws):
    msg = _guard_evidence_issues(ws, "t")[0].message
    assert "selfcheck.py start --only guard" in msg
    assert "selfcheck.py wait" in msg


def test_starting_but_not_collecting_passes(ws):
    """尝试过就不拦。2026-08-29 实测:三个提案连续被判 guard-not-run,而
    agent 的结束语写着"the judge API was down" —— 它们跑了探针,探针连不上
    判题器、没写出结果,闸判成"你没跑"。拦掉的全是好提案。"""
    _probe_state_path(ws).with_suffix(".jobs").write_text("guard")
    assert _codes(ws) == []


def test_a_probe_that_produced_nothing_passes(ws):
    """有 .log 没结果 —— 同样是工具坏了,不是候选坏了。"""
    _probe_state_path(ws).with_suffix(".log").write_text("boom")
    assert _codes(ws) == []


def test_a_timed_out_guard_row_is_not_a_regression(ws):
    """守卫行本机实测都 ≤13 秒。探针是在 32 路求解器占着核的时候跑的,
    一行超时说明机器忙,不说明候选把它改坏了。"""
    ids=_guard_ids()
    rows=[{"problem_id": r, "solved": False, "status": "solver_timeout",
           "elapsed_seconds": 90.0} for r in ids[:4]]
    rows+=[{"problem_id": r, "solved": True, "status": "accepted",
            "elapsed_seconds": 9.0} for r in ids[4:]]
    _write_probe(ws, rows)
    assert _codes(ws) == []


def test_everything_timed_out_passes(ws):
    """全超时 = 什么都没测出来,放行。"""
    _write_probe(ws, [{"problem_id": r, "solved": False,
                       "status": "solver_timeout", "elapsed_seconds": 90.0}
                      for r in _guard_ids()])
    assert _codes(ws) == []


def test_a_clean_probe_passes(ws):
    _write_probe(ws, [{"problem_id": r, "solved": True, "status": "accepted",
                       "elapsed_seconds": 9.0} for r in _guard_ids()])
    assert _codes(ws) == []


def test_a_regression_is_blocked_and_named(ws):
    ids = _guard_ids()
    rows = [{"problem_id": r, "solved": i > 1, "status": "accepted",
             "elapsed_seconds": 9.0} for i, r in enumerate(ids)]
    _write_probe(ws, rows)
    issues = _guard_evidence_issues(ws, "t")
    assert [i.code for i in issues] == ["guard-regression"]
    assert ids[0] in issues[0].message
    assert "两倍计价" in issues[0].message


def test_a_judge_outage_does_not_block(ws):
    """测不出来不等于改坏了。判题器挂掉时把所有提案堵死,是把故障放大成停摆。"""
    _write_probe(ws, [{"problem_id": r, "solved": False,
                       "status": "infrastructure_error", "elapsed_seconds": 0.4,
                       "error": "Connection refused"} for r in _guard_ids()])
    assert _codes(ws) == []


def test_editing_after_the_probe_invalidates_it(ws):
    """新鲜度按**内容**判:改完代码那份读数就不再描述这份代码了。

    按时间判不够 —— 同一个目录里上一次探针留下的状态比基因组还新,时间戳会
    放行,而它描述的是另一份东西。起跑前校验就是这么被骗过一次的。
    """
    _write_probe(ws, [{"problem_id": r, "solved": True, "status": "accepted",
                       "elapsed_seconds": 9.0} for r in _guard_ids()])
    assert _codes(ws) == []
    (ws / "infinite_addition.py").write_text(
        (ws / "infinite_addition.py").read_text(encoding="utf-8") + "\n# edit\n",
        encoding="utf-8")
    assert _codes(ws) == ["guard-stale"]


def test_a_result_without_a_stamp_is_not_trusted(ws):
    """没有 .meta 就无从判断它测的是什么 —— 不放行。"""
    _write_probe(ws, [{"problem_id": r, "solved": True, "status": "accepted",
                       "elapsed_seconds": 9.0} for r in _guard_ids()],
                 stamp=False)
    assert _codes(ws) == ["guard-stale"]


def test_every_issue_is_repairable(ws):
    """agent 要能在同一次会话里补跑一次就过,而不是整个提案作废。"""
    assert all(i.repairable for i in _guard_evidence_issues(ws, "t"))


# ─── 闸必须按**内容**找读数,不是按路径 ──────────────────────────────────────


def test_the_gate_finds_a_probe_run_from_a_different_directory(ws, tmp_path):
    """这是这道闸从上线起就是死的那个原因。

    loop 的 preflight 把候选 materialize 到一个全新的
    `tempfile.TemporaryDirectory(prefix="evoharness_preflight_")` 里再校验,
    而 agent 是在 `.agent_work/evoharness-<id>-<rand>/` 里编辑和跑探针的。
    按工作区路径做键,两边永远对不上 —— agent 写在 A,闸去 B 找。

    2026-08-29 实测:六个会话各自跑了 5~20 次 selfcheck、全部 completed,
    却统统被判 `guard-not-run`,一整轮零候选。**路径是"这次在哪跑",内容才是
    "测的是什么"。**
    """
    import shutil

    # agent 在 A 目录跑探针
    _write_probe(ws, [{"problem_id": r, "solved": True, "status": "accepted",
                       "elapsed_seconds": 9.0} for r in _guard_ids()])
    assert _codes(ws) == []

    # preflight 在 B 目录校验同一份代码(逐字节相同,路径不同)
    other = tmp_path / "materialized_elsewhere"
    shutil.copytree(ws, other)
    assert _codes(other) == [], "换个目录就找不到读数 —— 闸按路径做键了"


def test_content_addressing_ignores_a_probe_of_different_code(ws, tmp_path):
    """反过来也要成立:别的代码的读数不能被当成这份代码的证据。"""
    import shutil

    _write_probe(ws, [{"problem_id": r, "solved": True, "status": "accepted",
                       "elapsed_seconds": 9.0} for r in _guard_ids()])
    other = tmp_path / "different_code"
    shutil.copytree(ws, other)
    (other / "infinite_addition.py").write_text(
        (other / "infinite_addition.py").read_text(encoding="utf-8")
        + "\n# a real change\n", encoding="utf-8")
    assert _codes(other) == ["guard-not-run"]
