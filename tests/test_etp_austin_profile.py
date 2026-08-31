"""ETP_PROFILE=austin —— research_order5_hard 那条线的评分口径。

这条线和交付线(stage2)要的东西方向相反,所以它的四条性质每一条都得钉住:
种子在这里只解出 28/100,stage2 的每一道护栏在这里都会变成阻力。

profile 是**导入期**读的(TIERS 是模块级常量),所以每个用例都得在子进程里
设好环境再导入 —— monkeypatch.setenv 之后 reload 也行,但 reload grade 会连带
重置 official 的模块级状态,比子进程脆。
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _in_profile(body: str, **env: str) -> dict:
    """在 austin profile 下跑一段代码,拿回它 print 出来的那行 JSON。"""
    script = textwrap.dedent(
        """
        import json, sys
        sys.path.insert(0, %r)
        from experiments.etp_stage2 import grade as G, rung0
        from experiments.etp_stage2 import bundle as B
        """ % str(ROOT)
    ) + textwrap.dedent(body)
    base = {
        "ETP_PROFILE": "austin",
        "ETP_SCORING": "official",
        "ETP_ROTATING_ROWS": "0",
        "PYTHONPATH": str(ROOT),
    }
    import os
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        cwd=ROOT, env=os.environ | base | env,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_single_tier_is_plain_count():
    """单档等权 = 纯计数。用户要的是「自管解题多少」,不是加权分。"""
    got = _in_profile("""
        rows = G.fitness_set()
        vec = [i < 28 for i in range(len(rows))]
        t = G.tier_scores(rows, vec, None, set())
        print(json.dumps({
            "tiers": list(G.TIERS), "rows": len(rows),
            "strata": sorted({r["stratum"] for r in rows}),
            "fitness": G.combine(t),
        }))
    """)
    assert got["tiers"] == ["austin"]
    assert got["rows"] == 100
    assert got["strata"] == ["austin"]
    # 28 解出 → 恰好 0.28。任何加权、任何补充档都会让这个数偏离。
    assert got["fitness"] == pytest.approx(0.28)


def test_no_ratchet_no_rotating_no_sealed():
    """不建基准、不轮转、不跑密封集 —— 三样在这条线上都只会添乱。

    最要紧的是**不建基准**:第一个候选跑完就会把它解出的那 28 行变成守卫行,
    第二代起一次抖动就判回归。在只解出 28% 的靶子上那会把探索性改动全判死。
    """
    got = _in_profile("""
        import inspect
        src = inspect.getsource(G.grade_workspace)
        print(json.dumps({
            "rotating": len(G.rotating_set(0)),
            "hidden": list(G.hidden_sets()),
            # publish_reference 的调用点必须被 AUSTIN 挡住
            "publish_guarded": "not AUSTIN and not regressed" in src,
        }))
    """)
    assert got["rotating"] == 0
    assert got["hidden"] == []
    assert got["publish_guarded"], "publish_reference 没被 AUSTIN 挡住 —— 棘轮会回来"


def test_missing_reference_means_no_regression():
    """没有参照物时 guard 段不产生 suspects,fitness 就是原始分。

    run_guard 里那个 `if reference else []` 是承重的:没有它,全集都进 guard_rows,
    任何一道没解出都会被当成退步,而此时根本无从谈退步。
    """
    got = _in_profile("""
        ref, info = G._reference()
        print(json.dumps({"ref_is_none": ref is None, "why": info.get("reference")}))
    """, ETP_REFERENCE_PATH="/nonexistent/austin_ref.json")
    assert got["ref_is_none"] is True
    assert got["why"] == "missing"


def test_byte_cap_overridable_but_defaults_to_official():
    """字节上限可覆盖 —— 但不设变量时必须还是官方那 500,000。"""
    lifted = _in_profile('print(json.dumps({"cap": B.HARD_CAP_BYTES}))',
                         ETP_BYTE_CAP="2000000")
    assert lifted["cap"] == 2_000_000
    default = _in_profile('print(json.dumps({"cap": B.HARD_CAP_BYTES}))',
                          ETP_BYTE_CAP="")
    assert default["cap"] == 500_000, "没设变量时上限漂了 —— 交付线会跟着漂"


def test_austin_problems_are_scanned_for_hardcoding():
    """靶集的题面必须进反作弊扫描。

    不设退化限制、不设字节上限之后,「把 100 道题面连答案抄进去」是这条线上
    最便宜的作弊路径,也是唯一还拦得住的那道闸。
    """
    got = _in_profile("""
        from pathlib import Path
        ds = Path(%r) / "experiments/etp_stage2/datasets"
        eqs = rung0.scoring_equations(ds)
        rows = [json.loads(l) for l in (ds / "austin100.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()]
        norm = rung0._norm_equation
        covered = sum(1 for r in rows
                      if norm(r["equation1"]) in eqs and norm(r["equation2"]) in eqs)
        print(json.dumps({"covered": covered, "total": len(rows)}))
    """ % str(ROOT))
    assert got["covered"] == got["total"] == 100


def test_targets_are_the_ten_order5_austin_laws():
    """靶律就是 blueprint 确认的那 10 条 order-5 Austin 律。

    钉住它是因为**这个集合决定了什么方法可能奏效**:Austin 律「有无限模型、
    无非平凡有限模型」,所以有限枚举无论跑多久都不可能命中,能拿分的只有
    无限反模型构造。数据集换了而这条断言还绿,就说明换的不是 Austin 集。
    """
    ds = ROOT / "experiments/etp_stage2/datasets/austin100.jsonl"
    rows = [json.loads(l) for l in ds.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    assert sorted({r["eq2_id"] for r in rows}) == [
        4916, 15535, 17522, 20034, 22455, 22818, 25964, 28770, 30591, 41082]
    # 深层引擎只认 ◇,喂 * 会**静默失败** —— 不报错,只是一道都解不出。
    for r in rows:
        assert "◇" in r["equation1"] and "*" not in r["equation1"], r["id"]
        assert "◇" in r["equation2"] and "*" not in r["equation2"], r["id"]
