"""工作区自检脚本必须和 bundle() 算出同一个数。

为什么要有这个脚本:2026-08-27 实测,一个 agent 写了 158 行新代码,想确认会不会
超字节上限。工作区里没有任何东西能算出**打包后**的大小 —— 它找 selfcheck.sh、
找 preflight*,都不存在 —— 于是 `wc -c` 量源码,三个文件 648,852 字节,看着远超
500,000 的上限,`git checkout` 把 158 行全撤了,换成 3 行的改动,再打转 15 轮到
超时。真相是源码 1.4 MB 压完 411 KB、余量 89 KB。**它被一个错误的量尺杀死。**

为什么要有这个测试:脚本必须自包含(工作区是独立 git 仓库,没有 EvoHarness 可
import),所以它复制了一份装配逻辑。复制就会漂移,而漂移的后果是 agent 又拿到
一把错的尺子 —— 比没有尺子更糟,因为这次它会信。
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from experiments.etp_stage2 import bundle


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    src = bundle._HERE if hasattr(bundle, "_HERE") else None
    seed = tmp_path_factory.mktemp("ws")
    solver = _find_seed_solver()
    if solver is None:
        pytest.skip("找不到 gemma 谱系的 solver,跳过")
    bundle.split_gemma(solver, seed)
    return seed


def _find_seed_solver():
    from pathlib import Path

    for p in (
        Path("/Users/zhangkang/Documents/Projects/etp-work/solo_v3_1/solver.py"),
        Path("/Users/zhangkang/Downloads/2026-08-21_solo_google-gemma-4-31b-it_solver.py"),
    ):
        if p.exists():
            return p
    return None


def test_selfcheck_ships_with_the_workspace(workspace):
    assert (workspace / bundle.SELFCHECK_FILE).exists()


def test_selfcheck_is_not_part_of_the_genome(workspace):
    """它是量尺,不是被量的东西 —— 进了 order 就会被当成顶层引擎去装配。"""
    layout = json.loads((workspace / "layout.json").read_text(encoding="utf-8"))
    assert bundle.SELFCHECK_FILE not in layout["order"]


def test_selfcheck_agrees_with_bundle(workspace):
    """同一个工作区,两条实现必须报同一个字节数。"""
    expected = len(bundle.bundle(workspace))
    proc = subprocess.run(
        [sys.executable, bundle.SELFCHECK_FILE],
        cwd=workspace, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    line = [l for l in proc.stdout.splitlines() if l.startswith("submission")]
    assert line, proc.stdout
    reported = int(line[0].split()[1].replace(",", ""))
    assert reported == expected, (
        f"自检报 {reported},bundle() 报 {expected} —— 两条装配逻辑已经漂移"
    )


def test_selfcheck_reports_the_real_headroom(workspace):
    """余量必须按**提交产物**算,不是按源码 —— 那正是杀死那次提案的误算。"""
    proc = subprocess.run(
        [sys.executable, bundle.SELFCHECK_FILE],
        cwd=workspace, capture_output=True, text=True,
    )
    out = proc.stdout
    sub = int([l for l in out.splitlines() if l.startswith("submission")][0]
              .split()[1].replace(",", ""))
    room = int([l for l in out.splitlines() if l.startswith("headroom")][0]
               .split()[1].replace(",", ""))
    src = int([l for l in out.splitlines() if l.startswith("source")][0]
              .split()[1].replace(",", ""))
    assert room == bundle.HARD_CAP_BYTES - sub
    # 源码远大于上限而提交产物远小于 —— 这个反差就是那次误判的全部内容,
    # 所以两个数必须都打出来。
    assert src > bundle.HARD_CAP_BYTES > sub


def test_selfcheck_fails_loudly_on_broken_syntax(workspace, tmp_path):
    import shutil

    broken = tmp_path / "broken"
    shutil.copytree(workspace, broken)
    (broken / "infinite_addition.py").write_text("def f(:\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, bundle.SELFCHECK_FILE],
        cwd=broken, capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert "syntax error" in proc.stdout


# ─── 题目探针 ────────────────────────────────────────────────────────────────
#
# 尺寸自检只能回答「装得下吗」。第 18 轮之前 agent 没有任何办法回答「还解得出
# 吗」和「多解出了吗」—— 唯一的反馈是一轮完整评测,半小时起。于是有会话改完
# 代码直接提交,零增益,一整代作废。


def test_probe_paths_are_all_filled_in():
    """占位符漏一个,脚本就是语法错的字符串常量,而错误要到 agent 手里才现形。"""
    src = bundle.selfcheck_source()
    compile(src, "selfcheck.py", "exec")
    for slot in bundle._probe_paths():
        assert slot not in src


def test_probe_paths_follow_the_environment(monkeypatch):
    monkeypatch.setenv("ETP_PROBE_DIR", "/somewhere/probe")
    assert "'/somewhere/probe'" in bundle.selfcheck_source()


def test_probe_sets_exist_and_match_the_baseline():
    from pathlib import Path

    probe = Path(bundle.__file__).resolve().parent / "probe"
    base = json.loads((probe / "baseline.json").read_text(encoding="utf-8"))
    for name in ("guard", "gain"):
        rows = [l for l in (probe / f"{name}.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(rows) == base[name]["total"], f"{name} 行数与基线对不上"


def test_gain_probe_baseline_is_empty():
    """增益档的题是**筛出来的**种子解不出的行,所以基线必须是 0。

    第一版没筛,24 行里种子自己就解出 22 —— 「新解出」最多只能是 2,而 2 行
    是噪声。一个量不出变化的探针比没有探针更坏:它会让 agent 确信自己没进展。
    """
    from pathlib import Path

    probe = Path(bundle.__file__).resolve().parent / "probe"
    base = json.loads((probe / "baseline.json").read_text(encoding="utf-8"))
    assert base["gain"]["solved"] == []
    assert base["guard"]["solved"], "guard 基线不能为空,否则回归无从判定"


def test_probe_rows_carry_no_scoring_equations():
    """探针题面不进工作区,但它们仍然不能和记分的题重合 —— 重合就等于把
    评分集提前发给了 agent 去针对。"""
    import os
    from pathlib import Path

    os.environ.setdefault("ETP_ROTATING_ROWS", "60")
    os.environ.setdefault("ETP_ROTATING_SEED", "0")
    from experiments.etp_stage2 import grade

    probe = Path(bundle.__file__).resolve().parent / "probe"
    gain = [json.loads(l) for l in (probe / "gain.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    scored = {(r.get("eq1_id"), r.get("eq2_id")) for r in grade.fitness_set()}
    for generation in range(0, 14):
        scored |= {(r.get("eq1_id"), r.get("eq2_id"))
                   for r in grade.rotating_set(generation)}
    overlap = [r["id"] for r in gain
               if (r.get("eq1_id"), r.get("eq2_id")) in scored]
    assert not overlap, f"增益探针和记分行重合: {overlap}"


# ─── 探针必须区分「没测出来」和「测出来坏了」────────────────────────────────


def _probe_report(tmp_path, monkeypatch, rows, jobs="guard"):
    """把一批伪造的 run_local 记录喂给探针的 report(),拿回它印的东西。"""
    import io, json as _json, contextlib, zlib
    from pathlib import Path

    src = bundle.selfcheck_source()
    ws = tmp_path / "ws"
    ws.mkdir()
    mod = ws / "selfcheck.py"
    mod.write_text(src, encoding="utf-8")
    key = format(zlib.crc32(str(ws).encode()), "08x")
    out = Path("/tmp") / f"selfcheck_{key}"
    out.with_suffix(".jobs").write_text(jobs)
    out.with_suffix(f".{jobs}.jsonl").write_text(
        "\n".join(_json.dumps(r) for r in rows), encoding="utf-8")

    ns: dict = {"__file__": str(mod), "__name__": "selfcheck_under_test"}
    exec(compile(src, "selfcheck.py", "exec"), ns)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns["report"]()
    for suffix in (".jobs", f".{jobs}.jsonl"):
        out.with_suffix(suffix).unlink(missing_ok=True)
    return buf.getvalue()


def _guard_ids():
    from pathlib import Path

    probe = Path(bundle.__file__).resolve().parent / "probe"
    return [json.loads(l)["id"] for l in
            (probe / "guard.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]


def test_judge_outage_is_not_reported_as_regression(tmp_path, monkeypatch):
    """2026-08-28 实测:本机判题器有一段拒绝连接,12 行全部
    `infrastructure_error`、0.3 秒返回,探针报「REGRESSED 12」——
    而那个工作区和种子**逐字节相同**。agent 信了,回退了正确的改动,
    追着一个不存在的故障烧掉 27 轮。

    「没测出来」和「你改坏了」在种群里的后果完全不同,真正的评分器一直
    用 untrusted 把两者分开,而这个探针把那层丢了。"""
    rows = [{"problem_id": rid, "solved": False, "status": "infrastructure_error",
             "elapsed_seconds": 0.4,
             "error": "judge-v3 request failed: Connection refused"}
            for rid in _guard_ids()]
    out = _probe_report(tmp_path, monkeypatch, rows)
    assert "REGRESSED" not in out
    assert "NOT MEASURED 12" in out
    assert "broken probe" in out
    assert "do not change code because of it" in out


def test_a_real_regression_still_reports_as_one(tmp_path, monkeypatch):
    """放宽口径不等于放行:真掉了行还是要喊。"""
    ids = _guard_ids()
    rows = [{"problem_id": rid, "solved": i > 1, "status": "accepted",
             "elapsed_seconds": 9.0} for i, rid in enumerate(ids)]
    out = _probe_report(tmp_path, monkeypatch, rows)
    assert "REGRESSED 2" in out
    assert ids[0] in out and ids[1] in out


def test_instant_total_failure_is_named_as_a_broken_solver(tmp_path, monkeypatch):
    """全部秒挂 = 求解器起不来,和「掉了几行」的修法完全不同,不能打同一条消息。"""
    rows = [{"problem_id": rid, "solved": False, "status": "solver_exit",
             "elapsed_seconds": 0.3} for rid in _guard_ids()]
    out = _probe_report(tmp_path, monkeypatch, rows)
    assert "BROKEN 12" in out
    assert "not starting at all" in out
    assert "REGRESSED" not in out


def test_partial_outage_still_judges_the_rows_that_ran(tmp_path, monkeypatch):
    """判题器挂了一半 —— 跑起来的那些行照常判定,别整批作废。"""
    ids = _guard_ids()
    rows = [{"problem_id": rid, "solved": False, "status": "infrastructure_error",
             "elapsed_seconds": 0.4, "error": "Connection refused"}
            for rid in ids[:6]]
    rows += [{"problem_id": rid, "solved": True, "status": "accepted",
              "elapsed_seconds": 9.0} for rid in ids[6:]]
    out = _probe_report(tmp_path, monkeypatch, rows)
    assert "NOT MEASURED 6" in out
    assert "no regression" in out
    assert "broken probe" not in out
