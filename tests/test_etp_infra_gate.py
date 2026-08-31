"""基础设施故障的两道防线:沙箱补跑,和整场可信度闸。

2026-08-26 第 18 轮起跑时,种子评测在并发 60 下 622 行里有 225 行是
504 Gateway Time-out。逐行的 untrusted 标记本身是对的(那 225 行不计进退),
但不可信行仍然计入档内 total —— 分数因此从应有的 ~0.80 塌到 0.4161,而那是
**基准**。

真正的危害不在那一个数,在它之后:下一个候选只要运气好、跑出一次干净的 ~0.80,
就会显示成 +0.39 的巨大突破,而选择、归档全看这个数。基础设施抽奖驱动的漂移,
长得跟进化一模一样。
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("ETP_SCORING", "official")

from evoharness.evaluation.faults import FaultKind  # noqa: E402
from experiments.etp_stage2 import official  # noqa: E402


def _res(rid, *, solved=False, infra=0):
    return official.RunResult(rid, solved, "false" if solved else None,
                              1 if solved else 0, 1.0, infra_errors=infra)


def test_retry_only_touches_infra_rows(monkeypatch):
    """补跑只碰基础设施故障的行 —— 解出的和真没解出的都不再动。

    这个边界是承重的:补跑若能改写「真的没解出」,就成了给候选发免费重试,
    分数会随重试次数单调上涨。
    """
    rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    calls = []

    def fake(solver, batch, *, timeout_s=None, concurrency=None):
        calls.append([r["id"] for r in batch])
        if len(calls) == 1:
            return [_res("a", solved=True), _res("b", infra=1), _res("c")]
        return [_res(r["id"], solved=True) for r in batch]   # 补跑全解出

    monkeypatch.setattr(official, "_sandbox_pass", fake)
    out = {r.row_id: r for r in official.run_many_sandbox(b"x", rows)}
    assert calls == [["a", "b", "c"], ["b"]]        # 只补了 b
    assert out["a"].solved and out["b"].solved      # b 被补跑改写
    assert not out["c"].solved                      # c 真没解出,不许被洗白


def test_retry_keeps_original_when_retry_also_fails(monkeypatch):
    """补跑又故障 → 保留原记录(仍标 infra),别洗成一次干净的失败。"""
    rows = [{"id": "a"}]

    def fake(solver, batch, *, timeout_s=None, concurrency=None):
        return [_res("a", infra=1)]

    monkeypatch.setattr(official, "_sandbox_pass", fake)
    out = official.run_many_sandbox(b"x", rows)
    assert out[0].infra_errors == 1
    assert not out[0].trustworthy


def test_retry_survives_a_failing_retry_pass(monkeypatch):
    """补跑本身抛异常时不能把整批打死 —— 保留首轮结论交给上层的闸。"""
    rows = [{"id": "a"}]
    calls = []

    def fake(solver, batch, *, timeout_s=None, concurrency=None):
        calls.append(1)
        if len(calls) == 1:
            return [_res("a", infra=1)]
        raise official.OfficialRunnerError("补跑也挂了")

    monkeypatch.setattr(official, "_sandbox_pass", fake)
    out = official.run_many_sandbox(b"x", rows)
    assert out[0].infra_errors == 1


def test_retry_preserves_input_order(monkeypatch):
    """顺序是承重的:pass_vector 按行序比对参照物,乱序会把无辜的题记成回归。"""
    rows = [{"id": x} for x in ("d", "a", "c", "b")]

    def fake(solver, batch, *, timeout_s=None, concurrency=None):
        return [_res(r["id"], infra=1 if r["id"] == "c" else 0) for r in batch]

    monkeypatch.setattr(official, "_sandbox_pass", fake)
    assert [r.row_id for r in official.run_many_sandbox(b"x", rows)] == \
        ["d", "a", "c", "b"]


@pytest.mark.parametrize("untrusted,total,invalid", [
    (225, 622, True),     # 实测的那次:36%
    (24, 562, False),     # 实测的干净跑批:4%
    (62, 622, False),     # 恰好 9.97%,不触发
    (63, 622, True),      # 刚过 10%
    (0, 622, False),
])
def test_invalid_ratio_boundary(untrusted, total, invalid):
    """阈值边界。干净跑批 4%、故障跑批 36%,差一个数量级,10% 两边都留足余量。"""
    assert (untrusted / total > official_ratio()) is invalid


def official_ratio() -> float:
    from experiments.etp_stage2 import grade
    return grade.INFRA_INVALID_RATIO


def test_gate_actually_fires_in_grade_workspace(monkeypatch, tmp_path):
    """闸真的接在 grade_workspace 上,不只是算术对。

    前两次事故(`_MAIN` 写死、run18.sh 只 echo 不 export)都是「部件对、接线错」,
    而且都要等真跑起来才暴露。所以这里连着 grade_workspace 一起测:
    构造一次「大多数行基础设施故障」的评测,断言它走 fault 而不是低分。
    """
    from evoharness.serve import Grade, GradeContext

    from experiments.etp_stage2 import grade as G
    from experiments.etp_stage2 import rung0

    rows = G.fitness_set()[:20]
    monkeypatch.setattr(G, "fitness_set", lambda: rows)
    monkeypatch.setattr(G, "rotating_set", lambda gen: [])
    monkeypatch.setattr(G, "_reference", lambda: (None, {"reference": "none"}))
    monkeypatch.setattr(rung0, "run", lambda *a, **k: rung0.Rung0Report(
        True, [rung0.Check("bundle", True)], solver_bytes=1, headroom_bytes=1))
    monkeypatch.setattr(G, "bundle", lambda ws: b"x")

    # 18/20 行 504 —— 90%,远超 10% 上限
    def infra_heavy(solver_bytes, batch, *, timeout_s=None, **kw):
        return [official.RunResult(r["id"], False, None, 0, 1.0,
                                   reason="SandboxException: 504",
                                   infra_errors=1 if i >= 2 else 0)
                for i, r in enumerate(batch)]

    monkeypatch.setattr(official, "run_many", infra_heavy)

    from evoharness.serve import InfraError

    with pytest.raises(InfraError) as exc:
        G.grade_workspace(tmp_path,
                          GradeContext(candidate_id="t", workdir=tmp_path))
    assert "基础设施故障" in str(exc.value)
    # 诊断必须跟着异常走。InfraError 只带一个字符串,visible_metrics 到不了
    # 任何地方 —— 08-28 那次崩溃之后连"为什么不可信"都查不出来。
    assert "原因样本" in str(exc.value)
    assert "90%" in str(exc.value)


def test_the_gate_reaches_the_loop_instead_of_killing_it(monkeypatch, tmp_path):
    """闸必须走 InfraError 那条路,而不是 return 一个 no-verdict 的 Grade。

    2026-08-28 第 18 轮死在这上面。当时闸 `return Grade(fault_kind="infra")`:

      · "infra" 不在 FaultKind 词表里 → classify_fault 归成 UNKNOWN
      · UNKNOWN ∈ NO_VERDICT_FAULTS   → producer 抛 EvidenceProtocolError
      · 没有人接                       → 进程死

    而且**把字符串改成 "infra_error" 也救不了**:INFRA_ERROR 同样在
    NO_VERDICT_FAULTS 里,同样抛,同样没人接。返回 Grade 这条路的终点就是崩溃,
    与写哪个字符串无关。唯一能到达 loop 的 infra_streak 的通道是 raise
    InfraError —— `InfraError` 的 docstring 一直是这么写的。

    上一版测试断言 `g.fault_kind == "infra"`,把错的字符串锁死了,而且只测到
    grade.py 的出口为止,从没问过下一站接不接得住。单元测试全绿,通路是死的。
    """
    from evoharness.evaluation.faults import NO_VERDICT_FAULTS, classify_fault
    from evoharness.serve import InfraError

    # 两个字符串都是死路 —— 这就是为什么必须换机制,不是换字面量。
    for wire in ("infra", FaultKind.INFRA_ERROR.value):
        assert classify_fault(passed=False, fault_kind=wire) in NO_VERDICT_FAULTS

    # 而 InfraError 会被 grader 适配层转成 loop 认识的 EvalInfraError。
    from evoharness.core.remote import EvalInfraError
    from evoharness.runtime.grading import WorkspaceGradeFnGrader

    def boom(candidate_dir, ctx):
        raise InfraError("沙箱池挂了")

    grader = WorkspaceGradeFnGrader(boom)
    cand = _stub_candidate(tmp_path)
    with pytest.raises(EvalInfraError) as exc:
        grader.grade(cand, tmp_path / "gen")
    assert "沙箱池挂了" in str(exc.value)


def test_grade_no_longer_returns_a_no_verdict_fault_kind():
    """堵住回头路:grade.py 里不许再出现 no-verdict 的 fault_kind 字面量。"""
    import re
    from pathlib import Path

    from evoharness.evaluation.faults import NO_VERDICT_FAULTS

    src = (Path(__file__).resolve().parents[1]
           / "experiments/etp_stage2/grade.py").read_text(encoding="utf-8")
    literals = set(re.findall(r'fault_kind\s*=\s*"([^"]+)"', src))
    banned = {k.value for k in NO_VERDICT_FAULTS} | {"infra"}
    assert not (literals & banned), (
        f"这些 fault_kind 返回出去只会让 producer 抛异常: "
        f"{sorted(literals & banned)} —— 该 raise InfraError")


def _stub_candidate(tmp_path):
    from evoharness.core.population import Candidate
    from evoharness.core.workspace import GitWorkspace

    d = tmp_path / "seed"
    d.mkdir(exist_ok=True)
    (d / "main.py").write_text("x = 1\n", encoding="utf-8")
    ws = GitWorkspace.from_directory(d, main_file="main.py")
    return Candidate(
        id=Candidate.new_id(), code=ws.serialize(), workspace_kind=ws.kind,
        generation=0, parent_id=None, island_idx=0, operator="seed",
        change_title="stub")


def test_clean_run_is_not_gated(monkeypatch, tmp_path):
    """干净跑批不能被闸误伤 —— 否则整套评测直接瘫掉。"""
    from evoharness.serve import GradeContext

    from experiments.etp_stage2 import grade as G
    from experiments.etp_stage2 import rung0

    rows = G.fitness_set()[:20]
    monkeypatch.setattr(G, "fitness_set", lambda: rows)
    monkeypatch.setattr(G, "rotating_set", lambda gen: [])
    monkeypatch.setattr(G, "_reference", lambda: (None, {"reference": "none"}))
    monkeypatch.setattr(rung0, "run", lambda *a, **k: rung0.Rung0Report(
        True, [rung0.Check("bundle", True)], solver_bytes=1, headroom_bytes=1))
    monkeypatch.setattr(G, "bundle", lambda ws: b"x")
    monkeypatch.setattr(official, "run_many",
                        lambda s, batch, **kw: [
                            official.RunResult(r["id"], True, "false", 1, 1.0)
                            for r in batch])

    g = G.grade_workspace(tmp_path, GradeContext(candidate_id="t", workdir=tmp_path))
    assert g.fault_kind != FaultKind.INFRA_ERROR.value
    assert g.passed is True
