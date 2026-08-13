"""P0 修复契约:证据生产接入真实运行链,且对搜索透明。"""

import json

import pytest

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
from evoharness.core.remote import EvalInfraError
from evoharness.evaluation import (
    EvidenceEnvelope,
    EvidenceJsonlSink,
    EvidenceProducingGrader,
    EvidenceProtocolError,
    EvidenceSinkError,
    FaultKind,
    ScoreNamespace,
    decide_search_use,
)

NS = ScoreNamespace(
    criterion_hash="c1", measurement_hash="m1",
    evaluator_hash="e1", universe_hash="u1",
)

INITIAL = """# EDIT-REGION-BEGIN
x = 0
x += 1
# EDIT-REGION-END
print(x)
"""


def _candidate():
    from evoharness.core import Candidate

    return Candidate(
        id="c0", code=INITIAL, generation=0, parent_id=None,
        island_idx=0, operator="seed",
    )


def _wrap(inner, tmp_path):
    return EvidenceProducingGrader(
        inner, namespace=NS, planned_units=3,
        sink=EvidenceJsonlSink(tmp_path / "evidence.jsonl"),
    )


def _envelopes(tmp_path):
    path = tmp_path / "evidence.jsonl"
    return [
        EvidenceEnvelope.from_json(json.loads(line))
        for line in path.read_text().splitlines()
    ]


class OkGrader:
    lineage_dir = None

    def grade(self, cand, workdir):
        return EvalReport(
            fitness=0.7,
            passed=True,
            n_units=3,
            trustworthy_units=3,
        )


class DownGrader:
    def grade(self, cand, workdir):
        raise EvalInfraError("eval service down")


def test_verdict_evidence_is_persisted_and_roundtrips(tmp_path):
    grader = _wrap(OkGrader(), tmp_path)
    cand = _candidate()
    report = grader.grade(cand, tmp_path)

    assert report.passed is True          # 投影原样返回
    (env,) = _envelopes(tmp_path)
    assert env.candidate_id == "c0"
    assert env.evaluation_valid and env.fault_kind is None
    assert env.coverage.complete          # planned=3, n_units=3
    assert env.namespace == NS
    assert cand.metadata["evidence_refs"] == [env.evidence_id]
    assert cand.metadata["search_use"]["rankable"] is True


def test_infra_error_persists_no_verdict_and_reraises(tmp_path):
    grader = _wrap(DownGrader(), tmp_path)
    cand = _candidate()
    with pytest.raises(EvalInfraError):
        grader.grade(cand, tmp_path)

    (env,) = _envelopes(tmp_path)
    assert env.evaluation_valid is False
    assert env.fault_kind is FaultKind.INFRA_ERROR
    assert env.fitness is None
    assert cand.metadata["evidence_refs"] == [env.evidence_id]
    assert cand.metadata["search_use"]["selectable"] is False


class PartialGrader:
    def grade(self, cand, workdir):
        return EvalReport(
            fitness=0.99,
            passed=True,
            n_units=1,
            trustworthy_units=1,
        )


def test_partial_measurement_is_recorded_but_not_selected_or_repaired(tmp_path):
    cand = _candidate()
    report = _wrap(PartialGrader(), tmp_path).grade(cand, tmp_path)
    assert report.passed is True       # 原始评估裁决不被搜索策略篡改
    cand.report = report
    assert cand.passed is False        # Candidate 选择资格来自 search_use
    use = cand.metadata["search_use"]
    assert use == decide_search_use(_envelopes(tmp_path)[0]).to_json()
    assert use["recordable"] is True
    assert use["selectable"] is False
    assert use["repairable"] is False
    assert use["reason"] == "partial-coverage"
    store = PopulationStore(PopulationConfig(num_islands=1))
    store.insert(cand)
    store.refresh_archive()
    assert store.best() is None
    assert store.latest_failed() is None
    assert store.island_view(0).passed_candidates == []


def test_quarantine_flag_is_typed_and_excludes_from_search():
    """Core 只认类型化的 quarantined=True;search_use 只是记录,不是控制面。"""
    cand = _candidate()
    cand.report = EvalReport(fitness=1.0, passed=True)
    cand.metadata["search_use"] = {"selectable": False}   # 仅记录,无效力
    assert cand.passed is True
    cand.metadata["quarantined"] = True
    assert cand.passed is False
    store = PopulationStore(PopulationConfig(num_islands=1))
    store.insert(cand)
    store.refresh_archive()
    assert store.best() is None
    assert store.latest_failed() is None      # 隔离 ≠ 可修复的失败
    assert store.island_view(0).passed_candidates == []


class ProtocolGrader:
    def grade(self, cand, workdir):
        return EvalReport(
            fitness=1.0, passed=False, fault_kind="infra_error"
        )


class CrashingGrader:
    def grade(self, cand, workdir):
        raise RuntimeError("grader bug")


def test_protocol_error_is_persisted_and_aborts_the_run(tmp_path):
    cand = _candidate()
    with pytest.raises(EvidenceProtocolError):
        _wrap(ProtocolGrader(), tmp_path).grade(cand, tmp_path)
    (env,) = _envelopes(tmp_path)
    assert env.fault_kind is FaultKind.PROTOCOL_ERROR
    assert env.evaluation_valid is False
    assert cand.metadata["evidence_refs"] == [env.evidence_id]


def test_unexpected_grader_exception_is_audited_before_abort(tmp_path):
    cand = _candidate()
    with pytest.raises(EvidenceProtocolError, match="grader bug"):
        _wrap(CrashingGrader(), tmp_path).grade(cand, tmp_path)
    (env,) = _envelopes(tmp_path)
    assert env.fault_kind is FaultKind.PROTOCOL_ERROR
    assert "grader bug" in env.missing_reasons[0]


def test_sink_resume_repairs_truncated_tail_and_deduplicates(tmp_path):
    path = tmp_path / "evidence.jsonl"
    sink = EvidenceJsonlSink(path)
    # Produce a real envelope without relying on sink internals.
    source = tmp_path / "source"
    _wrap(OkGrader(), source).grade(_candidate(), source)
    env = _envelopes(source)[0]
    sink.append(env)
    with open(path, "ab") as handle:
        handle.write(b'{"schema_version":1')

    resumed = EvidenceJsonlSink(path)
    assert resumed.append(env) == env.evidence_id
    assert len(path.read_text().splitlines()) == 1


def test_inner_attributes_pass_through(tmp_path):
    inner = OkGrader()
    grader = _wrap(inner, tmp_path)
    inner.lineage_dir = tmp_path / "lineage"
    assert grader.lineage_dir == tmp_path / "lineage"


def test_sink_failure_is_run_fatal_not_a_verdict(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")               # 文件占住父目录位置 → OSError
    grader = EvidenceProducingGrader(
        OkGrader(), namespace=NS, planned_units=3,
        sink=EvidenceJsonlSink(blocker / "evidence.jsonl"),
    )
    with pytest.raises(EvidenceSinkError):
        grader.grade(_candidate(), tmp_path)


# --- 透明性契约:装饰器只观察,不干预搜索 ---


class OutageOnThirdChild:
    lineage_dir = None

    def __init__(self):
        self.children_seen = 0

    def grade(self, cand, workdir):
        if cand.operator != "seed":
            self.children_seen += 1
            if self.children_seen == 3:
                raise EvalInfraError("eval service down")
        return EvalReport(
            fitness=1.0,
            passed=True,
            n_units=3,
            trustworthy_units=3,
        )


def _rewrite_transport(messages, model, **kw):
    code = "# EDIT-REGION-BEGIN\nx = 0\nx += 1\nx += 1\n# EDIT-REGION-END\nprint(x)\n"
    return LLMResponse(
        text=f"TITLE: t\nSUMMARY: s\n```python\n{code}```",
        model=model, cost=0.001,
    )


def _run_loop(grader, tmp_path):
    cfg = SearchConfig(
        num_generations=5, operators=["rewrite"], operator_probs=[1.0], seed=7,
    )
    pop_cfg = PopulationConfig(num_islands=1)
    store = PopulationStore(pop_cfg)
    loop = SearchLoop(
        cfg=cfg, pop_cfg=pop_cfg, store=store, grader=grader,
        llm=LLMClient(transport=_rewrite_transport, sleep=lambda s: None),
        prompt_builder=PromptBuilder("goal"),
        parent_selector=make_parent_selector(pop_cfg),
        inspiration_selector=InspirationSelector(pop_cfg),
        model_router=StaticRouter(["mock-model"]),
        workdir=tmp_path,
    )
    return loop.run(INITIAL), store


def test_producer_is_transparent_to_the_search(tmp_path):
    bare_report, bare_store = _run_loop(
        OutageOnThirdChild(), tmp_path / "bare"
    )
    wrapped_report, wrapped_store = _run_loop(
        _wrap(OutageOnThirdChild(), tmp_path / "wrapped"),
        tmp_path / "wrapped",
    )

    def outcomes(store):
        return sorted(
            (c.operator, c.code, c.fitness) for c in store.all_candidates()
        )

    assert outcomes(bare_store) == outcomes(wrapped_store)
    assert bare_report.evaluations == wrapped_report.evaluations
    assert (
        [h["status"] for h in bare_report.history]
        == [h["status"] for h in wrapped_report.history]
    )
    # 证据行数 = 入库候选数 + infra 丢弃数
    lines = (tmp_path / "wrapped" / "evidence.jsonl").read_text().splitlines()
    dropped = [
        h for h in wrapped_report.history if h["status"] == "infra_error"
    ]
    assert len(lines) == wrapped_store.count() + len(dropped)
