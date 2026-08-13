"""Research Layer I-2: evidence envelopes, fault taxonomy, score namespaces.

模块布局(2026-08-13 二次重构,消除 evidence↔schema 运行时环):
strictjson(通用严格标量/冻结助手,唯一叶子)/ faults(词表)/
namespace(可比性)/ evidence(类型 + 不变量 + 内容寻址身份 + 编解码,
一个契约一个模块)/ policy(唯一政策表面)/ producer(可信构造、
落盘与 Grader 窄口)。
"""

from .evidence import (
    EVIDENCE_SCHEMA_VERSION,
    Coverage,
    EvidenceEnvelope,
    EvidenceProtocolError,
)
from .faults import FaultKind, classify_fault
from .namespace import NamespaceMismatch, ScoreNamespace
from .policy import (
    SearchUseDecision,
    decide_search_use,
    may_enter_population,
    may_rank,
    may_support_objective,
)
from .producer import (
    EvidenceJsonlSink,
    EvidenceProducingGrader,
    EvidenceSinkError,
    envelope_for_infra_error,
    envelope_for_missing,
    envelope_for_protocol_error,
    envelope_from_report,
    evidence_manifest,
    make_evidence_grader,
)

__all__ = [
    "Coverage",
    "EVIDENCE_SCHEMA_VERSION",
    "EvidenceEnvelope",
    "EvidenceJsonlSink",
    "EvidenceProducingGrader",
    "EvidenceProtocolError",
    "EvidenceSinkError",
    "FaultKind",
    "NamespaceMismatch",
    "ScoreNamespace",
    "SearchUseDecision",
    "classify_fault",
    "decide_search_use",
    "envelope_for_infra_error",
    "envelope_for_missing",
    "envelope_for_protocol_error",
    "envelope_from_report",
    "evidence_manifest",
    "make_evidence_grader",
    "may_enter_population",
    "may_rank",
    "may_support_objective",
]
