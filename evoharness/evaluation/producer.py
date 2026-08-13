"""Evidence 的可信构造与持久化(Grader 窄口)。

"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evoharness.core.population import EvalReport
from evoharness.core.remote import EvalInfraError

from .evidence import (
    EVIDENCE_SCHEMA_VERSION,
    Coverage,
    EvidenceEnvelope,
    EvidenceProtocolError,
    canonical_evidence_json,
    decode_evidence,
)
from .faults import NO_VERDICT_FAULTS, FaultKind, classify_fault
from .namespace import ScoreNamespace
from .policy import decide_search_use

if TYPE_CHECKING:
    from evoharness.contracts import TaskSpec
    from evoharness.core.interfaces import Grader
    from evoharness.core.population import Candidate


# --- trusted construction paths -------------------------------------------


def _report_provenance(report: EvalReport) -> dict[str, Any]:
    return {
        "stage_reached": report.stage_reached,
        "execution_time": report.execution_time,
        "sem": report.sem,
        "n_units": report.n_units,
        "trustworthy_units": report.trustworthy_units,
        "fault": report.fault,
        "raw_report_hash": hashlib.sha256(
            json.dumps(
                report.to_json(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def envelope_from_report(
    report: EvalReport,
    *,
    candidate_id: str,
    namespace: ScoreNamespace,
    planned_units: int,
) -> EvidenceEnvelope:
    if not isinstance(report, EvalReport):
        raise EvidenceProtocolError(
            "grader must return EvalReport",
            fault_kind=FaultKind.PROTOCOL_ERROR,
        )
    try:
        normalized_report = EvalReport.from_json(report.to_json())
    except (TypeError, ValueError) as exc:
        raise EvidenceProtocolError(
            f"invalid EvalReport: {exc}",
            fault_kind=FaultKind.PROTOCOL_ERROR,
        ) from exc
    if normalized_report.to_json() != report.to_json():
        raise EvidenceProtocolError(
            "EvalReport changed during strict wire validation",
            fault_kind=FaultKind.PROTOCOL_ERROR,
        )
    report = normalized_report
    fault = classify_fault(
        passed=report.passed,
        fault_kind=report.fault_kind,
    )
    if fault in NO_VERDICT_FAULTS:
        raise EvidenceProtocolError(
            f"{fault} cannot be returned as a verdict report",
            fault_kind=fault,
        )
    trustworthy_units = report.trustworthy_units
    if trustworthy_units is None:
        if planned_units > 0 and fault is None:
            raise EvidenceProtocolError(
                "successful declared measurement must report trustworthy_units",
                fault_kind=FaultKind.PROTOCOL_ERROR,
            )
        trustworthy_units = report.n_units if planned_units == 0 else 0
    return EvidenceEnvelope(
        evidence_id="auto",
        candidate_id=candidate_id,
        namespace=namespace,
        evaluation_valid=True,
        admissible=fault is not FaultKind.INVALID_CANDIDATE,
        fitness=report.fitness,
        fault_kind=fault,
        coverage=Coverage(
            planned_units=planned_units,
            executed_units=report.n_units,
            trustworthy_units=trustworthy_units,
        ),
        observations={
            "visible_metrics": dict(report.visible_metrics),
            "hidden_metrics": dict(report.hidden_metrics),
        },
        budget_used_usd=report.eval_cost_usd,
        provenance=_report_provenance(report),
        artifacts_ref=report.artifacts_ref,
    )


def _no_verdict_envelope(
    *,
    candidate_id: str,
    namespace: ScoreNamespace,
    planned_units: int,
    fault_kind: FaultKind,
    reason: str,
) -> EvidenceEnvelope:
    if fault_kind not in NO_VERDICT_FAULTS:
        raise ValueError("no-verdict envelope requires a no-verdict fault")
    normalized_reason = reason.strip() if isinstance(reason, str) else ""
    if not normalized_reason:
        normalized_reason = fault_kind.value
    return EvidenceEnvelope(
        evidence_id="auto",
        candidate_id=candidate_id,
        namespace=namespace,
        evaluation_valid=False,
        admissible=None,
        fitness=None,
        fault_kind=fault_kind,
        coverage=Coverage(planned_units, 0, 0),
        missing_reasons=(normalized_reason,),
    )


def envelope_for_infra_error(**kwargs) -> EvidenceEnvelope:
    return _no_verdict_envelope(
        fault_kind=FaultKind.INFRA_ERROR, reason=kwargs.pop("error"), **kwargs
    )


def envelope_for_missing(**kwargs) -> EvidenceEnvelope:
    return _no_verdict_envelope(
        fault_kind=FaultKind.MISSING, reason=kwargs.pop("reason"), **kwargs
    )


def envelope_for_protocol_error(**kwargs) -> EvidenceEnvelope:
    return _no_verdict_envelope(
        fault_kind=FaultKind.PROTOCOL_ERROR,
        reason=kwargs.pop("error"),
        **kwargs,
    )


# --- durable sink ----------------------------------------------------------


class EvidenceSinkError(RuntimeError):
    """The audit chain broke; continuing would make the run untrustworthy."""


class EvidenceJsonlSink:
    """Durable, append-only, content-addressed JSONL evidence store.

    Existing complete lines are validated on open. A crash-truncated final
    line is removed; corruption before the tail is fatal. Re-appending the
    same content-addressed envelope is idempotent.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._records: dict[str, str] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = self.path.read_bytes()
            if not raw:
                return
            if not raw.endswith(b"\n"):
                boundary = raw.rfind(b"\n") + 1
                tail = raw[boundary:]
                try:
                    envelope = decode_evidence(
                        json.loads(tail.decode("utf-8"))
                    )
                except Exception:
                    with open(self.path, "r+b") as handle:
                        handle.truncate(boundary)
                        handle.flush()
                        os.fsync(handle.fileno())
                    raw = raw[:boundary]
                else:
                    with open(self.path, "ab") as handle:
                        handle.write(b"\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    raw += b"\n"
                    self._remember(envelope)
            for line_number, raw_line in enumerate(raw.splitlines(), start=1):
                if not raw_line.strip():
                    continue
                try:
                    envelope = decode_evidence(
                        json.loads(raw_line.decode("utf-8"))
                    )
                    self._remember(envelope)
                except Exception as exc:
                    raise EvidenceSinkError(
                        f"invalid evidence line {line_number}: {exc}"
                    ) from exc
        except EvidenceSinkError:
            raise
        except OSError as exc:
            raise EvidenceSinkError(
                f"evidence sink read failed: {exc}"
            ) from exc

    def _remember(self, envelope: EvidenceEnvelope) -> None:
        canonical = canonical_evidence_json(envelope)
        existing = self._records.get(envelope.evidence_id)
        if existing is not None and existing != canonical:
            raise EvidenceSinkError(
                f"conflicting payload for evidence {envelope.evidence_id}"
            )
        self._records[envelope.evidence_id] = canonical

    def append(self, envelope: EvidenceEnvelope | dict) -> str:
        try:
            normalized = (
                envelope
                if isinstance(envelope, EvidenceEnvelope)
                else decode_evidence(envelope)
            )
            line = canonical_evidence_json(normalized)
            with self._lock:
                existing = self._records.get(normalized.evidence_id)
                if existing is not None:
                    if existing != line:
                        raise EvidenceSinkError(
                            "conflicting payload for evidence "
                            f"{normalized.evidence_id}"
                        )
                    return normalized.evidence_id
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                self._records[normalized.evidence_id] = line
            return normalized.evidence_id
        except EvidenceSinkError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise EvidenceSinkError(
                f"evidence sink write failed: {exc}"
            ) from exc


# --- the grader chokepoint -------------------------------------------------


class EvidenceProducingGrader:
    """Persist evidence, attach provenance, and flag quarantine.
    """

    def __init__(
        self,
        inner: "Grader",
        *,
        namespace: ScoreNamespace,
        planned_units: int,
        sink: EvidenceJsonlSink,
    ):
        self._inner = inner
        self._namespace = namespace
        self._planned_units = planned_units
        self._sink = sink

    def _record(self, cand: "Candidate", envelope: EvidenceEnvelope) -> None:
        evidence_id = self._sink.append(envelope)
        refs = cand.metadata.setdefault("evidence_refs", [])
        if not isinstance(refs, list):
            raise EvidenceSinkError("candidate evidence_refs must be a list")
        if evidence_id not in refs:
            refs.append(evidence_id)
        decision = decide_search_use(envelope)
        cand.metadata["search_use"] = decision.to_json()
        if decision.quarantine:
            cand.metadata["quarantined"] = True

    def grade(self, cand: "Candidate", workdir: Path) -> EvalReport:
        try:
            report = self._inner.grade(cand, workdir)
        except EvalInfraError as exc:
            self._record(
                cand,
                envelope_for_infra_error(
                    candidate_id=cand.id,
                    namespace=self._namespace,
                    planned_units=self._planned_units,
                    error=str(exc),
                ),
            )
            raise
        except EvidenceSinkError:
            raise
        except Exception as exc:
            self._record(
                cand,
                envelope_for_protocol_error(
                    candidate_id=cand.id,
                    namespace=self._namespace,
                    planned_units=self._planned_units,
                    error=f"grader raised {type(exc).__name__}: {exc}",
                ),
            )
            raise EvidenceProtocolError(
                f"grader raised {type(exc).__name__}: {exc}",
                fault_kind=FaultKind.PROTOCOL_ERROR,
            ) from exc

        try:
            envelope = envelope_from_report(
                report,
                candidate_id=cand.id,
                namespace=self._namespace,
                planned_units=self._planned_units,
            )
        except (EvidenceProtocolError, TypeError, ValueError) as exc:
            self._record(
                cand,
                envelope_for_protocol_error(
                    candidate_id=cand.id,
                    namespace=self._namespace,
                    planned_units=self._planned_units,
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
            if isinstance(exc, EvidenceProtocolError):
                raise
            raise EvidenceProtocolError(
                str(exc), fault_kind=FaultKind.PROTOCOL_ERROR
            ) from exc

        self._record(cand, envelope)
        return report

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def make_evidence_grader(
    grader: "Grader",
    *,
    task_spec: "TaskSpec",
    run_dir: Path,
) -> EvidenceProducingGrader:
    expected_namespace = ScoreNamespace.from_task(task_spec)
    expected_path = Path(run_dir) / "evidence.jsonl"
    if isinstance(grader, EvidenceProducingGrader):
        if (
            grader._namespace != expected_namespace
            or grader._planned_units != task_spec.measurement.planned_units
            or grader._sink.path != expected_path
        ):
            raise ValueError(
                "grader is already evidence-wrapped for a different task or run"
            )
        return grader
    return EvidenceProducingGrader(
        grader,
        namespace=expected_namespace,
        planned_units=task_spec.measurement.planned_units,
        sink=EvidenceJsonlSink(expected_path),
    )


def evidence_manifest(task_spec: "TaskSpec") -> dict:
    planned_units = task_spec.measurement.planned_units
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "path": "evidence.jsonl",
        "namespace": ScoreNamespace.from_task(task_spec).to_json(),
        "planned_units": planned_units,
        "coverage_mode": (
            "declared" if planned_units > 0 else "legacy_undeclared"
        ),
    }


__all__ = [
    "EvidenceJsonlSink",
    "EvidenceProducingGrader",
    "EvidenceSinkError",
    "envelope_for_infra_error",
    "envelope_for_missing",
    "envelope_for_protocol_error",
    "envelope_from_report",
    "evidence_manifest",
    "make_evidence_grader",
]
