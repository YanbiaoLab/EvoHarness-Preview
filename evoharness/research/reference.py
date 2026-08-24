import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from evoharness.core.checkpoint import append_jsonl, atomic_write_json
from evoharness.evaluation import ScoreNamespace


class ReferenceConflict(RuntimeError):
    """expected_current 不匹配:别人先晋升了,重读再决定。"""

@dataclass(frozen=True)
class ReferenceRecord:
    candidate_id: str
    evidence_id: str
    namespace: ScoreNamespace
    fitness: float|None
    policy_hash: str
    assessor_hash: str
    reasons: tuple[str, ...]
    promoted_at: float
    previous_candidate_id: str | None = None

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["namespace"] = self.namespace.to_json()
        payload["reasons"] = list(self.reasons)
        return {"schema_version": 1, **payload}

    @classmethod
    def from_json(cls, d: dict) -> "ReferenceRecord":
        payload = {k: v for k, v in d.items() if k != "schema_version"}
        payload["namespace"] = ScoreNamespace.from_json(payload["namespace"])
        payload["reasons"] = tuple(payload.get("reasons", ()))
        return cls(**payload)

class ReferenceStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._history =    self.path.with_name(self.path.stem + "_history.jsonl")

        self._lock = threading.Lock()

    def current(self) -> ReferenceRecord | None:
        if not self.path.exists():
            return None
        return ReferenceRecord.from_json(json.loads(self.path.read_text()))

    def promote(
            self,
            record: ReferenceRecord,
            *,
            expected_current: str | None,
            now= time.time,
    ) -> ReferenceRecord:
        with self._lock:
            current = self.current()
            current_id = current.candidate_id if current else None
            if current_id != expected_current:
                raise ReferenceConflict(
                    f"reference moved: expected {expected_current!r}, "
                    f"found {current_id!r}"
                )
            if current is not None:
                current.namespace.require_comparable(record.namespace)
            stamped = ReferenceRecord(
                **{
                    **asdict(record),
                    "namespace": record.namespace,
                    "reasons": record.reasons,
                    "promoted_at": record.promoted_at or now(),
                    "previous_candidate_id": current_id,
                }
            )
            atomic_write_json(self.path, stamped.to_json())
            append_jsonl(self._history, stamped.to_json())
            return stamped

    def rebase(
        self,
        record: ReferenceRecord,
        *,
        migration_decision: dict,
        now=time.time,
    ) -> ReferenceRecord:
        """评测迁移后的新 baseline:唯一允许换 namespace 的写入路径。

        只应经由 migration.establish_baseline() 调用——那里强制校验
        approve-protocol-change 决定与迁移面板报告;直接调用本方法而
        绕过审批,历史记录里会留下没有 migration 依据的 rebase 行,
        audit 应视为违规。普通晋升走 promote()。
        """
        if not isinstance(migration_decision, dict) or not migration_decision:
            raise ValueError("rebase requires the approving decision payload")
        with self._lock:
            current = self.current()
            current_id = current.candidate_id if current else None
            stamped = ReferenceRecord(
                **{
                    **asdict(record),
                    "namespace": record.namespace,
                    "reasons": record.reasons,
                    "promoted_at": record.promoted_at or now(),
                    "previous_candidate_id": current_id,
                }
            )
            atomic_write_json(self.path, stamped.to_json())
            append_jsonl(
                self._history,
                {**stamped.to_json(), "migration": migration_decision},
            )
            return stamped

