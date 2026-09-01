"""TaskSpec: frozen domain semantics, measurement and component identities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .component import ComponentSpec
from .fingerprint import canonical_json, canonical_object_json, spec_hash


@dataclass(frozen=True)
class CriterionSpec:
    name: str
    direction: str = "maximize"
    description: str = ""
    version: str = "v1"
    #: The fitness at which this criterion considers the problem solved, if it
    #: has such a point at all. Most do not: a criterion measuring a speedup or
    #: a byte count has no value that means "done", and naming one would end a
    #: run at the first good generation. A criterion that either accepts or
    #: does not — one problem, one judge — has exactly one, and declaring it
    #: here rather than at launch makes it part of what the task IS.
    solved_at: float | None = None

    def __post_init__(self) -> None:
        direction = self.direction.lower()
        if direction not in {"maximize", "minimize"}:
            raise ValueError(f"invalid criterion direction: {self.direction}")
        if not self.name.strip() or not self.version.strip():
            raise ValueError("criterion name and version must be non-empty")
        if self.solved_at is not None and direction == "minimize":
            # For a minimizing criterion the number a person would write here
            # is a criterion value, and the loop compares it against fitness,
            # which graders already orient so that higher is better. Accepting
            # it would stop the run on the opposite condition to the one the
            # task declared, and nothing downstream could tell.
            raise ValueError(
                "solved_at is a fitness threshold and only applies to a "
                "maximizing criterion; a minimizing one must convert in its "
                "grader"
            )
        object.__setattr__(self, "direction", direction)

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "description": self.description,
            "version": self.version,
            # Present only when declared, so adding this field left every
            # existing task's hash byte-identical. Not a trick to keep old
            # runs resumable: a criterion that names no solved point is the
            # same criterion it was before there was a way to name one.
            **({} if self.solved_at is None else {"solved_at": self.solved_at}),
        }

    @property
    def hash(self) -> str:
        return spec_hash({"kind": "criterion", **self.to_payload()})


@dataclass(frozen=True)
class MeasurementSpec:
    name: str
    version: str = "v1"
    universe_hash: str = "unknown"
    planned_units: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.version.strip():
            raise ValueError("measurement name and version must be non-empty")
        if (
            isinstance(self.planned_units, bool)
            or not isinstance(self.planned_units, int)
            or self.planned_units < 0
        ):
            raise ValueError("planned_units must be a non-negative integer")
        if not self.universe_hash.strip():
            raise ValueError("universe_hash must be non-empty")

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "universe_hash": self.universe_hash,
            "planned_units": self.planned_units,
            "notes": self.notes,
        }

    @property
    def hash(self) -> str:
        return spec_hash({"kind": "measurement", **self.to_payload()})


@dataclass(frozen=True)
class FeedbackSpec:
    version: str = "v1"
    exposes_stdout: bool = True
    exposes_stderr: bool = True
    exposes_structured_feedback: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("feedback version must be non-empty")
        for name in (
            "exposes_stdout",
            "exposes_stderr",
            "exposes_structured_feedback",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "exposes_stdout": self.exposes_stdout,
            "exposes_stderr": self.exposes_stderr,
            "exposes_structured_feedback": self.exposes_structured_feedback,
            "notes": self.notes,
        }

    @property
    def hash(self) -> str:
        return spec_hash({"kind": "feedback", **self.to_payload()})


@dataclass(frozen=True)
class WorkspaceSpec:
    kind: str
    blob: str
    main_file: str = "main.py"

    def __post_init__(self) -> None:
        if self.kind not in {"file", "git"}:
            raise ValueError(f"unsupported workspace kind: {self.kind}")
        if not isinstance(self.blob, str):
            raise TypeError("workspace blob must be a string")
        if not isinstance(self.main_file, str) or not self.main_file.strip():
            raise ValueError("workspace main_file must be non-empty")
        if self.kind == "git":
            try:
                payload = json.loads(self.blob)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "git workspace blob must be valid JSON"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError("git workspace blob must encode an object")
            object.__setattr__(self, "blob", canonical_json(payload))

    @classmethod
    def from_workspace(cls, workspace: object) -> "WorkspaceSpec":
        return cls(
            kind=str(getattr(workspace, "kind")),
            blob=str(getattr(workspace, "serialize")()),
            main_file=str(getattr(workspace, "main_file", None) or getattr(
                workspace, "filename", "main.py"
            )),
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "blob": self.blob,
            "main_file": self.main_file,
        }

    @property
    def hash(self) -> str:
        return spec_hash(self.to_payload())


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    version: str
    initial_workspace: WorkspaceSpec
    criterion: CriterionSpec
    measurement: MeasurementSpec
    feedback: FeedbackSpec
    grader: ComponentSpec
    domain_prompt: str = ""
    knowledge: tuple[str, ...] = ()
    preflight_validators: tuple[ComponentSpec, ...] = ()
    success_policy: ComponentSpec | None = None
    domain_tools: tuple[ComponentSpec, ...] = ()
    extra_seeds: tuple[WorkspaceSpec, ...] = ()
    metadata_json: str = "{}"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.task_id, str)
            or not isinstance(self.version, str)
            or not self.task_id.strip()
            or not self.version.strip()
        ):
            raise ValueError("task_id and version must be non-empty")
        self.grader.require_role("grader")
        for validator in self.preflight_validators:
            validator.require_role("preflight_validator")
        if self.success_policy is not None:
            self.success_policy.require_role("success_policy")
        for tool in self.domain_tools:
            tool.require_role("domain_tool")
        names = [item.name for item in self.preflight_validators]
        if len(names) != len(set(names)):
            raise ValueError("preflight validator names must be unique")
        if any(not item.strip() for item in self.knowledge):
            raise ValueError("knowledge entries must be non-empty")
        try:
            metadata = json.loads(self.metadata_json)
        except json.JSONDecodeError as exc:
            raise ValueError("metadata_json must be valid JSON") from exc
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json must encode a JSON object")
        object.__setattr__(self, "metadata_json", canonical_object_json(metadata))

    @property
    def task_sys_msg(self) -> str:
        return self.domain_prompt

    @property
    def research_brief(self) -> str:
        return "\n\n".join(self.knowledge)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task_id": self.task_id,
            "version": self.version,
            "initial_workspace": self.initial_workspace.to_payload(),
            "criterion": self.criterion.to_payload(),
            "measurement": self.measurement.to_payload(),
            "feedback": self.feedback.to_payload(),
            "grader": self.grader.to_payload(),
            "domain_prompt": self.domain_prompt,
            "knowledge": list(self.knowledge),
            "preflight_validators": [
                item.to_payload() for item in self.preflight_validators
            ],
            "success_policy": (
                self.success_policy.to_payload()
                if self.success_policy is not None else None
            ),
            "domain_tools": [item.to_payload() for item in self.domain_tools],
            "extra_seeds": [item.to_payload() for item in self.extra_seeds],
            "metadata": json.loads(self.metadata_json),
        }

    @property
    def hash(self) -> str:
        return spec_hash({"kind": "task", **self.to_payload()})
