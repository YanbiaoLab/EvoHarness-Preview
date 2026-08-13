"""Durable, append-only transcript artifacts for agent proposals."""

from __future__ import annotations

import difflib
import json
import os
import re
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

from ..checkpoint import atomic_write_json
from ..workspace import GitWorkspace, Workspace
from .contracts import (
    AgentEvent,
    PreflightTraceRecord,
    ProposalTraceSummary,
)


TRANSCRIPT_SCHEMA_VERSION = 1
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=str(path.parent),
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def render_workspace_patch(parent: Workspace, child: Workspace) -> str:
    """Render the one-step child delta used to reconstruct a proposal."""

    if isinstance(parent, GitWorkspace) and isinstance(child, GitWorkspace):
        parent_patches = parent.patches
        child_patches = child.patches
        if (
            child.base_files == parent.base_files
            and child.main_file == parent.main_file
            and child_patches[:-1] == parent_patches
            and len(child_patches) == len(parent_patches) + 1
        ):
            return child_patches[-1]

    parent_texts = parent.texts()
    child_texts = child.texts()
    chunks: list[str] = []

    for relative_path in sorted(set(parent_texts) | set(child_texts)):
        before = parent_texts.get(relative_path)
        after = child_texts.get(relative_path)
        if before == after:
            continue

        diff_lines = difflib.unified_diff(
                [] if before is None else before.splitlines(keepends=True),
                [] if after is None else after.splitlines(keepends=True),
                fromfile=(
                    "/dev/null"
                    if before is None
                    else f"a/{relative_path}"
                ),
                tofile=(
                    "/dev/null"
                    if after is None
                    else f"b/{relative_path}"
                ),
            )
        for line in diff_lines:
            chunks.append(line)
            if not line.endswith("\n"):
                chunks.append("\n\\ No newline at end of file\n")

    return "".join(chunks)


class JsonlEventSink:
    """Serialize one proposal's events and artifacts without overwriting."""

    def __init__(
        self,
        directory: Path,
        *,
        proposal_id: str,
        parent_id: str,
        operator: str,
    ):
        for name, value in (
            ("proposal_id", proposal_id),
            ("parent_id", parent_id),
            ("operator", operator),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if _SAFE_ID.fullmatch(proposal_id) is None:
            raise ValueError("proposal_id contains unsafe characters")

        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.proposal_id = proposal_id
        self.parent_id = parent_id
        self.operator = operator
        self.trace_path = str(self.directory)

        self._validate_existing_history()

        self._lock = threading.Lock()
        self._closed = False
        self._finalized = False
        self._events = self._open_append(self.directory / "events.jsonl")
        try:
            self._preflight = self._open_append(
                self.directory / "preflight.jsonl"
            )
        except BaseException:
            self._events.close()
            raise

    @staticmethod
    def _open_append(path: Path) -> TextIO:
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        return os.fdopen(
            descriptor,
            "a",
            encoding="utf-8",
            buffering=1,
        )

    def emit(self, event: AgentEvent) -> None:
        if not isinstance(event, AgentEvent):
            raise TypeError("event must be AgentEvent")

        payload = {
            "schema_version": TRANSCRIPT_SCHEMA_VERSION,
            "proposal_id": self.proposal_id,
            "parent_id": self.parent_id,
            "operator": self.operator,
            "round": event.round_index,
            "session_id": event.session_id,
            "event": {
                "sequence": event.sequence,
                "kind": event.kind.value,
                "turn": event.turn,
                "elapsed_s": event.elapsed_s,
                "call_id": event.call_id,
                "tool_name": event.tool_name,
                "content": event.content,
                "data": dict(event.data),
            },
        }
        self._append(self._events, payload)

    def record_preflight(self, record: PreflightTraceRecord) -> None:
        if not isinstance(record, PreflightTraceRecord):
            raise TypeError("record must be PreflightTraceRecord")

        report = record.report
        payload = {
            "schema_version": TRANSCRIPT_SCHEMA_VERSION,
            "proposal_id": self.proposal_id,
            "parent_id": self.parent_id,
            "operator": self.operator,
            "round": record.round_index,
            "session_id": record.session_id,
            "report": {
                "ok": report.ok,
                "repairable": report.repairable,
                "failed_stage": report.failed_stage,
                "elapsed_s": report.elapsed_s,
                "results": [
                    {
                        "stage": result.stage,
                        "elapsed_s": result.elapsed_s,
                        "issues": [
                            asdict(issue) for issue in result.issues
                        ],
                    }
                    for result in report.results
                ],
            },
        }
        self._append(self._preflight, payload)

    def finalize(
        self,
        summary: ProposalTraceSummary,
        final_patch: str | None,
    ) -> None:
        if not isinstance(summary, ProposalTraceSummary):
            raise TypeError("summary must be ProposalTraceSummary")
        if (
            summary.proposal_id != self.proposal_id
            or summary.parent_id != self.parent_id
            or summary.operator != self.operator
        ):
            raise ValueError("summary identity does not match transcript")
        if final_patch is not None and not isinstance(final_patch, str):
            raise TypeError("final_patch must be text when present")

        payload = {
            "schema_version": TRANSCRIPT_SCHEMA_VERSION,
            **asdict(summary),
            "final_patch_present": bool(final_patch),
        }
        patch_path = self.directory / "final.patch"

        with self._lock:
            self._ensure_writable()
            self._events.flush()
            self._preflight.flush()
            if final_patch:
                _atomic_write_text(patch_path, final_patch)
            elif patch_path.exists():
                patch_path.unlink()
            atomic_write_json(self.directory / "summary.json", payload)
            self._finalized = True

    def flush(self) -> None:
        with self._lock:
            self._ensure_open()
            self._events.flush()
            self._preflight.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            first_error: Exception | None = None
            for handle in (self._events, self._preflight):
                try:
                    handle.close()
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
            if first_error is not None:
                raise first_error

    def _append(self, handle: TextIO, payload: dict[str, object]) -> None:
        line = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._lock:
            self._ensure_writable()
            handle.write(line + "\n")
            handle.flush()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("transcript sink is closed")

    def _ensure_writable(self) -> None:
        self._ensure_open()
        if self._finalized:
            raise RuntimeError("transcript sink is already finalized")

    def _validate_existing_history(self) -> None:
        for path in (
            self.directory / "events.jsonl",
            self.directory / "preflight.jsonl",
        ):
            if not path.exists():
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError(
                    f"existing transcript is not a regular file: {path.name}"
                )
            try:
                with path.open(encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        self._validate_identity(
                            json.loads(line),
                            path=path,
                            line_number=line_number,
                        )
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"existing transcript is unreadable: {path.name}"
                ) from exc

        summary_path = self.directory / "summary.json"
        if not summary_path.exists():
            return
        if summary_path.is_symlink() or not summary_path.is_file():
            raise ValueError("existing summary is not a regular file")
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self._validate_identity(payload, path=summary_path)
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("existing transcript summary is unreadable") from exc
        raise ValueError("proposal transcript is already finalized")

    def _validate_identity(
        self,
        payload: dict[str, object],
        *,
        path: Path,
        line_number: int | None = None,
    ) -> None:
        if payload["schema_version"] != TRANSCRIPT_SCHEMA_VERSION:
            location = path.name
            if line_number is not None:
                location += f":{line_number}"
            raise ValueError(
                f"unsupported transcript schema at {location}"
            )
        actual = (
            payload["proposal_id"],
            payload["parent_id"],
            payload["operator"],
        )
        expected = (self.proposal_id, self.parent_id, self.operator)
        if actual != expected:
            raise ValueError(
                "existing transcript identity does not match proposal"
            )


class JsonlEventSinkFactory:
    """Create proposal transcript directories below one run directory."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir).resolve()
        self.sessions_dir = self.run_dir / "agent_sessions"

    def open(
        self,
        *,
        proposal_id: str,
        parent_id: str,
        operator: str,
    ) -> JsonlEventSink:
        if (
            not isinstance(proposal_id, str)
            or _SAFE_ID.fullmatch(proposal_id) is None
        ):
            raise ValueError("proposal_id contains unsafe characters")

        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        directory = self.sessions_dir / proposal_id
        if directory.is_symlink():
            raise ValueError("proposal transcript directory cannot be symlink")
        directory.mkdir(exist_ok=True)

        resolved = directory.resolve()
        if resolved.parent != self.sessions_dir.resolve():
            raise ValueError("proposal transcript escaped sessions directory")

        return JsonlEventSink(
            resolved,
            proposal_id=proposal_id,
            parent_id=parent_id,
            operator=operator,
        )
