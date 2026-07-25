"""Strict, workspace-confined file tools for agent sessions."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Iterable

from ...llm import LLMToolCall, LLMToolDefinition, LLMToolResult
from .run import Runner, RunnerResult
from .base import (
    RETENTION_EPHEMERAL,
    AgentToolContext,
    AgentToolError,
    make_tool_result,
)


_VCS_DIRECTORIES = frozenset({".git", ".hg", ".svn", ".bzr", ".jj"})


def resolve_workspace_path(
    workdir: Path,
    raw_path: str,
    *,
    allow_root: bool = False,
    allow_leaf_symlink: bool = False,
) -> Path:
    """Resolve a model-supplied path without leaving the proposal workspace."""

    root = Path(workdir).resolve(strict=True)
    if not root.is_dir():
        raise AgentToolError("invalid-path", "workspace root is not a directory")
    if not isinstance(raw_path, str):
        raise AgentToolError("invalid-path", "path must be a string")
    if "\x00" in raw_path:
        raise AgentToolError("invalid-path", "path cannot contain NUL bytes")
    if raw_path in {"", "."}:
        if allow_root:
            return root
        raise AgentToolError("invalid-path", "path must name a workspace entry")

    relative = Path(raw_path)
    if relative.is_absolute():
        raise AgentToolError("path-escape", "absolute paths are not allowed")
    if ".." in relative.parts:
        raise AgentToolError("path-escape", "parent traversal is not allowed")
    if any(part in _VCS_DIRECTORIES for part in relative.parts):
        raise AgentToolError("invalid-path", "version-control metadata is protected")

    current = root
    meaningful_parts = tuple(part for part in relative.parts if part not in {"", "."})
    for index, part in enumerate(meaningful_parts):
        current = current / part
        if current.is_symlink():
            is_leaf = index == len(meaningful_parts) - 1
            if is_leaf and allow_leaf_symlink:
                return current
            raise AgentToolError(
                "symlink",
                f"symlink paths are not allowed: {relative.as_posix()}",
            )

    resolved = current.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise AgentToolError("path-escape", "path leaves the workspace")
    if resolved == root and not allow_root:
        raise AgentToolError("invalid-path", "path must name a workspace entry")
    return resolved


def _validate_exact_arguments(
    arguments: dict[str, object],
    expected: set[str],
) -> None:
    actual = set(arguments)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"unexpected={extra}")
        raise AgentToolError(
            "invalid-arguments",
            "invalid arguments: " + ", ".join(details),
        )


def _required_string(
    arguments: dict[str, object],
    name: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = arguments[name]
    if not isinstance(value, str) or (not allow_empty and not value):
        requirement = "a string" if allow_empty else "a non-empty string"
        raise AgentToolError(
            "invalid-arguments",
            f"{name} must be {requirement}",
        )
    return value


def _required_bool(arguments: dict[str, object], name: str) -> bool:
    value = arguments[name]
    if not isinstance(value, bool):
        raise AgentToolError("invalid-arguments", f"{name} must be a boolean")
    return value


def _required_int(
    arguments: dict[str, object],
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    value = arguments[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise AgentToolError("invalid-arguments", f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        upper = f" and at most {maximum}" if maximum is not None else ""
        raise AgentToolError(
            "invalid-arguments",
            f"{name} must be at least {minimum}{upper}",
        )
    return value


def _display_path(path: Path, workdir: Path) -> str:
    relative = path.relative_to(workdir)
    return relative.as_posix() if relative.parts else "."


def _require_file(path: Path, raw_path: str) -> None:
    if not path.exists():
        raise AgentToolError("not-found", f"path does not exist: {raw_path}")
    if not path.is_file():
        raise AgentToolError("not-file", f"not a file: {raw_path}")


def _validate_glob_pattern(pattern: str) -> str:
    if not pattern or "\x00" in pattern:
        raise AgentToolError(
            "invalid-pattern",
            "glob pattern must be non-empty and contain no NUL bytes",
        )
    normalized = pattern.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts:
        raise AgentToolError(
            "invalid-pattern",
            "glob pattern must stay within its search path",
        )
    if any(part in _VCS_DIRECTORIES for part in pure.parts):
        raise AgentToolError(
            "invalid-pattern",
            "glob pattern cannot access version-control metadata",
        )
    return normalized


def _matches_glob(relative_path: str, pattern: str) -> bool:
    path = PurePosixPath(relative_path)
    if path.match(pattern):
        return True
    return pattern.startswith("**/") and path.match(pattern[3:])


def _iter_workspace_files(base: Path, workdir: Path) -> Iterable[Path]:
    """Yield files deterministically without following directory symlinks."""

    if base.is_file():
        yield base
        return
    if not base.is_dir():
        return

    for directory, dirnames, filenames in os.walk(base, followlinks=False):
        directory_path = Path(directory)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in _VCS_DIRECTORIES
            and not (directory_path / name).is_symlink()
        )
        for name in sorted(filenames):
            path = directory_path / name
            relative = path.relative_to(workdir)
            if (
                any(part in _VCS_DIRECTORIES for part in relative.parts)
                or path.is_symlink()
                or not path.is_file()
            ):
                continue
            yield path


def _read_utf8(path: Path, raw_path: str, max_file_bytes: int) -> str:
    if path.stat().st_size > max_file_bytes:
        raise AgentToolError(
            "output-too-large",
            f"file exceeds {max_file_bytes} bytes: {raw_path}",
        )
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise AgentToolError(
            "not-text",
            f"file is not UTF-8 text: {raw_path}",
        ) from exc
    if "\x00" in content:
        raise AgentToolError("not-text", f"file is not text: {raw_path}")
    return content


def _atomic_write(path: Path, content: str, max_write_bytes: int) -> int:
    encoded_size = len(content.encode("utf-8"))
    if encoded_size > max_write_bytes:
        raise AgentToolError(
            "output-too-large",
            f"file exceeds {max_write_bytes} bytes",
        )

    mode = (path.stat().st_mode & 0o777) if path.exists() else 0o644
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, mode)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return encoded_size


class WorkspaceReadTool:
    """Read a bounded line range from one UTF-8 workspace file."""

    # File dumps go stale as soon as the agent acts on them.
    retention = RETENTION_EPHEMERAL

    definition = LLMToolDefinition(
        name="workspace_read",
        description=(
            "Read a UTF-8 file in the proposal workspace. Paths are workspace-"
            "relative. offset is 1-based; limit is the maximum number of lines."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer", "minimum": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 2000},
            },
            "required": ["path", "offset", "limit"],
            "additionalProperties": False,
        },
    )

    def __init__(
        self,
        *,
        max_file_bytes: int = 2 * 1024 * 1024,
        max_output_chars: int = 100_000,
        max_lines: int = 2000,
    ):
        if min(max_file_bytes, max_output_chars, max_lines) < 1:
            raise ValueError("workspace read limits must be positive")
        if max_lines > 2000:
            raise ValueError("max_lines cannot exceed the tool schema maximum")
        self.max_file_bytes = max_file_bytes
        self.max_output_chars = max_output_chars
        self.max_lines = max_lines

    def is_concurrency_safe(self, call: LLMToolCall, ctx: AgentToolContext) -> bool:
        return True

    def invoke(self, call: LLMToolCall, ctx: AgentToolContext) -> LLMToolResult:
        _validate_exact_arguments(call.arguments, {"path", "offset", "limit"})
        raw_path = _required_string(call.arguments, "path")
        offset = _required_int(call.arguments, "offset", minimum=1)
        limit = _required_int(
            call.arguments,
            "limit",
            minimum=1,
            maximum=self.max_lines,
        )
        path = resolve_workspace_path(ctx.workdir, raw_path)
        _require_file(path, raw_path)
        display = _display_path(path, ctx.workdir)
        # Re-reading an unchanged range dumps the same file into the
        # conversation twice, and every copy is resent on every later turn.
        # Point the model at the earlier result instead; any mtime change
        # defeats the check and the fresh content is sent in full.
        fingerprint = (path.stat().st_mtime_ns, offset, limit)
        if ctx.read_state.get(display) == fingerprint:
            return make_tool_result(
                call.call_id,
                {
                    "ok": True,
                    "path": display,
                    "unchanged": True,
                    "message": (
                        "File unchanged since your earlier workspace_read of "
                        "this same range. Refer to that result instead of "
                        "re-reading; it is still current."
                    ),
                },
            )
        content = _read_utf8(path, raw_path, self.max_file_bytes)
        lines = content.splitlines()
        start_index = offset - 1
        selected = lines[start_index : start_index + limit]
        numbered = "\n".join(
            f"{line_number}\t{line}"
            for line_number, line in enumerate(selected, start=offset)
        )
        if len(numbered) > self.max_output_chars:
            raise AgentToolError(
                "output-too-large",
                "selected range is too large; retry with a smaller limit",
            )

        end_line = min(start_index + len(selected), len(lines))
        has_more = end_line < len(lines)
        ctx.read_state[display] = fingerprint
        return make_tool_result(
            call.call_id,
            {
                "ok": True,
                "path": display,
                "content": numbered,
                "start_line": offset,
                "end_line": end_line,
                "total_lines": len(lines),
                "has_more": has_more,
                "next_offset": end_line + 1 if has_more else None,
            },
        )


class WorkspaceGlobTool:
    """Find workspace files by glob pattern with deterministic pagination."""

    retention = RETENTION_EPHEMERAL

    definition = LLMToolDefinition(
        name="workspace_glob",
        description=(
            "Find files by glob pattern inside the proposal workspace. Use "
            "pattern='**/*' to enumerate all files below path."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 500},
            },
            "required": ["pattern", "path", "offset", "limit"],
            "additionalProperties": False,
        },
    )

    def __init__(self, *, max_results: int = 500):
        if not 1 <= max_results <= 500:
            raise ValueError("max_results must be between 1 and 500")
        self.max_results = max_results

    def is_concurrency_safe(self, call: LLMToolCall, ctx: AgentToolContext) -> bool:
        return True

    def invoke(self, call: LLMToolCall, ctx: AgentToolContext) -> LLMToolResult:
        _validate_exact_arguments(
            call.arguments,
            {"pattern", "path", "offset", "limit"},
        )
        pattern = _validate_glob_pattern(
            _required_string(call.arguments, "pattern")
        )
        raw_path = _required_string(call.arguments, "path")
        offset = _required_int(call.arguments, "offset", minimum=0)
        limit = _required_int(
            call.arguments,
            "limit",
            minimum=1,
            maximum=self.max_results,
        )
        base = resolve_workspace_path(ctx.workdir, raw_path, allow_root=True)
        if not base.exists():
            raise AgentToolError("not-found", f"path does not exist: {raw_path}")
        if not base.is_dir():
            raise AgentToolError("not-directory", f"not a directory: {raw_path}")

        matches = sorted(
            _display_path(path, ctx.workdir)
            for path in _iter_workspace_files(base, ctx.workdir)
            if _matches_glob(path.relative_to(base).as_posix(), pattern)
        )
        selected = matches[offset : offset + limit]
        has_more = offset + len(selected) < len(matches)
        return make_tool_result(
            call.call_id,
            {
                "ok": True,
                "pattern": pattern,
                "path": _display_path(base, ctx.workdir),
                "files": selected,
                "num_files": len(selected),
                "total_files": len(matches),
                "offset": offset,
                "has_more": has_more,
                "next_offset": offset + len(selected) if has_more else None,
            },
        )


class WorkspaceGrepTool:
    """Search workspace text with an injected, timeout-bounded ripgrep runner."""

    retention = RETENTION_EPHEMERAL

    definition = LLMToolDefinition(
        name="workspace_grep",
        description=(
            "Regex-search file contents inside the proposal workspace. glob='' "
            "searches every file. output_mode is content, files_with_matches, "
            "or count."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "glob": {"type": "string"},
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                },
                "case_insensitive": {"type": "boolean"},
                "head_limit": {"type": "integer", "minimum": 1, "maximum": 500},
                "offset": {"type": "integer", "minimum": 0},
            },
            "required": [
                "pattern",
                "path",
                "glob",
                "output_mode",
                "case_insensitive",
                "head_limit",
                "offset",
            ],
            "additionalProperties": False,
        },
    )

    def __init__(
        self,
        runner: Runner,
        *,
        max_results: int = 500,
        max_line_chars: int = 500,
        max_backend_output_bytes: int = 20 * 1024 * 1024,
        runner_timeout_cap_s: float = 20.0,
    ):
        if not isinstance(runner, Runner):
            raise TypeError("runner must implement the Runner protocol")
        if min(
            max_results,
            max_line_chars,
            max_backend_output_bytes,
            runner_timeout_cap_s,
        ) < 1:
            raise ValueError("workspace grep limits must be positive")
        if max_results > 500:
            raise ValueError("max_results cannot exceed the tool schema maximum")
        self.runner = runner
        self.max_results = max_results
        self.max_line_chars = max_line_chars
        self.max_backend_output_bytes = max_backend_output_bytes
        self.runner_timeout_cap_s = float(runner_timeout_cap_s)

    def is_concurrency_safe(self, call: LLMToolCall, ctx: AgentToolContext) -> bool:
        return True

    def invoke(self, call: LLMToolCall, ctx: AgentToolContext) -> LLMToolResult:
        expected = {
            "pattern",
            "path",
            "glob",
            "output_mode",
            "case_insensitive",
            "head_limit",
            "offset",
        }
        _validate_exact_arguments(call.arguments, expected)
        pattern = _required_string(call.arguments, "pattern")
        raw_path = _required_string(call.arguments, "path")
        glob_pattern = _required_string(
            call.arguments,
            "glob",
            allow_empty=True,
        )
        if glob_pattern:
            glob_pattern = _validate_glob_pattern(glob_pattern)
        output_mode = _required_string(call.arguments, "output_mode")
        if output_mode not in {"content", "files_with_matches", "count"}:
            raise AgentToolError(
                "invalid-arguments",
                "output_mode must be content, files_with_matches, or count",
            )
        case_insensitive = _required_bool(call.arguments, "case_insensitive")
        head_limit = _required_int(
            call.arguments,
            "head_limit",
            minimum=1,
            maximum=self.max_results,
        )
        offset = _required_int(call.arguments, "offset", minimum=0)

        base = resolve_workspace_path(ctx.workdir, raw_path, allow_root=True)
        if not base.exists():
            raise AgentToolError("not-found", f"path does not exist: {raw_path}")
        if not base.is_file() and not base.is_dir():
            raise AgentToolError(
                "not-file",
                f"not a file or directory: {raw_path}",
            )

        argv = [
            "rg",
            "--json",
            "--hidden",
            "--sort",
            "path",
            "--max-columns",
            str(self.max_line_chars),
            "--max-filesize",
            "2M",
        ]
        for vcs_directory in sorted(_VCS_DIRECTORIES):
            argv.extend(("--glob", f"!{vcs_directory}/**"))
        if case_insensitive:
            argv.append("--ignore-case")
        if glob_pattern:
            argv.extend(("--glob", glob_pattern))
        argv.extend(("--regexp", pattern, _display_path(base, ctx.workdir)))

        timeout_s = min(ctx.remaining_timeout_s, self.runner_timeout_cap_s)
        try:
            backend_result = self.runner.run(
                argv,
                workdir=ctx.workdir,
                timeout_s=timeout_s,
                env={"PATH": os.environ.get("PATH", os.defpath)},
            )
        except FileNotFoundError as exc:
            raise AgentToolError(
                "search-backend-unavailable",
                "ripgrep executable was not found",
            ) from exc
        except OSError as exc:
            raise AgentToolError(
                "search-failed",
                f"failed to start ripgrep: {exc}",
            ) from exc

        self._validate_backend_result(backend_result)
        if backend_result.timed_out:
            raise AgentToolError("search-timeout", "ripgrep search timed out")
        if backend_result.return_code not in {0, 1}:
            message = backend_result.stderr.strip() or "ripgrep search failed"
            code = (
                "invalid-pattern"
                if "regex parse error" in message.lower()
                else "search-failed"
            )
            raise AgentToolError(code, message[:2000])
        if len(backend_result.stdout.encode("utf-8")) > self.max_backend_output_bytes:
            raise AgentToolError(
                "output-too-large",
                "ripgrep output exceeded the search backend limit; narrow the query",
            )

        selected_results: list[object] = []
        content_result_count = 0
        total_matches = 0
        counts_by_path: dict[str, int] = {}

        for raw_line in backend_result.stdout.splitlines():
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise AgentToolError(
                    "search-failed",
                    "ripgrep returned malformed JSON output",
                ) from exc
            if event.get("type") != "match":
                continue

            data = event.get("data")
            if not isinstance(data, dict):
                continue
            path_data = data.get("path")
            lines_data = data.get("lines")
            submatches = data.get("submatches")
            if not isinstance(path_data, dict) or not isinstance(lines_data, dict):
                continue
            result_path = path_data.get("text")
            line_text = lines_data.get("text")
            line_number = data.get("line_number")
            if (
                not isinstance(result_path, str)
                or not isinstance(line_text, str)
                or not isinstance(line_number, int)
                or not isinstance(submatches, list)
            ):
                continue
            result_path = Path(result_path).as_posix()

            match_count = len(submatches)
            total_matches += match_count
            counts_by_path[result_path] = (
                counts_by_path.get(result_path, 0) + match_count
            )
            if output_mode == "content":
                if offset <= content_result_count < offset + head_limit:
                    normalized_line = line_text.rstrip("\r\n")
                    selected_results.append(
                        {
                            "path": result_path,
                            "line": line_number,
                            "text": normalized_line[: self.max_line_chars],
                            "line_truncated": (
                                len(normalized_line) > self.max_line_chars
                            ),
                        }
                    )
                content_result_count += 1

        if output_mode == "content":
            total_results = content_result_count
        else:
            paths = sorted(counts_by_path)
            if output_mode == "count":
                all_results: list[object] = [
                    {"path": path, "count": counts_by_path[path]}
                    for path in paths
                ]
            else:
                all_results = paths
            total_results = len(all_results)
            selected_results = all_results[offset : offset + head_limit]

        has_more = total_results > offset + len(selected_results)
        return make_tool_result(
            call.call_id,
            {
                "ok": True,
                "mode": output_mode,
                "pattern": pattern,
                "path": _display_path(base, ctx.workdir),
                "results": selected_results,
                "num_results": len(selected_results),
                "total_results": total_results,
                "total_matches": total_matches,
                "offset": offset,
                "has_more": has_more,
                "next_offset": (
                    offset + len(selected_results) if has_more else None
                ),
                "backend": "ripgrep",
                "elapsed_s": backend_result.elapsed_s,
            },
        )

    @staticmethod
    def _validate_backend_result(result: RunnerResult) -> None:
        if not isinstance(result, RunnerResult):
            raise TypeError("runner returned an invalid result")


class WorkspaceWriteTool:
    """Atomically create or overwrite one workspace file."""

    definition = LLMToolDefinition(
        name="workspace_write",
        description="Create or overwrite a UTF-8 file in the proposal workspace.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    )

    def __init__(self, *, max_write_bytes: int = 2 * 1024 * 1024):
        if max_write_bytes < 1:
            raise ValueError("max_write_bytes must be positive")
        self.max_write_bytes = max_write_bytes

    def is_concurrency_safe(self, call: LLMToolCall, ctx: AgentToolContext) -> bool:
        return False

    def invoke(self, call: LLMToolCall, ctx: AgentToolContext) -> LLMToolResult:
        _validate_exact_arguments(call.arguments, {"path", "content"})
        raw_path = _required_string(call.arguments, "path")
        content = _required_string(call.arguments, "content", allow_empty=True)
        path = resolve_workspace_path(ctx.workdir, raw_path)
        if path.exists() and not path.is_file():
            raise AgentToolError("not-file", f"not a file: {raw_path}")
        bytes_written = _atomic_write(path, content, self.max_write_bytes)
        return make_tool_result(
            call.call_id,
            {
                "ok": True,
                "path": _display_path(path, ctx.workdir),
                "bytes_written": bytes_written,
            },
        )


class WorkspaceEditTool:
    """Replace exactly one text occurrence in a workspace file."""

    definition = LLMToolDefinition(
        name="workspace_edit",
        description=(
            "Replace exactly one occurrence of old_text in a UTF-8 workspace file."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
    )

    def __init__(self, *, max_write_bytes: int = 2 * 1024 * 1024):
        if max_write_bytes < 1:
            raise ValueError("max_write_bytes must be positive")
        self.max_write_bytes = max_write_bytes

    def is_concurrency_safe(self, call: LLMToolCall, ctx: AgentToolContext) -> bool:
        return False

    def invoke(self, call: LLMToolCall, ctx: AgentToolContext) -> LLMToolResult:
        _validate_exact_arguments(
            call.arguments,
            {"path", "old_text", "new_text"},
        )
        raw_path = _required_string(call.arguments, "path")
        old_text = _required_string(call.arguments, "old_text")
        new_text = _required_string(call.arguments, "new_text", allow_empty=True)
        path = resolve_workspace_path(ctx.workdir, raw_path)
        _require_file(path, raw_path)
        content = _read_utf8(path, raw_path, self.max_write_bytes)
        occurrences = content.count(old_text)
        if occurrences == 0:
            raise AgentToolError("replace-not-found", "old_text was not found")
        if occurrences != 1:
            raise AgentToolError(
                "replace-not-unique",
                f"old_text matched {occurrences} times",
            )
        updated = content.replace(old_text, new_text, 1)
        bytes_written = _atomic_write(path, updated, self.max_write_bytes)
        return make_tool_result(
            call.call_id,
            {
                "ok": True,
                "path": _display_path(path, ctx.workdir),
                "replacements": 1,
                "bytes_written": bytes_written,
            },
        )


class WorkspaceDeleteTool:
    """Delete one workspace file or leaf symlink without following it."""

    definition = LLMToolDefinition(
        name="workspace_delete",
        description="Delete one file or leaf symlink from the proposal workspace.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    )

    def is_concurrency_safe(self, call: LLMToolCall, ctx: AgentToolContext) -> bool:
        return False

    def invoke(self, call: LLMToolCall, ctx: AgentToolContext) -> LLMToolResult:
        _validate_exact_arguments(call.arguments, {"path"})
        raw_path = _required_string(call.arguments, "path")
        path = resolve_workspace_path(
            ctx.workdir,
            raw_path,
            allow_leaf_symlink=True,
        )
        if not path.exists() and not path.is_symlink():
            raise AgentToolError("not-found", f"path does not exist: {raw_path}")
        if not path.is_symlink() and not path.is_file():
            raise AgentToolError("not-file", f"delete only accepts files: {raw_path}")
        path.unlink()
        return make_tool_result(
            call.call_id,
            {"ok": True, "path": raw_path},
        )
