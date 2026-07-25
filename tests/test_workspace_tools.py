"""Strict workspace tool safety, paging, and edit semantics."""

import json

import pytest

from evoharness.evocore import (
    Candidate,
    LLMToolCall,
    PreflightPipeline,
    ProposalPreflight,
)
from evoharness.evocore.agent import (
    AgentToolContext,
    AgentToolError,
    AgentToolRegistry,
    WorkspaceDeleteTool,
    WorkspaceEditTool,
    WorkspaceGlobTool,
    WorkspaceGrepTool,
    WorkspaceReadTool,
    WorkspaceWriteTool,
    resolve_workspace_path,
)
from evoharness.evocore.workspace import GitWorkspace
from evoharness.evoguard import Sandbox


def make_context(tmp_path):
    parent = Candidate(
        id="parent",
        code=GitWorkspace(
            base_files={
                "main.py": "value = 1\nneedle = 'first'\n",
                "pkg/helper.py": "value = 1\nneedle = 'second'\nvalue = 1\n",
                "pkg/notes.txt": "needle\nother\nneedle\n",
            },
            main_file="main.py",
        ).serialize(),
        generation=0,
        parent_id=None,
        island_idx=0,
        operator="seed",
        workspace_kind="git",
    )
    workdir = parent.workspace.materialize(tmp_path / "work")
    return AgentToolContext(
        workdir=workdir,
        parent=parent,
        operator="rewrite",
        preflight=ProposalPreflight(PreflightPipeline()),
        remaining_timeout_s=30,
    )


def invoke(tool, ctx, **arguments):
    call = LLMToolCall("call-1", tool.definition.name, arguments)
    registry = AgentToolRegistry([tool])
    result = registry.invoke(call, ctx)
    return result, json.loads(result.content), registry, call


def test_path_resolver_rejects_escape_git_and_symlink(tmp_path):
    ctx = make_context(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (ctx.workdir / "link").symlink_to(outside, target_is_directory=True)

    assert resolve_workspace_path(ctx.workdir, "main.py") == (
        ctx.workdir / "main.py"
    )

    for raw_path, code in [
        (str(outside / "secret"), "path-escape"),
        ("../secret", "path-escape"),
        (".git/config", "invalid-path"),
        ("link/secret", "symlink"),
    ]:
        with pytest.raises(AgentToolError) as error:
            resolve_workspace_path(ctx.workdir, raw_path)
        assert error.value.code == code


def test_read_returns_numbered_line_page_and_next_offset(tmp_path):
    ctx = make_context(tmp_path)
    tool = WorkspaceReadTool()

    result, payload, registry, call = invoke(
        tool,
        ctx,
        path="pkg/notes.txt",
        offset=1,
        limit=2,
    )

    assert not result.is_error
    assert payload["content"] == "1\tneedle\n2\tother"
    assert payload["total_lines"] == 3
    assert payload["has_more"]
    assert payload["next_offset"] == 3
    assert registry.is_concurrency_safe(call, ctx)


def test_read_rejects_bad_range_binary_and_large_output(tmp_path):
    ctx = make_context(tmp_path)
    (ctx.workdir / "binary.dat").write_bytes(b"\xff\xfe")

    result, payload, _, _ = invoke(
        WorkspaceReadTool(),
        ctx,
        path="binary.dat",
        offset=1,
        limit=10,
    )
    assert result.is_error
    assert payload["error"]["code"] == "not-text"

    result, payload, _, _ = invoke(
        WorkspaceReadTool(max_output_chars=4),
        ctx,
        path="main.py",
        offset=1,
        limit=2,
    )
    assert result.is_error
    assert payload["error"]["code"] == "output-too-large"


def test_glob_is_deterministic_paged_and_does_not_follow_symlinks(tmp_path):
    ctx = make_context(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("secret = True\n")
    (ctx.workdir / "linked").symlink_to(outside, target_is_directory=True)

    tool = WorkspaceGlobTool(max_results=2)
    _, first, registry, call = invoke(
        tool,
        ctx,
        pattern="**/*",
        path=".",
        offset=0,
        limit=2,
    )
    _, second, _, _ = invoke(
        tool,
        ctx,
        pattern="**/*",
        path=".",
        offset=2,
        limit=2,
    )

    assert first["files"] == sorted(first["files"])
    assert first["files"] + second["files"] == [
        "main.py",
        "pkg/helper.py",
        "pkg/notes.txt",
    ]
    assert first["has_more"] and first["next_offset"] == 2
    assert not second["has_more"]
    assert all(".git" not in path for path in first["files"])
    assert all("linked" not in path for path in first["files"])
    assert registry.is_concurrency_safe(call, ctx)


def test_glob_filters_by_pattern_and_rejects_escape(tmp_path):
    ctx = make_context(tmp_path)

    _, payload, _, _ = invoke(
        WorkspaceGlobTool(),
        ctx,
        pattern="**/*.py",
        path=".",
        offset=0,
        limit=20,
    )
    assert payload["files"] == ["main.py", "pkg/helper.py"]

    result, payload, _, _ = invoke(
        WorkspaceGlobTool(),
        ctx,
        pattern="../**/*",
        path=".",
        offset=0,
        limit=20,
    )
    assert result.is_error
    assert payload["error"]["code"] == "invalid-pattern"


def test_grep_content_mode_is_regex_aware_and_paged(tmp_path):
    ctx = make_context(tmp_path)
    tool = WorkspaceGrepTool(Sandbox(), max_results=2)

    _, first, registry, call = invoke(
        tool,
        ctx,
        pattern="needle\\s*=|needle$",
        path=".",
        glob="",
        output_mode="content",
        case_insensitive=False,
        head_limit=2,
        offset=0,
    )
    _, second, _, _ = invoke(
        tool,
        ctx,
        pattern="needle\\s*=|needle$",
        path=".",
        glob="",
        output_mode="content",
        case_insensitive=False,
        head_limit=2,
        offset=2,
    )

    assert [item["path"] for item in first["results"]] == [
        "main.py",
        "pkg/helper.py",
    ]
    assert first["has_more"] and first["next_offset"] == 2
    assert len(second["results"]) == 2
    assert first["total_matches"] == 4
    assert registry.is_concurrency_safe(call, ctx)


def test_grep_files_count_glob_filter_and_invalid_regex(tmp_path):
    ctx = make_context(tmp_path)
    tool = WorkspaceGrepTool(Sandbox())
    common = {
        "pattern": "needle",
        "path": ".",
        "glob": "**/*.py",
        "case_insensitive": False,
        "head_limit": 20,
        "offset": 0,
    }

    _, files, _, _ = invoke(
        tool,
        ctx,
        output_mode="files_with_matches",
        **common,
    )
    _, counts, _, _ = invoke(tool, ctx, output_mode="count", **common)

    assert files["results"] == ["main.py", "pkg/helper.py"]
    assert counts["results"] == [
        {"count": 1, "path": "main.py"},
        {"count": 1, "path": "pkg/helper.py"},
    ]

    result, payload, _, _ = invoke(
        tool,
        ctx,
        pattern="[",
        path=".",
        glob="",
        output_mode="content",
        case_insensitive=False,
        head_limit=20,
        offset=0,
    )
    assert result.is_error
    assert payload["error"]["code"] == "invalid-pattern"


def test_write_edit_and_delete_are_separate_exclusive_tools(tmp_path):
    ctx = make_context(tmp_path)
    write = WorkspaceWriteTool()
    edit = WorkspaceEditTool()
    delete = WorkspaceDeleteTool()

    result, _, registry, call = invoke(
        write,
        ctx,
        path="pkg/new.py",
        content="x = 1\n",
    )
    assert not result.is_error
    assert not registry.is_concurrency_safe(call, ctx)
    assert (ctx.workdir / "pkg/new.py").read_text() == "x = 1\n"

    result, payload, registry, call = invoke(
        edit,
        ctx,
        path="pkg/new.py",
        old_text="x = 1",
        new_text="x = 2",
    )
    assert not result.is_error and payload["replacements"] == 1
    assert not registry.is_concurrency_safe(call, ctx)
    assert (ctx.workdir / "pkg/new.py").read_text() == "x = 2\n"

    result, _, registry, call = invoke(delete, ctx, path="pkg/new.py")
    assert not result.is_error
    assert not registry.is_concurrency_safe(call, ctx)
    assert not (ctx.workdir / "pkg/new.py").exists()


def test_edit_requires_one_match_and_delete_unlinks_leaf_symlink(tmp_path):
    ctx = make_context(tmp_path)

    for old_text, code in [
        ("missing", "replace-not-found"),
        ("value = 1", "replace-not-unique"),
    ]:
        result, payload, _, _ = invoke(
            WorkspaceEditTool(),
            ctx,
            path="pkg/helper.py",
            old_text=old_text,
            new_text="value = 2",
        )
        assert result.is_error
        assert payload["error"]["code"] == code

    target = tmp_path / "outside.py"
    target.write_text("secret = True\n")
    link = ctx.workdir / "linked.py"
    link.symlink_to(target)
    result, _, _, _ = invoke(WorkspaceDeleteTool(), ctx, path="linked.py")
    assert not result.is_error
    assert not link.exists()
    assert target.exists()


def test_every_workspace_tool_has_a_strict_schema():
    tools = (
        WorkspaceReadTool(),
        WorkspaceGlobTool(),
        WorkspaceGrepTool(Sandbox()),
        WorkspaceWriteTool(),
        WorkspaceEditTool(),
        WorkspaceDeleteTool(),
    )

    assert all(tool.definition.strict for tool in tools)
    assert all(
        set(tool.definition.input_schema["required"])
        == set(tool.definition.input_schema["properties"])
        for tool in tools
    )


def test_read_dedups_an_unchanged_range_and_reissues_after_a_change(tmp_path):
    """A re-read of an unchanged range is answered with a pointer, not a
    second copy: every copy in the conversation is resent on every later
    turn. Any mtime change must defeat the check."""
    ctx = make_context(tmp_path)
    tool = WorkspaceReadTool()

    _, first, _, _ = invoke(tool, ctx, path="main.py", offset=1, limit=100)
    assert "value = 1" in first["content"]
    assert not first.get("unchanged")

    _, second, _, _ = invoke(tool, ctx, path="main.py", offset=1, limit=100)
    assert second["unchanged"] is True
    assert "content" not in second          # no second copy of the file
    assert "still current" in second["message"]

    # a different range is a different fingerprint
    _, ranged, _, _ = invoke(tool, ctx, path="main.py", offset=2, limit=1)
    assert "content" in ranged and not ranged.get("unchanged")
    _, ranged_again, _, _ = invoke(tool, ctx, path="main.py", offset=2, limit=1)
    assert ranged_again["unchanged"] is True

    # editing the file must invalidate the dedup
    invoke(
        WorkspaceEditTool(),
        ctx,
        path="main.py",
        old_text="value = 1",
        new_text="value = 42",
    )
    _, after_edit, _, _ = invoke(tool, ctx, path="main.py", offset=2, limit=1)
    assert not after_edit.get("unchanged")   # mtime moved, dedup defeated
    _, full, _, _ = invoke(tool, ctx, path="main.py", offset=1, limit=100)
    assert "value = 42" in full["content"]


def test_read_dedup_is_per_session(tmp_path):
    """read_state lives on the session, so a fresh session never inherits
    another session's belief about what the model has already seen."""
    ctx = make_context(tmp_path)
    tool = WorkspaceReadTool()
    invoke(tool, ctx, path="main.py", offset=1, limit=100)

    other = make_context(tmp_path / "other")
    _, fresh, _, _ = invoke(tool, other, path="main.py", offset=1, limit=100)
    assert "content" in fresh and not fresh.get("unchanged")
