"""Tests for the Workspace abstraction (WS-3 M1)."""

import subprocess
from pathlib import Path

import pytest

from evoharness.evocore.workspace import (
    FileWorkspace,
    GitWorkspace,
    Workspace,
    WorkspaceError,
    load_workspace,
)


def tree(root: Path) -> dict[str, str]:
    """Directory snapshot {relpath: text}, ignoring git bookkeeping."""
    return {
        str(p.relative_to(root)): p.read_text()
        for p in root.rglob("*")
        if p.is_file() and ".git" not in p.parts
    }


def make_patch(ws: GitWorkspace, workdir: Path, edits: dict[str, str]) -> str:
    """Materialize ws, apply edits {relpath: new_text}, return the diff.

    `git add -A` + `diff --cached` (not plain `diff`) so that newly created
    files appear in the patch too — plain diff ignores untracked files."""
    root = ws.materialize(workdir)
    for rel, text in edits.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    proc = subprocess.run(
        ["git", "-C", str(root), "diff", "--cached"],
        capture_output=True, text=True, check=True,
    )
    return proc.stdout


# -- 1. FileWorkspace roundtrip ------------------------------------------------

def test_file_workspace_roundtrip(tmp_path):
    ws = FileWorkspace("print('Hello, World!')")
    blob = ws.serialize()
    assert blob == "print('Hello, World!')"  # parity: serialized form IS the code

    back = load_workspace("file", blob)
    assert isinstance(back, FileWorkspace)
    assert back.main_text() == "print('Hello, World!')"

    root = back.materialize(tmp_path / "w")
    assert tree(root) == {"main.py": "print('Hello, World!')"}  # exactly one file


# -- 2. GitWorkspace roundtrip -------------------------------------------------

def test_git_workspace_roundtrip(tmp_path):
    parent = GitWorkspace(
        base_files={"main.py": "x = 1\n", "util/helper.py": "def f(): return 42\n"}
    )
    p1 = make_patch(parent, tmp_path / "w1", {"main.py": "x = 1\ny = 2\n"})
    ws1 = parent.child(p1)
    # second patch CREATES a file — exercises new-file diffs
    p2 = make_patch(ws1, tmp_path / "w2", {"config.json": '{"n": 2}\n'})
    ws2 = ws1.child(p2)

    back = GitWorkspace.deserialize(ws2.serialize())
    assert tree(back.materialize(tmp_path / "b")) == tree(ws2.materialize(tmp_path / "a"))

    via_factory = load_workspace("git", ws2.serialize())
    assert via_factory.main_text() == "x = 1\ny = 2\n"


# -- 3. lineage: parent + one diff = child (the M3 handshake) -------------------

def test_git_workspace_child_lineage(tmp_path):
    parent = GitWorkspace(
        base_files={"main.py": "x = 1\n", "util/helper.py": "def f(): return 42\n"}
    )
    # materialize leaves a real git repo, HEAD at the end of the patch chain
    work = parent.materialize(tmp_path / "parent")
    # play the M3 agent: edit a file in place
    (work / "main.py").write_text("x = 1\ny = x + 1\n")
    # collect the change relative to HEAD — how M3 hands a proposal back
    patch = subprocess.run(
        ["git", "-C", str(work), "diff"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "y = x + 1" in patch

    child = parent.child(patch)
    rebuilt = child.materialize(tmp_path / "child")
    assert (rebuilt / "main.py").read_text() == "x = 1\ny = x + 1\n"
    # untouched files survive unchanged
    assert (rebuilt / "util" / "helper.py").read_text() == "def f(): return 42\n"
    assert child.main_text() == "x = 1\ny = x + 1\n"


# -- 4. error paths: every failure is a classified WorkspaceError ----------------

def test_bad_patch_raises(tmp_path):
    ws = GitWorkspace(base_files={"main.py": "x = 1\n"}, patches=["not a diff"])
    with pytest.raises(WorkspaceError, match="apply"):
        ws.materialize(tmp_path / "w")


def test_missing_main_file_raises(tmp_path):
    ws = GitWorkspace(base_files={"other.py": "x = 1\n"}, main_file="main.py")
    with pytest.raises(WorkspaceError, match="missing"):
        ws.materialize(tmp_path / "w")


def test_unsafe_path_raises(tmp_path):
    ws = GitWorkspace(base_files={"../evil": "x"})
    with pytest.raises(WorkspaceError, match="unsafe"):
        ws.materialize(tmp_path / "w")


def test_unknown_kind_raises():
    with pytest.raises(WorkspaceError, match="unknown"):
        load_workspace("blockchain", "x")


# -- 5. protocol conformance ----------------------------------------------------

def test_workspaces_satisfy_protocol():
    assert isinstance(FileWorkspace("x"), Workspace)
    assert isinstance(GitWorkspace(base_files={"main.py": "x"}), Workspace)


# -- M2.5 lens primitives --------------------------------------------------------

def test_with_files_multi_edit(tmp_path):
    ws = GitWorkspace(base_files={"main.py": "x = 1\n", "util.py": "y = 2\n"})
    child = ws.with_files({"main.py": "x = 9\n", "new.py": "z = 3\n"})
    assert len(child.patches) == 1  # N edits, ONE atomic lineage step
    rebuilt = child.materialize(tmp_path / "c")
    assert rebuilt.joinpath("main.py").read_text() == "x = 9\n"
    assert rebuilt.joinpath("new.py").read_text() == "z = 3\n"  # created
    assert rebuilt.joinpath("util.py").read_text() == "y = 2\n"  # untouched


def test_with_files_noop_returns_equivalent(tmp_path):
    ws = GitWorkspace(base_files={"main.py": "x = 1\n"})
    same = ws.with_files({"main.py": "x = 1\n"})
    assert len(same.patches) == len(ws.patches)  # no empty patch appended
    same.materialize(tmp_path / "c")  # and it still rebuilds


def test_with_files_rejects_unsafe_path():
    ws = GitWorkspace(base_files={"main.py": "x = 1\n"})
    with pytest.raises(WorkspaceError, match="unsafe"):
        ws.with_files({"../evil": "x"})


def test_file_workspace_refuses_new_files():
    with pytest.raises(WorkspaceError, match="cannot grow"):
        FileWorkspace("x = 1").with_files({"main.py": "x = 2", "extra.py": "y"})
# -- M3 agent workspace capture -------------------------------------------------


def test_file_workspace_captures_modified_main(tmp_path):
    parent = FileWorkspace("x = 1\n")
    work = parent.materialize(tmp_path / "work")

    (work / "main.py").write_text("x = 2\n")

    child = parent.capture_child(work)

    assert isinstance(child, FileWorkspace)
    assert child.main_text() == "x = 2\n"
    assert parent.main_text() == "x = 1\n"


def test_file_workspace_rejects_extra_agent_files(tmp_path):
    parent = FileWorkspace("x = 1\n")
    work = parent.materialize(tmp_path / "work")
    (work / "helper.py").write_text("y = 2\n")

    with pytest.raises(WorkspaceError, match="extra files"):
        parent.capture_child(work)


def test_git_workspace_captures_modify_create_delete(tmp_path):
    parent = GitWorkspace(
        base_files={
            "main.py": "x = 1\n",
            "old.py": "obsolete = True\n",
            "keep.py": "unchanged = True\n",
        }
    )
    work = parent.materialize(tmp_path / "work")

    (work / "main.py").write_text("from helper import value\nx = value\n")
    (work / "helper.py").write_text("value = 2\n")
    (work / "old.py").unlink()

    child = parent.capture_child(work)

    assert isinstance(child, GitWorkspace)
    assert len(child.patches) == len(parent.patches) + 1

    rebuilt = child.materialize(tmp_path / "rebuilt")
    assert (rebuilt / "main.py").read_text() == (
        "from helper import value\nx = value\n"
    )
    assert (rebuilt / "helper.py").read_text() == "value = 2\n"
    assert not (rebuilt / "old.py").exists()
    assert (rebuilt / "keep.py").read_text() == "unchanged = True\n"


def test_git_workspace_capture_noop_does_not_append_patch(tmp_path):
    parent = GitWorkspace(base_files={"main.py": "x = 1\n"})
    work = parent.materialize(tmp_path / "work")

    child = parent.capture_child(work)

    assert isinstance(child, GitWorkspace)
    assert len(child.patches) == len(parent.patches)
    assert child.serialize() == parent.serialize()


def test_capture_ignores_agent_git_history(tmp_path):
    parent = GitWorkspace(base_files={"main.py": "x = 1\n"})
    work = parent.materialize(tmp_path / "work")
    (work / "main.py").write_text("x = 2\n")

    subprocess.run(
        [
            "git",
            "-C",
            str(work),
            "-c",
            "user.name=Agent",
            "-c",
            "user.email=agent@example.test",
            "add",
            "-A",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(work),
            "-c",
            "user.name=Agent",
            "-c",
            "user.email=agent@example.test",
            "commit",
            "-q",
            "-m",
            "agent commit",
        ],
        check=True,
    )

    # HEAD 已被 Agent 移动，但最终文件仍必须被捕获。
    child = parent.capture_child(work)
    rebuilt = child.materialize(tmp_path / "rebuilt")

    assert (rebuilt / "main.py").read_text() == "x = 2\n"
    assert len(child.patches) == len(parent.patches) + 1


def test_capture_rejects_binary_files(tmp_path):
    parent = GitWorkspace(base_files={"main.py": "x = 1\n"})
    work = parent.materialize(tmp_path / "work")
    (work / "payload.bin").write_bytes(b"\xff\xfe\x00")

    with pytest.raises(WorkspaceError, match="binary"):
        parent.capture_child(work)