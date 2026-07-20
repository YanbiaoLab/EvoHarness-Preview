# EvoHarness original (WS-3 M1): the genome becomes a Workspace. FileWorkspace
# is the single-file degenerate case: its serialize() IS the raw code string,
# so the store's `code` column keeps its historical meaning and old run.db
# files load unchanged. GitWorkspace is the multi-file substrate: genome =
# base snapshot + ordered unified-diff chain, rebuilt with `git apply`
# (reference: third_party/HyperAgents archive.jsonl patch-chain rebuild).
"""Workspace abstraction: what a candidate's genome IS (WS-3 M1)."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable


MAX_TEXT_FILE_SIZE = 2 * 1024 * 1024


def _read_text_tree(root: Path) -> dict[str, str]:
    """Read candidate-visible UTF-8 text files from workspace."""
    root = Path(root).resolve()

    if not root.is_dir():
        raise WorkspaceError(f"workspace directory does not exist: {root}")

    texts: dict[str, str] = {}

    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if ".." in rel.parts:
            raise WorkspaceError(f"invalid workspace path: {rel}")

        if ".git" in rel.parts:
            continue

        if path.is_symlink():
            raise WorkspaceError(f"symlinks are not supported: {rel}")

        if not path.is_file():
            continue

        if path.stat().st_size > MAX_TEXT_FILE_SIZE:
            raise WorkspaceError(f"candidate file too large: {rel}")

        try:
            texts[rel.as_posix()] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceError(f"binary files are not supported: {rel}") from exc

    return texts

class WorkspaceError(RuntimeError):
    """The genome cannot be reconstructed (bad patch, unsafe path, no git)."""


@runtime_checkable
class Workspace(Protocol):
    kind: str  # "file" or "git"

    def main_text(self) -> str:
        ...

    def with_files(self, edits: dict[str, str]) -> Workspace:
        ...

    def materialize(self, dest: Path) -> Path:
        ...

    def serialize(self) -> str:
        ...

    def with_main_text(self, new_text: str) -> Workspace:
        ...

    def texts(self) -> dict[str, str]:
        ...

    def capture_child(self, source: Path) -> Workspace:
        """Capture final files from source as one atomic child genome."""
        ...


@dataclass
class FileWorkspace:
    code: str
    filename: str = "main.py"
    kind: ClassVar[str] = "file"

    def main_text(self) -> str:
        return self.code

    def with_main_text(self, new_text: str) -> FileWorkspace:
        return self.with_files({self.filename: new_text})

    def with_files(self, edits: dict[str, str]) -> FileWorkspace:
        extra = set(edits) - {self.filename}
        if extra:
            raise WorkspaceError(
                f"single-file genome cannot grow files: {sorted(extra)}"
            )
        return FileWorkspace(edits.get(self.filename, self.code), self.filename)

    def materialize(self, dest: Path) -> Path:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / self.filename).write_text(self.code)
        return dest

    def serialize(self) -> str:
        return self.code

    def texts(self) -> dict[str, str]:
        return {self.filename: self.code}

    def capture_child(self, source: Path) -> FileWorkspace:
        texts = _read_text_tree(source)

        if self.filename not in texts:
            raise WorkspaceError(
                f"main file {self.filename!r} missing after agent session"
            )

        extra = set(texts) - {self.filename}
        if extra:
            raise WorkspaceError(
                "single-file genome cannot capture extra files: "
                f"{sorted(extra)}"
            )

        return FileWorkspace(
            code=texts[self.filename],
            filename=self.filename,
        )


def _git(cwd: Path, *args: str, input_text: str | None = None) -> str:
    if shutil.which("git") is None:
        raise WorkspaceError("git binary not found; GitWorkspace requires it")

    cmd = [
        "git",
        "-c",
        "user.name=EvoHarness",
        "-c",
        "user.email=evo@local",
        *args,
    ]

    proc = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        cwd=cwd,
    )
    return_code = proc.returncode
    if return_code != 0:
        raise WorkspaceError(f"git {args[0]} failed: {proc.stderr.strip()}")
    return proc.stdout


@dataclass
class GitWorkspace:
    base_files: dict[str, str]
    patches: list[str] = field(default_factory=list)
    main_file: str = "main.py"
    kind: ClassVar[str] = "git"
    _cache: Path | None = field(default=None, repr=False, compare=False)

    def materialize(self, dest: Path) -> Path:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        for rel in self.base_files:
            parts = Path(rel).parts
            if Path(rel).is_absolute() or ".." in parts:
                raise WorkspaceError(f"unsafe path in base_files: {rel}")

        for rel, text in self.base_files.items():
            path = dest / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

        _git(dest, "init", "-q")
        _git(dest, "add", "-A")
        _git(dest, "commit", "-q", "--allow-empty", "-m", "base")
        for i, patch in enumerate(self.patches):
            _git(dest, "apply", "--whitespace=nowarn", "-", input_text=patch)
            _git(dest, "add", "-A")
            _git(dest, "commit", "-q", "-m", f"patch {i + 1}")

        if not (dest / self.main_file).exists():
            raise WorkspaceError(
                f"main file {self.main_file!r} missing after rebuild"
            )
        return dest

    def main_text(self) -> str:
        if self._cache is None:
            self._cache = self.materialize(
                Path(tempfile.mkdtemp(prefix="evoworkspace_"))
            )
        return (self._cache / self.main_file).read_text()

    def with_main_text(self, new_text: str) -> GitWorkspace:
        return self.with_files({self.main_file: new_text})

    def with_files(self, edits: dict[str, str]) -> GitWorkspace:
        for rel in edits:
            if Path(rel).is_absolute() or ".." in Path(rel).parts:
                raise WorkspaceError(f"unsafe path in edits: {rel}")
        work = self.materialize(Path(tempfile.mkdtemp(prefix="evoworkspace_")))

        for rel, text in edits.items():
            path = work / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

        _git(work, "add", "-A")
        patch = _git(work, "diff", "--cached")

        if not patch.strip():
            return GitWorkspace(
                dict(self.base_files),
                list(self.patches),
                self.main_file,
            )

        return self.child(patch)

    def capture_child(self, source: Path) -> GitWorkspace:
        current_texts = _read_text_tree(source)

        if self.main_file not in current_texts:
            raise WorkspaceError(
                f"main file {self.main_file!r} missing after agent session"
            )

        with tempfile.TemporaryDirectory(prefix="evocapture_") as temp_dir:
            clean = self.materialize(Path(temp_dir))
            parent_texts = _read_text_tree(clean)

            if current_texts == parent_texts:
                return GitWorkspace(
                    base_files=dict(self.base_files),
                    patches=list(self.patches),
                    main_file=self.main_file,
                )

            for rel in set(parent_texts) - set(current_texts):
                (clean / rel).unlink()

            for rel, text in current_texts.items():
                path = clean / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)

            _git(clean, "add", "-A")
            patch = _git(
                clean,
                "diff",
                "--cached",
                "--binary",
                "HEAD",
            )

        if not patch.strip():
            raise WorkspaceError(
                "workspace files changed but no canonical patch was produced"
            )

        return self.child(patch)

    def child(self, patch: str) -> "GitWorkspace":
        return GitWorkspace(
            dict(self.base_files),
            [*self.patches, patch],
            self.main_file,
        )

    def serialize(self) -> str:
        return json.dumps(
            {
                "base_files": self.base_files,
                "main_file": self.main_file,
                "patches": self.patches,
            },
            ensure_ascii=False,
        )

    @classmethod
    def deserialize(cls, blob: str) -> "GitWorkspace":
        try:
            data = json.loads(blob)
            return cls(
                dict(data["base_files"]),
                list(data["patches"]),
                str(data["main_file"]),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise WorkspaceError(f"unreadable git workspace blob: {e}") from e

    def texts(self) -> dict[str, str]:
        if self._cache is None:
            self._cache = self.materialize(
                Path(tempfile.mkdtemp(prefix="evoworkspace_"))
            )
        out: dict[str, str] = {}
        for p in self._cache.rglob("*"):
            if not p.is_file() or ".git" in p.parts:
                continue
            try:
                rel = str(p.relative_to(self._cache))
                out[rel] = p.read_text()
            except UnicodeDecodeError:
                continue

        return out


def load_workspace(kind: str, blob: str) -> FileWorkspace | GitWorkspace:
    if kind == "file":
        return FileWorkspace(code=blob)
    if kind == "git":
        return GitWorkspace.deserialize(blob)
    raise WorkspaceError(f"unknown workspace kind: {kind}")
