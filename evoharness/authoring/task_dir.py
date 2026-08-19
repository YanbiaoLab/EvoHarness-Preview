"""Load an authored task directory into a `ResolvedTask`.

Tasks have until now been Python modules in the `tasks/` registry, written by
hand. An authored task arrives instead as a directory of files, which is what
a model can produce and what a human can read in a diff.

The declaration is deliberately small. Everything that decides whether a
candidate is good — the scoring rule — is ordinary Python in `grade.py`,
written by whoever owns the domain; this module does not second-guess it. What
it does own is the boring, mechanical part: read the declaration, refuse the
malformed ones, and hand back a task whose identity is its contents.

Layout::

    task_dir/
    ├── task.json      # the declaration: paths and numbers, no prose
    ├── prompt.md      # the task statement the agent reads every generation
    ├── grade.py       # def grade(candidate_dir, ctx) -> fitness or dict
    ├── seed/          # initial workspace, main file included
    └── docs/          # optional reference material named by `knowledge`

One rule covers every text field: `task.json` gives a PATH, never the prose
itself. Long text belongs in a file a human can read and revise, not in a JSON
string literal with its newlines escaped.

JSON rather than YAML: the same canonical-JSON hashing the specs already use,
and no indentation traps for an author writing the file programmatically.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evoharness.contracts import CriterionSpec, MeasurementSpec
from evoharness.core.workspace import GitWorkspace
from evoharness.runtime.grading import WorkspaceGradeFnGrader
from evoharness.runtime.task import ResolvedTask
from evoharness.serve import GradeContext

from .command_check import DEFAULT_TIMEOUT_S, CommandCheck

TASK_FILE = "task.json"


class TaskDirError(RuntimeError):
    """The authored directory is not a loadable task."""


#: Directories that are build output or version-control bookkeeping rather
#: than authored content. Skipped both when hashing and when reading reference
#: material: a checkout's `.git` would otherwise dominate a knowledge folder
#: and change the task hash on every unrelated commit.
_SKIP_DIRS = {
    "__pycache__", ".git", ".pytest_cache", ".ruff_cache",
    "node_modules", ".venv", ".mypy_cache",
}

#: Total reference material handed to the agent, in bytes. Roughly 60k tokens —
#: already a large share of a proposal's context. A knowledge folder that
#: silently exceeded this would produce an unusable prompt and a large bill,
#: and the cause would not be visible from either.
DEFAULT_KNOWLEDGE_MAX_BYTES = 256 * 1024


def _text_files(root: Path):
    """Every readable text file under `root`, in a stable order."""

    if root.is_file():
        yield root, root.name
        return
    for path in sorted(root.rglob("*")):
        if any(part in _SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            yield path, path.relative_to(root).as_posix()


def directory_content_hash(root: Path) -> str:
    """Hash what the task directory CONTAINS, not where it sits.

    The task identity has to travel: the same directory on another machine, or
    moved to another path, must be the same task. Recording the absolute path
    instead would put the checkout location into `TaskSpec.hash` and make two
    identical tasks compare as different experiments — the exact failure the
    spec-hash machinery exists to prevent.

    Reference material living OUTSIDE this directory needs no special handling
    here: `TaskSpec` carries the knowledge text itself, so editing an external
    knowledge base already changes the task hash through its contents.
    """

    entries: list[tuple[str, str]] = []
    for path, relative in _text_files(root):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append((relative, digest))
    payload = json.dumps(sorted(entries), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require(mapping: dict, key: str, where: str) -> object:
    if key not in mapping:
        raise TaskDirError(f"{where}: missing required key {key!r}")
    return mapping[key]


def _load_declaration(root: Path) -> dict:
    path = root / TASK_FILE
    if not path.is_file():
        raise TaskDirError(f"{root} has no {TASK_FILE}")
    try:
        declaration = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TaskDirError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(declaration, dict):
        raise TaskDirError(f"{path} must contain a JSON object")
    return declaration


def _resolve_inside(root: Path, relative: str, where: str) -> Path:
    """Resolve a declared path, refusing to leave the task directory.

    An authored declaration is untrusted input: without this an author could
    point `seed.dir` at the evaluation harness and have it materialized as a
    candidate genome.
    """

    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise TaskDirError(f"{where}: {relative!r} escapes the task directory")
    return candidate


def _load_prompt(root: Path, declaration: dict) -> str:
    """Read the task statement the agent sees every generation.

    A path rather than a JSON string: the statement is prose that a human
    revises, and escaping newlines into a string literal makes it unreadable
    in exactly the file where clarity matters most.
    """

    relative = str(_require(declaration, "prompt", TASK_FILE))
    path = _resolve_inside(root, relative, "prompt")
    if not path.is_file():
        raise TaskDirError(f"prompt: no such file: {path}")
    return path.read_text(encoding="utf-8")


def _knowledge_roots(root: Path, declaration: dict) -> tuple[Path, ...]:
    """Resolve the declared reference-material paths.

    Unlike `seed.dir` these may point outside the task directory: a knowledge
    base is often a checkout someone else maintains. That is safe because the
    material is only ever read into a prompt, never materialized as a genome —
    but it is only CORRECT because those trees are hashed into the task
    identity as well.
    """

    raw = declaration.get("knowledge", [])
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise TaskDirError("knowledge must be a list of paths")
    roots: list[Path] = []
    for index, item in enumerate(raw):
        where = f"knowledge[{index}]"
        path = (root / str(item)).resolve()
        if not path.exists():
            raise TaskDirError(f"{where}: no such file or directory: {path}")
        roots.append(path)
    return tuple(roots)


def _read_knowledge(
    roots: tuple[Path, ...], max_bytes: int
) -> tuple[str, ...]:
    """Read reference material, labelled by path and bounded in size.

    Each chunk carries its own path because a folder becomes many files, and a
    wall of concatenated prose with no boundaries is worse than no reference
    material at all — the agent cannot tell an API reference from a worked
    example.
    """

    chunks: list[str] = []
    total = 0
    largest: list[tuple[int, str]] = []
    for tree in roots:
        for path, relative in _text_files(tree):
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                # Binary or unreadable files inside a checkout are skipped
                # rather than refused: a knowledge folder legitimately
                # contains images and archives beside its prose.
                continue
            size = len(text.encode("utf-8"))
            total += size
            largest.append((size, f"{tree.name}/{relative}"))
            chunks.append(f"# {tree.name}/{relative}\n\n{text}")
    if total > max_bytes:
        worst = ", ".join(
            f"{name} ({size // 1024} KiB)"
            for size, name in sorted(largest, reverse=True)[:5]
        )
        raise TaskDirError(
            f"knowledge is {total // 1024} KiB, over the "
            f"{max_bytes // 1024} KiB limit; the agent reads all of it every "
            f"generation. Narrow the paths or raise knowledge_max_bytes. "
            f"Largest: {worst}"
        )
    return tuple(chunks)


def _load_preflight(declaration: dict) -> tuple[CommandCheck, ...]:
    """Build the task's own candidate checks from the declaration."""

    raw = declaration.get("preflight", [])
    if not isinstance(raw, list):
        raise TaskDirError("preflight must be a list of checks")
    checks: list[CommandCheck] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        where = f"preflight[{index}]"
        if not isinstance(item, dict):
            raise TaskDirError(f"{where} must be an object")
        name = str(_require(item, "name", where))
        if name in seen:
            raise TaskDirError(f"{where}: duplicate check name {name!r}")
        seen.add(name)
        argv = _require(item, "argv", where)
        if not isinstance(argv, list) or not argv:
            raise TaskDirError(f"{where}.argv must be a non-empty list")
        checks.append(
            CommandCheck(
                name=name,
                argv=tuple(str(part) for part in argv),
                timeout_s=float(item.get("timeout_s", DEFAULT_TIMEOUT_S)),
                repairable=bool(item.get("repairable", True)),
            )
        )
    return tuple(checks)


def _load_grade_fn(module_path: Path, entry: str):
    """Import the authored grade function."""

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"evo_authored_grade_{module_path.parent.name}", module_path
    )
    if spec is None or spec.loader is None:
        raise TaskDirError(f"cannot load a module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    grade_fn = getattr(module, entry, None)
    if not callable(grade_fn):
        raise TaskDirError(f"{module_path.name} defines no callable {entry!r}")

    def grade_workspace(candidate_dir: Path, ctx: GradeContext):
        # Deliberately NOT normalized here: `WorkspaceGradeFnGrader` already
        # calls `coerce_grade`, and coercing twice fails — the first pass adds
        # `schema_version`, which the second rejects as an unknown field.
        # Normalization belongs to whoever owns the EvalReport, once.
        return grade_fn(Path(candidate_dir), ctx)

    grade_workspace.__name__ = f"authored_{entry}"
    # `ResolvedTask._component_config` hashes the wrapped implementation, so
    # the authored source lands in the TaskSpec by itself: editing grade.py is
    # a different task, with no separate bookkeeping to forget.
    grade_workspace.__wrapped__ = grade_fn
    return grade_workspace


def load_task_from_dir(root: str | Path) -> ResolvedTask:
    """Turn an authored directory into a task."""

    root = Path(root).resolve()
    if not root.is_dir():
        raise TaskDirError(f"{root} is not a directory")
    declaration = _load_declaration(root)

    grade_section = declaration.get("grade", {})
    if not isinstance(grade_section, dict):
        raise TaskDirError("grade must be an object")
    module_path = _resolve_inside(
        root, str(grade_section.get("module", "grade.py")), "grade.module"
    )
    entry = str(grade_section.get("entry", "grade"))
    if not module_path.is_file():
        raise TaskDirError(f"grade module not found: {module_path}")

    seed_section = declaration.get("seed", {})
    if not isinstance(seed_section, dict):
        raise TaskDirError("seed must be an object")
    seed_dir = _resolve_inside(
        root, str(seed_section.get("dir", "seed")), "seed.dir"
    )
    main_file = str(seed_section.get("main_file", "main.py"))
    if not seed_dir.is_dir():
        raise TaskDirError(f"seed directory not found: {seed_dir}")

    measurement_raw = _require(declaration, "measurement", TASK_FILE)
    if not isinstance(measurement_raw, dict):
        raise TaskDirError("measurement must be an object")
    if "planned_units" not in measurement_raw:
        # Zero is legal and means coverage can never be complete. Defaulting to
        # it would make every run of an authored task quietly unusable, so the
        # author has to write the zero on purpose.
        raise TaskDirError(
            "measurement.planned_units must be stated explicitly; 0 is legal "
            "but means coverage is never complete, which is not a default "
            "anyone should arrive at by omission"
        )

    criterion_raw = declaration.get("criterion")
    criterion = (
        CriterionSpec(**criterion_raw)
        if isinstance(criterion_raw, dict)
        else None
    )

    return ResolvedTask.create(
        task_id=str(_require(declaration, "task_id", TASK_FILE)),
        version=str(declaration.get("version", "v1")),
        grader=WorkspaceGradeFnGrader(_load_grade_fn(module_path, entry)),
        initial_workspace=GitWorkspace.from_directory(
            seed_dir, main_file=main_file
        ),
        domain_prompt=_load_prompt(root, declaration),
        knowledge=_read_knowledge(
            _knowledge_roots(root, declaration),
            int(
                declaration.get(
                    "knowledge_max_bytes", DEFAULT_KNOWLEDGE_MAX_BYTES
                )
            ),
        ),
        criterion=criterion,
        measurement=MeasurementSpec(**measurement_raw),
        preflight_validators=_load_preflight(declaration),
        # Everything here enters `TaskSpec.to_payload()` and therefore the task
        # hash, so it must contain only things that make this a DIFFERENT task.
        # The directory's location is not one of them; the run manifest records
        # argv if a human needs to find the source again.
        metadata_json=json.dumps(
            {"authored": True, "source_hash": directory_content_hash(root)},
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
