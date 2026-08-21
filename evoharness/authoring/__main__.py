"""Check that a task directory can actually become a task.

Written for the loop where a directory is authored, run, and revised. Without
this step the first thing that tells an author their declaration is wrong is a
run that already spent twenty minutes reaching the same conclusion.

It loads the directory the way a run does, so a pass means the run will get as
far as scoring. What it cannot tell you is whether the scoring rule is any
good — that is the domain expert's judgement, and no mechanical check stands
in for it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .task_dir import TaskDirError, directory_content_hash, load_task_from_dir

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_REFUSED = 2


def _emit(payload: object) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, default=str)
    sys.stdout.write("\n")


def describe(root: Path) -> dict:
    """What the loader made of the directory, in the terms a run will use."""

    task = load_task_from_dir(root)
    spec = task.spec
    workspace = task.initial_workspace
    return {
        "ok": True,
        "task_id": spec.task_id,
        "version": spec.version,
        "task_hash": spec.hash,
        "content_hash": directory_content_hash(root),
        "criterion_hash": spec.criterion.hash,
        "measurement_hash": spec.measurement.hash,
        # The counts an author gets wrong in silence. A knowledge folder that
        # matched no files and a preflight key that was misspelled both load
        # cleanly, and both mean the run is not the experiment intended.
        "knowledge_chunks": len(spec.knowledge),
        "knowledge_bytes": sum(
            len(chunk.encode("utf-8")) for chunk in spec.knowledge
        ),
        "domain_prompt_bytes": len(spec.domain_prompt.encode("utf-8")),
        "preflight_validators": len(task.preflight_validators),
        "seed_main": workspace.main_file,
        "seed_files": sorted(workspace.texts()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evoharness.authoring", description=__doc__
    )
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="load a task directory and report")
    check.add_argument("--task-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _emit(describe(args.task_dir))
    except (TaskDirError, OSError, ValueError) as exc:
        # ValueError included on purpose: a malformed spec is refused by the
        # contract layer, and that is the author's mistake to fix, not a bug
        # to report.
        _emit({"ok": False, "error": str(exc), "kind": type(exc).__name__})
        return EXIT_REFUSED
    except Exception as exc:  # noqa: BLE001
        _emit(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "kind": "unexpected",
            }
        )
        return EXIT_UNEXPECTED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
