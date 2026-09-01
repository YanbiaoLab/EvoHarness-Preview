"""Start a run from the command line and return immediately.

`python -m evoharness.launch` IS the run — it blocks for hours, which is right
for the process doing the work and useless to anything that owes a caller an
answer. This starts that command detached and prints where it went.

Two entry points rather than one with a flag: a command that sometimes blocks
and sometimes does not is one mistake away from a caller waiting forever.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import evoharness

from .config import LaunchConfig, LaunchConfigError
from .starter import StartError, start_run

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_REFUSED = 2


def repo_root() -> Path:
    """Where the child must run so `recipes` and `tasks` import.

    Not the caller's working directory. A tool invoking this from a candidate
    workspace would start a child that dies on `import recipes`, and because
    the child is detached that failure surfaces only in its log — the start
    itself would look like it worked.

    Wrong if evoharness is installed rather than run from a checkout; `--cwd`
    is the way out.
    """

    if evoharness.__file__ is None:
        # A namespace package has no file to locate the checkout from. Raised
        # as a refusal, not an OSError: OSError lands in the unexpected branch
        # and the caller is told nothing it can act on.
        raise LaunchConfigError(
            "cannot locate the EvoHarness checkout; pass --cwd explicitly"
        )
    return Path(evoharness.__file__).resolve().parent.parent

def _emit(payload: object) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, default=str)
    sys.stdout.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evoharness.launch.start", description=__doc__
    )
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task", default="demo_counter")
    parser.add_argument("--task-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--budget-usd", type=float, default=None)
    parser.add_argument("--brief", type=Path, default=None)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--dsh-config", type=Path, default=None)
    parser.add_argument("--dsh-runtime", type=Path, default=None)
    parser.add_argument(
        "--dsh-provider",
        default="deepseek-official",
        help="the provider route the dsh config declares; a model is resolved "
        "against that route's catalog, so a mismatch fails every request",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="the model to ask for. It has to be a name the --dsh-provider "
        "route's catalog holds: the runtime resolves the request against that "
        "catalog, and a name it does not know fails every call rather than "
        "falling back to something it has",
    )
    # `action="extend"` rather than a bare `nargs="*"`: without it a second
    # --set REPLACES the first instead of adding to it, and nothing reports
    # the loss — the dropped settings quietly take their defaults, and the
    # run records those defaults as what was asked for.
    parser.add_argument(
        "--set", dest="overrides", action="extend", nargs="*", default=[]
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=None,
        help="working directory for the run; defaults to the EvoHarness repo",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="start even though the directory holds an unfinished run; two "
             "processes writing one checkpoint corrupt it",
    )
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    overrides = list(args.overrides)
    if args.model is not None:
        # Prepended, not appended: these are the flag's expansion, so an
        # explicit --set on the same key is a deliberate override of it and
        # has to win. load_experiment_config applies them in order.
        overrides[:0] = [
            f"proposal.model={args.model}",
            f"search.llm_models={json.dumps([args.model])}",
        ]
    try:
        # Built here rather than in the child so a bad combination is rejected
        # while there is still someone to tell. The child validates it again
        # on the way in, which is what makes a resume the same run.
        cfg = LaunchConfig(
            recipe=args.recipe,
            run_dir=run_dir,
            task=args.task,
            task_dir=args.task_dir,
            config_path=args.config,
            budget_usd=args.budget_usd,
            brief=args.brief,
            live=args.live,
            dsh_config=args.dsh_config,
            dsh_runtime=args.dsh_runtime,
            dsh_provider=args.dsh_provider,
            overrides=tuple(overrides),
        )
        started = start_run(
            run_dir,
            [sys.executable,
                "-m",
                "evoharness.launch",
                "--run-dir",
                str(run_dir)

            ],
            cwd=args.cwd or repo_root(),
            launch=cfg.to_json(),
            force=args.force,
        )
    except (LaunchConfigError, StartError) as exc:
        _emit({"error": str(exc), "kind": type(exc).__name__})
        return EXIT_REFUSED
    except Exception as exc:  # noqa: BLE001
        _emit({"error": f"{type(exc).__name__}: {exc}", "kind": "unexpected"})
        return EXIT_UNEXPECTED
    _emit(started.to_json())
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
