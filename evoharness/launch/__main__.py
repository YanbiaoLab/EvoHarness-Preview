"""Run whatever `job.json` in a run directory says to run.

This is what a detached start executes. It takes one argument on purpose: the
run directory already holds its own invocation, so a resume is the same
command as the original start and cannot accidentally carry a different
configuration than the run it claims to continue.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .build import build, configure_logging, execute
from .config import LaunchConfig, LaunchConfigError
from .starter import JOB_FILE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    job = args.run_dir / JOB_FILE
    if not job.is_file():
        parser.error(f"{job} not found; a run directory carries its own {JOB_FILE}")

    payload = json.loads(job.read_text(encoding="utf-8"))
    # `start_run` records how it launched the process; the settings it launched
    # WITH live under their own key, so the two never have to be told apart by
    # guessing which keys belong to which.
    settings = payload.get("launch", payload)
    try:
        cfg = LaunchConfig.from_json(settings)
    except LaunchConfigError as exc:
        parser.error(str(exc))

    # The recorded run directory may have moved with the files. What the
    # caller pointed at now is authoritative, or a copied run would write its
    # results back into the original.
    cfg = LaunchConfig(**{**cfg.to_json(), "run_dir": str(args.run_dir)})

    configure_logging()
    summary = execute(build(cfg), cfg, argv=argv)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
