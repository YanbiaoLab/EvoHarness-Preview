# evoserve/__main__.py
"""CLI: python -m evoharness.evoserve --grade-fn tasks.modmul:grade_fn --task-version v1 ..."""

from __future__ import annotations

import argparse
import importlib

from .http import serve
from .service import EvalService


def load_grade_fn(spec: str):
    """'pkg.module:attr' -> callable (veRL's custom_reward_function pattern)."""
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"--grade-fn must look like 'pkg.module:fn', got {spec!r}")
    fn = getattr(importlib.import_module(module_name), attr)
    if not callable(fn):
        raise SystemExit(f"{spec} is not callable")
    return fn


def main() -> None:
    p = argparse.ArgumentParser(prog="evoserve")
    p.add_argument("--grade-fn", required=True, help="pkg.module:fn")
    p.add_argument("--task-version", required=True)
    p.add_argument("--eval-set-version", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8321)
    p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--token", default=None, help="bearer token; omit to disable auth")
    args = p.parse_args()

    service = EvalService(
        load_grade_fn(args.grade_fn),
        task_version=args.task_version,
        eval_set_version=args.eval_set_version,
        max_workers=args.max_workers,
    )
    server = serve(service, args.host, args.port, args.token)
    print(
        f"evoserve listening on http://{args.host}:{server.server_address[1]} "
        f"(task={args.task_version}, eval_set={args.eval_set_version})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
