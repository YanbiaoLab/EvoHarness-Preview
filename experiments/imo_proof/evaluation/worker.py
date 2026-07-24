
"""Independent evaluation worker and isolated candidate subprocess entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable

import dotenv

dotenv.load_dotenv()

CONTROL_OUT = sys.stdout


def _send(value: dict[str, object]) -> None:
    CONTROL_OUT.write(json.dumps(value, ensure_ascii=False) + "\n")
    CONTROL_OUT.flush()


class _RpcLLM:
    def complete(self, stage: str, prompt: str) -> str:
        _send({"type": "llm_request", "stage": stage, "prompt": prompt})
        line = sys.stdin.readline()
        if not line:
            raise RuntimeError("model broker closed")
        response = json.loads(line)
        if response.get("type") == "error":
            raise RuntimeError(str(response.get("message", "model broker error")))
        if (
            response.get("type") != "llm_response"
            or not isinstance(response.get("text"), str)
        ):
            raise RuntimeError("invalid model broker response")
        return response["text"]


def _load_solver(root: Path, module_name: str, function_name: str):
    path = root / (module_name.replace(".", "/") + ".py")
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("benchmark_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load candidate module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    solver = getattr(module, function_name, None)
    if not callable(solver):
        raise TypeError(f"candidate entrypoint {function_name!r} is not callable")
    return solver


def _normalize_candidate_result(value: object) -> dict[str, object]:
    if isinstance(value, str):
        return {"proof": value, "status": "completed", "metadata": {}}
    if isinstance(value, dict):
        return {
            "proof": value.get("proof"),
            "status": value.get("status", "completed"),
            "metadata": value.get("metadata", {}),
        }
    return {
        "proof": getattr(value, "proof", None),
        "status": getattr(value, "status", "completed"),
        "metadata": getattr(value, "metadata", {}),
    }


def _candidate_main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit("usage: worker.py candidate ROOT module:function")
    root = Path(argv[0]).resolve()
    module_name, function_name = argv[1].split(":", 1)
    request = json.loads(sys.stdin.readline())
    problem = request.get("problem")
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("problem must be non-empty")
    with contextlib.redirect_stdout(sys.stderr):
        solver = _load_solver(root, module_name, function_name)
        result = solver(problem, _RpcLLM())
    normalized = _normalize_candidate_result(result)
    if not isinstance(normalized["proof"], str) or not normalized["proof"].strip():
        raise ValueError("candidate returned an empty proof")
    _send({"type": "result", **normalized})
    return 0


class LocalProcessBackend:
    """Run the task evaluator through its command-line worker contract."""

    def __init__(
        self,
        *,
        project_root: Path,
        spec_path: Path,
        timeout_s: float = 3600.0,
        env: dict[str, str] | None = None,
        runner: Callable[..., subprocess.CompletedProcess] | None = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.spec_path = Path(spec_path).resolve()
        self.timeout_s = timeout_s
        self.env = dict(env or {})
        self.runner = runner or subprocess.run

    def evaluate_directory(
        self,
        *,
        candidate_id: str,
        candidate_root: Path,
        split: str,
        output_dir: Path | None = None,
    ):
        from .contract import (
            CandidateEvaluation,
            EvaluationProtocolError,
            EvaluationUnavailable,
        )

        with tempfile.TemporaryDirectory(prefix="imo_eval_worker_") as temporary:
            result_dir = (
                Path(output_dir).resolve()
                if output_dir is not None
                else Path(temporary) / "result"
            )
            result_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "experiments.imo_proof.evaluation.worker",
                "evaluate",
                "--project-root",
                str(self.project_root),
                "--spec",
                str(self.spec_path),
                "--candidate-dir",
                str(Path(candidate_root).resolve()),
                "--candidate-id",
                candidate_id,
                "--split",
                split,
                "--result-dir",
                str(result_dir),
            ]
            completed = self.runner(
                command,
                cwd=self.project_root,
                env={**os.environ, **self.env},
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                raise EvaluationUnavailable(
                    f"evaluation worker exited with {completed.returncode}: "
                    f"{detail[-2000:]}"
                )
            path = result_dir / "evaluation.json"
            if not path.is_file():
                raise EvaluationProtocolError(
                    "evaluation worker did not write evaluation.json"
                )
            try:
                evaluation = CandidateEvaluation.read(path)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise EvaluationProtocolError(
                    f"invalid evaluation worker result: {exc}"
                ) from exc
            if evaluation.candidate_id != candidate_id or evaluation.split != split:
                raise EvaluationProtocolError(
                    "evaluation worker returned mismatched candidate or split"
                )
            return evaluation


def _live_client(spec, model, api_base: str, api_key: str):
    from evoharness.evocore import LLMClient

    from ..transport import SpecOpenAITransport

    return LLMClient(
        model.temperature,
        model.max_output_tokens,
        transport=SpecOpenAITransport(
            api_base,
            api_key,
            model.enable_thinking,
            model.input_cost_per_million,
            model.output_cost_per_million,
        ),
    )


def _evaluation_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run one IMO evaluation job.")
    parser.add_argument("command", choices=("evaluate",))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        required=True,
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    from ..protocol import BenchmarkSpec
    from .contract import EvaluationUnavailable
    from .engine import make_live_evaluator

    api_base = os.environ.get("EVOHARNESS_API_BASE")
    api_key = os.environ.get("EVOHARNESS_API_KEY")
    if not api_base or not api_key:
        raise EvaluationUnavailable(
            "evaluation worker requires EVOHARNESS_API_BASE and "
            "EVOHARNESS_API_KEY"
        )
    spec = BenchmarkSpec.load(args.spec)
    evaluator = make_live_evaluator(
        project_root=args.project_root,
        spec=spec,
        solver_client=_live_client(spec, spec.solver, api_base, api_key),
        grader_client=_live_client(spec, spec.grader.model, api_base, api_key),
    )
    evaluator.evaluate_directory(
        candidate_id=args.candidate_id,
        candidate_root=args.candidate_dir,
        split=args.split,
        output_dir=args.result_dir,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "candidate":
        try:
            return _candidate_main(args[1:])
        except Exception as exc:
            # Full chained traceback to stderr for debugging; the one-line
            # message stays on the control channel for the protocol.
            traceback.print_exc()
            _send(
                {
                    "type": "worker_error",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
            return 1
    try:
        return _evaluation_main(args)
    except Exception:
        # print_exc walks the __cause__ chain (engine -> transport -> HTTP
        # body), which the previous one-line str(exc) discarded.
        traceback.print_exc()
        return 75


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["LocalProcessBackend", "main"]
