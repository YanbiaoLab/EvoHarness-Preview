"""Run one candidate coding harness over the Polyglot Python exercises.

Launched as a subprocess by `grade.py` so that a candidate that leaks memory,
spawns threads, or imports something poisonous cannot damage the search loop.
Writes a single JSON report to --output and returns 0 unless the harness could
not be run at all (which is infrastructure, not a candidate's fault).

The candidate never sees the reference solution, and the score always comes
from a fresh test run the candidate did not perform itself.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polyglot_py import corpus  # noqa: E402


class BudgetExhausted(RuntimeError):
    """Raised into the candidate when it exceeds a per-exercise cap."""


class _LLM:
    """The model handle handed to the candidate harness."""

    def __init__(self, client, model: str, max_calls: int):
        self._client = client
        self._model = model
        self._max_calls = max_calls
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost = 0.0

    def complete(self, stage: str, prompt: str, system: str = "") -> str:
        if self.calls >= self._max_calls:
            raise BudgetExhausted(
                f"model call budget exhausted ({self._max_calls} calls)"
            )
        self.calls += 1
        response = self._client.query(
            system or "You are a careful Python programmer.",
            prompt,
            self._model,
        )
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens
        self.cost += response.cost
        return response.text


class _Tools:
    """Everything the candidate harness may do besides call the model."""

    def __init__(self, exercise, workroot: Path, max_runs: int, timeout_s: float):
        self._exercise = exercise
        self._workroot = workroot
        self._max_runs = max_runs
        self._timeout_s = timeout_s
        self.runs = 0

    def run_tests(self, files: dict) -> dict:
        """Run the exercise's own test suite against `files`.

        Returns {"passed": bool, "output": str}. Costs one of a small number
        of runs — the cap is what stops a candidate from turning the grader
        into its own search loop.
        """

        if self.runs >= self._max_runs:
            raise BudgetExhausted(f"test-run budget exhausted ({self._max_runs} runs)")
        self.runs += 1
        result = run_tests(
            self._exercise,
            files,
            self._workroot / f"try_{self.runs:02d}",
            timeout_s=self._timeout_s,
        )
        return {"passed": result["passed"], "output": result["output"]}


def run_tests(exercise, files: dict, workdir: Path, *, timeout_s: float) -> dict:
    """Materialise a candidate's files beside the real tests and run pytest."""

    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    written = 0
    for name, text in (files or {}).items():
        # A harness that tries to write outside the exercise directory is
        # rejected rather than sandboxed: nothing legitimate needs it.
        target = (workdir / str(name)).resolve()
        if not str(target).startswith(str(workdir.resolve())):
            return {
                "passed": False,
                "output": f"refused to write outside the workspace: {name}",
                "error_category": "path-escape",
            }
        if not isinstance(text, str):
            return {
                "passed": False,
                "output": f"file {name} is {type(text).__name__}, expected str",
                "error_category": "bad-file-type",
            }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        written += 1
    if written == 0:
        return {
            "passed": False,
            "output": "the harness returned no files",
            "error_category": "no-files",
        }

    for name, text in exercise.tests.items():
        (workdir / name).write_text(text, encoding="utf-8")

    command = [
        sys.executable, "-m", "pytest",
        *sorted(exercise.tests),
        "-q", "--no-header", "-x", "-p", "no:cacheprovider",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "output": f"tests exceeded {timeout_s:.0f}s",
            "error_category": "test-timeout",
        }
    output = ((completed.stdout or "") + (completed.stderr or ""))[-4000:]
    passed = completed.returncode == 0
    return {
        "passed": passed,
        "output": output,
        "error_category": "" if passed else _classify(output),
    }


def _classify(output: str) -> str:
    """One short, stable label per failure mode — this is what reaches the
    mutation prompt, so it must distinguish 'wrong answer' from 'never ran'."""

    if "SyntaxError" in output:
        return "syntax-error"
    if "ImportError" in output or "ModuleNotFoundError" in output:
        return "import-error"
    if "NameError" in output:
        return "name-error"
    if "TypeError" in output:
        return "type-error"
    if "AttributeError" in output:
        return "attribute-error"
    if "NotImplementedError" in output:
        return "not-implemented"
    if "assert" in output or "AssertionError" in output:
        return "assertion-failed"
    if "error" in output.lower():
        return "runtime-error"
    return "tests-failed"


def _load_solver(candidate_dir: Path):
    path = candidate_dir / "solver.py"
    if not path.is_file():
        raise FileNotFoundError("candidate has no solver.py")
    sys.path.insert(0, str(candidate_dir))
    spec = importlib.util.spec_from_file_location("polyglot_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    solve = getattr(module, "solve", None)
    if not callable(solve):
        raise TypeError("solver.py must define a callable solve(task, llm, tools)")
    return solve


def _normalise(result) -> dict:
    if isinstance(result, dict) and isinstance(result.get("files"), dict):
        return result["files"]
    if isinstance(result, dict):
        # A harness that returned {name: text} directly is being helpful, not
        # wrong; accept it rather than scoring a working solution zero.
        if all(isinstance(key, str) and isinstance(val, str)
               for key, val in result.items()):
            return dict(result)
    raise TypeError(
        "solve() must return {'files': {filename: source}}; "
        f"got {type(result).__name__}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--train-size", type=int, default=12)
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--max-llm-calls", type=int, default=6)
    parser.add_argument("--max-test-runs", type=int, default=3)
    parser.add_argument("--test-timeout-s", type=float, default=60.0)
    parser.add_argument("--exercise-timeout-s", type=float, default=600.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    from evoharness.evocore import LLMClient, make_openai_compat_transport

    api_base = os.environ.get("EVOHARNESS_API_BASE")
    api_key = os.environ.get("EVOHARNESS_API_KEY")
    if not api_base or not api_key:
        print("EVOHARNESS_API_BASE/KEY unset", file=sys.stderr)
        return 3

    exercises = corpus.select(args.split, args.train_size)
    solve = _load_solver(args.candidate_dir.resolve())

    client = LLMClient(
        temperature=0.2,
        max_tokens=8192,
        transport=make_openai_compat_transport(api_base, api_key, timeout_s=300.0),
    )

    def evaluate(exercise, workroot: Path) -> dict:
        started = time.monotonic()
        llm = _LLM(client, args.model, args.max_llm_calls)
        exercise_root = workroot / exercise.slug
        tools = _Tools(
            exercise,
            exercise_root,
            args.max_test_runs,
            args.test_timeout_s,
        )
        failure = ""
        files: dict = {}
        try:
            files = _normalise(solve(exercise.as_task(), llm, tools))
        except BudgetExhausted as exc:
            failure = f"budget: {exc}"
        except Exception as exc:  # the candidate's own bug
            failure = f"{type(exc).__name__}: {exc}"
            traceback.print_exc(file=sys.stderr)

        if failure and not files:
            return {
                "item_id": exercise.slug,
                "passed": False,
                "error_category": "harness-" + failure.split(":", 1)[0],
                "failure": failure[:300],
                "llm_calls": llm.calls,
                "test_runs": tools.runs,
                "prompt_tokens": llm.prompt_tokens,
                "completion_tokens": llm.completion_tokens,
                "cost_usd": llm.cost,
                "elapsed_s": time.monotonic() - started,
            }

        # The score always comes from a run the candidate did not do.
        verdict = run_tests(
            exercise,
            files,
            exercise_root / "final",
            timeout_s=args.test_timeout_s,
        )
        return {
            "item_id": exercise.slug,
            "passed": bool(verdict["passed"]),
            "error_category": verdict.get("error_category", ""),
            "failure": failure[:300],
            "llm_calls": llm.calls,
            "test_runs": tools.runs,
            "prompt_tokens": llm.prompt_tokens,
            "completion_tokens": llm.completion_tokens,
            "cost_usd": llm.cost,
            "elapsed_s": time.monotonic() - started,
        }

    items = []
    wall_start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="polyglot_") as tmp:
        workroot = Path(tmp)
        # Exercises are independent, the model endpoint takes concurrency, and
        # a serial pass over twelve exercises at this provider's latency costs
        # more wall-clock than a whole PPO training run in the other domain.
        # The candidate harness itself stays single-threaded: one thread per
        # exercise, each with its own llm and tools instance.
        workers = max(1, min(args.workers, len(exercises)))
        if workers == 1:
            items = [evaluate(exercise, workroot) for exercise in exercises]
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as pool:
                items = list(pool.map(
                    lambda exercise: evaluate(exercise, workroot), exercises
                ))
    items.sort(key=lambda item: item["item_id"])

    passed = sum(1 for item in items if item["passed"])
    report = {
        "split": args.split,
        "model": args.model,
        "items": items,
        "passed": passed,
        "total": len(items),
        "pass_rate": passed / len(items) if items else 0.0,
        "llm_calls": sum(item.get("llm_calls", 0) for item in items),
        "test_runs": sum(item.get("test_runs", 0) for item in items),
        "cost_usd": sum(item.get("cost_usd", 0.0) for item in items),
        # Summed per-exercise time is model-seconds bought; wall time is what
        # the run actually waited. With workers > 1 they differ by a lot, and
        # confusing them makes a parallel run look catastrophically slow.
        "solver_seconds": sum(item.get("elapsed_s", 0.0) for item in items),
        "elapsed_s": time.monotonic() - wall_start,
        "workers": args.workers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
