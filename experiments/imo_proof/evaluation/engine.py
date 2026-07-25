"""IMO dataset, admission, candidate runtime, judge, and score aggregation."""

from __future__ import annotations

import ast
import csv
import json
import os
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Mapping, Protocol, runtime_checkable

from evoharness.evocore import LLMClient, LLMMessage, LLMStopReason

from ..protocol import BenchmarkSpec, CandidateSpec, ModelSpec, SolverBudget
from .contract import (
    CandidateEvaluation,
    EvaluationUnavailable,
    ModelUsage,
    ProblemResult,
    ProofResult,
    normalize_proof_result,
)


_TRACE_LOCK = threading.Lock()


def _trace(block: str) -> None:
    """Print a multi-line trace block atomically across worker threads."""

    with _TRACE_LOCK:
        print(f"\n{'-' * 72}\n{block}", flush=True)


_BANNED_IMPORT_ROOTS = {
    "aiohttp",
    "concurrent",
    "ctypes",
    "glob",
    "httpx",
    "importlib",
    "inspect",
    "multiprocessing",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}
_BANNED_CALLS = {"__import__", "compile", "eval", "exec", "open"}
_BANNED_ATTRIBUTES = {
    "glob",
    "listdir",
    "open",
    "read_bytes",
    "read_text",
    "rglob",
    "walk",
}


@dataclass(frozen=True)
class AdmissionIssue:
    code: str
    message: str
    path: str | None = None
    line: int | None = None


@dataclass(frozen=True)
class AdmissionReport:
    issues: tuple[AdmissionIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues


class AdmissionGate:
    """Authoritative deterministic checks for a materialized candidate."""

    def __init__(self, candidate: CandidateSpec):
        self.candidate = candidate

    def check(self, root: Path) -> AdmissionReport:
        root = Path(root).resolve()
        issues: list[AdmissionIssue] = []
        if not root.is_dir():
            return AdmissionReport(
                (
                    AdmissionIssue(
                        "workspace-missing",
                        "candidate workspace does not exist",
                    ),
                )
            )

        files = {
            path.relative_to(root).as_posix(): path
            for path in root.rglob("*")
            if path.is_file()
            and ".git" not in path.parts
            and "__pycache__" not in path.parts
        }
        allowed = set(self.candidate.mutable_files)
        for rel, path in sorted(files.items()):
            if path.is_symlink():
                issues.append(
                    AdmissionIssue(
                        "symlink-forbidden",
                        "candidate files cannot be symlinks",
                        rel,
                    )
                )
                continue
            if rel not in allowed:
                issues.append(
                    AdmissionIssue(
                        "file-outside-allowlist",
                        "candidate modified or created a forbidden file",
                        rel,
                    )
                )
            if path.stat().st_size > self.candidate.max_file_bytes:
                issues.append(
                    AdmissionIssue(
                        "file-too-large",
                        "candidate file exceeds the frozen size limit",
                        rel,
                    )
                )

        for rel in sorted(allowed - set(files)):
            issues.append(
                AdmissionIssue(
                    "required-file-missing",
                    "candidate file is missing",
                    rel,
                )
            )

        module_name, function_name = self.candidate.entrypoint.split(":", 1)
        entry_rel = module_name.replace(".", "/") + ".py"
        for rel, path in sorted(files.items()):
            if path.suffix != ".py" or rel not in allowed:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            except (UnicodeDecodeError, SyntaxError) as exc:
                issues.append(
                    AdmissionIssue(
                        "python-invalid",
                        str(exc),
                        rel,
                        getattr(exc, "lineno", None),
                    )
                )
                continue
            issues.extend(self._scan_imports(tree, rel))
            issues.extend(self._scan_calls(tree, rel))
            if rel == entry_rel and not self._has_entrypoint(tree, function_name):
                issues.append(
                    AdmissionIssue(
                        "entrypoint-missing",
                        f"expected function {function_name}(problem, llm)",
                        rel,
                    )
                )

        return AdmissionReport(tuple(issues))

    @staticmethod
    def _scan_imports(tree: ast.AST, rel: str) -> list[AdmissionIssue]:
        issues = []
        for node in ast.walk(tree):
            roots: list[str] = []
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = [node.module.split(".", 1)[0]]
            for root in roots:
                if root in _BANNED_IMPORT_ROOTS:
                    issues.append(
                        AdmissionIssue(
                            "network-or-process-import",
                            f"import {root!r} is forbidden in benchmark candidates",
                            rel,
                            getattr(node, "lineno", None),
                        )
                    )
        return issues

    @staticmethod
    def _scan_calls(tree: ast.AST, rel: str) -> list[AdmissionIssue]:
        issues = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            forbidden = None
            if isinstance(node.func, ast.Name) and node.func.id in _BANNED_CALLS:
                forbidden = node.func.id
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _BANNED_ATTRIBUTES
            ):
                forbidden = node.func.attr
            if forbidden is not None:
                issues.append(
                    AdmissionIssue(
                        "filesystem-or-dynamic-code-call",
                        f"call {forbidden!r} is forbidden in benchmark candidates",
                        rel,
                        getattr(node, "lineno", None),
                    )
                )
        return issues

    @staticmethod
    def _has_entrypoint(tree: ast.Module, name: str) -> bool:
        for node in tree.body:
            if (
                not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                or node.name != name
            ):
                continue
            positional = [*node.args.posonlyargs, *node.args.args]
            return (
                len(positional) >= 2
                and positional[0].arg == "problem"
                and positional[1].arg == "llm"
            )
        return False


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetedLLM:
    client: LLMClient
    model: ModelSpec
    budget: SolverBudget
    system_prompt: str = "Produce rigorous mathematical work."
    label: str = ""

    def __post_init__(self) -> None:
        if self.client.temperature != self.model.temperature:
            raise ValueError("solver client temperature does not match BenchmarkSpec")
        if self.client.max_tokens != self.model.max_output_tokens:
            raise ValueError("solver client max_tokens does not match BenchmarkSpec")
        self._usage = ModelUsage()
        self._started_at = monotonic()

    @property
    def usage(self) -> ModelUsage:
        return self._usage

    def complete(self, stage: str, prompt: str) -> str:
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("stage must be non-empty")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty")
        self._admit_next_call()
        remaining = self.budget.timeout_s_per_problem - (
            monotonic() - self._started_at
        )
        if remaining <= 0:
            raise BudgetExceeded("solver timeout exceeded")
        # One call may not wait out the whole problem: the budget bounds the
        # problem, request_timeout_s bounds a single stuck connection.
        remaining = min(remaining, self.budget.request_timeout_s)

        response = self.client.query_messages(
            (
                LLMMessage("system", self.system_prompt),
                LLMMessage("user", f"Stage: {stage}\n\n{prompt}"),
            ),
            self.model.name,
            timeout_s=remaining,
        )

        print("Response", response)
        next_usage = self._usage + ModelUsage(
            calls=1,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            cost_usd=response.cost,
        )
        self._check_usage(next_usage)
        self._usage = next_usage
        if response.stop_reason is not LLMStopReason.COMPLETED:
            raise BudgetExceeded(
                f"solver {stage} did not complete within the output budget: "
                f"{response.stop_reason.value}"
            )
        if not response.text.strip():
            raise RuntimeError("solver model returned empty text")
        text = response.text.strip()
        if os.environ.get("IMO_TRACE"):
            tag = self.label or "solver"
            _trace(
                f"[{tag}] stage={stage} "
                f"call={next_usage.calls}/{self.budget.max_calls_per_problem} "
                f"completion_tokens={response.completion_tokens}\n{text}"
            )
        return text

    def _admit_next_call(self) -> None:
        if self._usage.calls >= self.budget.max_calls_per_problem:
            raise BudgetExceeded("solver call limit exceeded")
        self._check_usage(self._usage)

    def _check_usage(self, usage: ModelUsage) -> None:
        if usage.prompt_tokens > self.budget.max_prompt_tokens_per_problem:
            raise BudgetExceeded("solver prompt-token limit exceeded")
        if usage.completion_tokens > self.budget.max_completion_tokens_per_problem:
            raise BudgetExceeded("solver completion-token limit exceeded")
        cap = self.budget.max_cost_usd_per_problem
        if cap is not None and usage.cost_usd > cap:
            raise BudgetExceeded("solver cost limit exceeded")


class CandidateExecutionError(RuntimeError):
    pass


class ModelBrokerError(RuntimeError):
    pass


class SubprocessCandidateExecutor:
    """Execute candidate code in an isolated process with a model-broker RPC."""

    def __init__(
        self,
        entrypoint: str,
        timeout_s: float,
        max_protocol_line_bytes: int = 2_000_000,
    ):
        self.entrypoint = entrypoint
        self.timeout_s = timeout_s
        self.max_protocol_line_bytes = max_protocol_line_bytes

    def execute(
        self,
        root: Path,
        problem: str,
        llm: BudgetedLLM,
    ) -> ProofResult:
        worker = Path(__file__).with_name("worker.py")

        with tempfile.TemporaryFile(mode="w+t") as stderr:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    str(worker),
                    "candidate",
                    str(Path(root).resolve()),
                    self.entrypoint,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                text=True,
                start_new_session=True,
                env={
                    "PATH": os.environ.get("PATH", os.defpath),
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "HOME": str(Path(root).resolve()),
                    "TMPDIR": str(Path(root).resolve()),
                    "NO_PROXY": "",
                },
            )
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(json.dumps({"problem": problem}) + "\n")
            proc.stdin.flush()
            selector = selectors.DefaultSelector()
            selector.register(proc.stdout, selectors.EVENT_READ)
            deadline = monotonic() + self.timeout_s
            budget_error: BudgetExceeded | None = None
            try:
                while True:
                    remaining = deadline - monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise CandidateExecutionError("candidate timeout")
                    line = proc.stdout.readline(self.max_protocol_line_bytes + 1)
                    if len(line.encode()) > self.max_protocol_line_bytes:
                        raise CandidateExecutionError(
                            "candidate protocol line too large"
                        )
                    if not line:
                        code = proc.wait(timeout=1)
                        stderr.seek(0)
                        detail = stderr.read().strip()
                        raise CandidateExecutionError(
                            f"candidate exited with {code}: {detail[-2000:]}"
                        )
                    try:
                        message = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise CandidateExecutionError(
                            "candidate emitted invalid protocol JSON"
                        ) from exc
                    kind = message.get("type")
                    if kind == "llm_request":
                        try:
                            text = llm.complete(
                                message.get("stage"),
                                message.get("prompt"),
                            )
                            response = {"type": "llm_response", "text": text}
                        except BudgetExceeded as exc:
                            budget_error = exc
                            response = {
                                "type": "error",
                                "message": f"{type(exc).__name__}: {exc}",
                            }
                        except Exception as exc:
                            raise ModelBrokerError(
                                f"solver model broker failed: {exc}"
                            ) from exc
                        proc.stdin.write(json.dumps(response) + "\n")
                        proc.stdin.flush()
                        continue
                    if kind == "worker_error":
                        if budget_error is not None:
                            raise CandidateExecutionError(
                                str(budget_error)
                            ) from budget_error
                        raise CandidateExecutionError(
                            str(message.get("message", "candidate worker error"))
                        )
                    if kind == "result":
                        result = normalize_proof_result(message)
                        code = proc.wait(timeout=2)
                        if code != 0:
                            raise CandidateExecutionError(
                                f"candidate exited with {code} after result"
                            )
                        return result
                    raise CandidateExecutionError(
                        f"unknown candidate protocol message: {kind!r}"
                    )
            finally:
                selector.close()
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()


@dataclass(frozen=True)
class ProblemRecord:
    problem_id: str
    problem: str
    solution: str
    grading_guidelines: str
    category: str
    level: str


class ProofDataset:
    def __init__(self, records: Mapping[str, ProblemRecord]):
        self.records = dict(records)
        if not self.records:
            raise ValueError("proof dataset cannot be empty")

    @classmethod
    def load(cls, root: Path, spec: BenchmarkSpec) -> "ProofDataset":
        path = Path(root) / spec.dataset.path
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        records = {}
        for row in rows:
            record = ProblemRecord(
                problem_id=row[spec.dataset.id_column].strip(),
                problem=row[spec.dataset.problem_column].strip(),
                solution=row["Solution"].strip(),
                grading_guidelines=row["Grading guidelines"].strip(),
                category=row.get("Category", "").strip(),
                level=row.get("Level", "").strip(),
            )
            if record.problem_id in records:
                raise ValueError(f"duplicate problem ID {record.problem_id}")
            records[record.problem_id] = record
        return cls(records)

    def select(self, ids: tuple[str, ...]) -> tuple[ProblemRecord, ...]:
        missing = [problem_id for problem_id in ids if problem_id not in self.records]
        if missing:
            raise ValueError(f"unknown problem IDs: {missing}")
        return tuple(self.records[problem_id] for problem_id in ids)


@dataclass(frozen=True)
class GraderOutcome:
    label: str
    usage: ModelUsage = ModelUsage()
    raw_response: str = ""


@runtime_checkable
class ProofGrader(Protocol):
    def grade(self, record: ProblemRecord, proof: str) -> GraderOutcome: ...


def load_frozen_grader_prompt(root: Path, spec: BenchmarkSpec) -> str:
    prompt = (Path(root) / spec.grader.prompt_path).read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError("frozen grader prompt is empty")
    return prompt


class LLMProofGrader:
    _POINTS_RE = re.compile(
        r"<points>\s*(\d+)\s+out\s+of\s+7\s*</points>",
        re.IGNORECASE,
    )

    def __init__(self, client: LLMClient, spec: BenchmarkSpec, prompt: str):
        if client.temperature != spec.grader.model.temperature:
            raise ValueError("grader client temperature does not match BenchmarkSpec")
        if client.max_tokens != spec.grader.model.max_output_tokens:
            raise ValueError("grader client max_tokens does not match BenchmarkSpec")
        self.client = client
        self.spec = spec
        self.prompt = prompt
        self.points_to_label = {
            points: label for label, points in spec.scoring.label_points
        }

    def grade(self, record: ProblemRecord, proof: str) -> GraderOutcome:
        instruction = self.prompt.format(
            problem_statement=record.problem,
            solution=record.solution,
            grading_guidelines=record.grading_guidelines,
            student_answer=proof,
        )
        usage = ModelUsage()
        last_error = "grader response lacks a valid <points> block"
        for _ in range(3):
            response = self.client.query_messages(
                (LLMMessage("user", instruction),),
                self.spec.grader.model.name,
            )
            usage += ModelUsage(
                calls=1,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                cost_usd=response.cost,
            )
            if response.stop_reason is not LLMStopReason.COMPLETED:
                raise RuntimeError(
                    "grader model did not complete: "
                    f"{response.stop_reason.value}"
                )
            match = self._POINTS_RE.search(response.text)
            if match is None:
                continue
            points = int(match.group(1))
            label = self.points_to_label.get(points)
            if label is None:
                last_error = f"grader returned unsupported score {points}"
                continue
            return GraderOutcome(
                label=label,
                usage=usage,
                raw_response=response.text,
            )
        raise ValueError(last_error)


class IMOEvaluator:
    """Authoritative task evaluator; contains no EvoHarness Grade logic."""

    def __init__(
        self,
        *,
        project_root: Path,
        spec: BenchmarkSpec,
        solver_client: LLMClient,
        grader: ProofGrader,
        executor: SubprocessCandidateExecutor | None = None,
        max_workers: int = 1,
    ):
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self.project_root = Path(project_root).resolve()
        self.spec = spec
        self.spec.verify_workspace(self.project_root)
        self.dataset = ProofDataset.load(self.project_root, spec)
        self.solver_client = solver_client
        self.grader = grader
        self.max_workers = max_workers
        self.admission = AdmissionGate(spec.candidate)
        self.executor = executor or SubprocessCandidateExecutor(
            spec.candidate.entrypoint,
            spec.solver_budget.timeout_s_per_problem,
        )

    def evaluate_directory(
        self,
        *,
        candidate_id: str,
        candidate_root: Path,
        split: str,
        output_dir: Path | None = None,
    ) -> CandidateEvaluation:
        candidate_root = Path(candidate_root).resolve()
        ids = self.spec.splits.ids(split)
        admission = self.admission.check(candidate_root)
        if not admission.ok:
            evaluation = CandidateEvaluation(
                candidate_id=candidate_id,
                split=split,
                admitted=False,
                admission_issues=tuple(
                    f"{issue.code}:{issue.path or '-'}:{issue.message}"
                    for issue in admission.issues
                ),
            )
        else:
            evaluation = CandidateEvaluation(
                candidate_id=candidate_id,
                split=split,
                admitted=True,
                problems=self._evaluate_problems(candidate_root, ids),
            )
        if output_dir is not None:
            evaluation.write(Path(output_dir) / "evaluation.json")
        return evaluation

    def _evaluate_problems(
        self,
        candidate_root: Path,
        ids: tuple[str, ...],
    ) -> tuple[ProblemResult, ...]:
        records = self.dataset.select(ids)


        if self.max_workers == 1:
            return tuple(
                self._evaluate_problem(candidate_root, record)
                for record in records
            )
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            # map preserves input order; the first EvaluationUnavailable
            # re-raises on iteration, aborting the whole evaluation.
            return tuple(
                pool.map(
                    lambda record: self._evaluate_problem(candidate_root, record),
                    records,
                )
            )

    def _evaluate_problem(
        self,
        candidate_root: Path,
        record: ProblemRecord,
    ) -> ProblemResult:

        started = monotonic()
        _trace(f"[{record.problem_id}] evaluating: {record.problem[:100]}...")
        llm = BudgetedLLM(
            client=self.solver_client,
            model=self.spec.solver,
            budget=self.spec.solver_budget,
            label=record.problem_id,
        )
        proof = ""
        grader_usage = ModelUsage()
        failure = None
        label = "incorrect"
        grader_critique = ""

        try:
            outcome = self.executor.execute(candidate_root, record.problem, llm)
            proof = outcome.proof
            grade = self.grader.grade(record, proof)
            if grade.label not in self.spec.scoring.points:
                raise ValueError(f"grader returned unknown label {grade.label!r}")
            label = grade.label
            grader_usage = grade.usage
            grader_critique = grade.raw_response
        except CandidateExecutionError as exc:
            failure = f"{type(exc).__name__}: {exc}"
        except ModelBrokerError as exc:
            raise EvaluationUnavailable(str(exc)) from exc
        except Exception as exc:
            raise EvaluationUnavailable(
                f"grader or evaluator failed for {record.problem_id}: {exc}"
            ) from exc

        result = ProblemResult(
            problem_id=record.problem_id,
            label=label,
            points=self.spec.scoring.points[label],
            max_points=self.spec.scoring.max_points,
            proof=proof,
            solver_usage=llm.usage,
            grader_usage=grader_usage,
            elapsed_s=max(0.0, monotonic() - started),
            failure=failure,
            grader_critique=grader_critique
        )
        if os.environ.get("IMO_TRACE"):
            verdict = failure or f"{label} ({result.points}/{result.max_points})"
            _trace(f"[{record.problem_id}] done in {result.elapsed_s:.1f}s -> {verdict}")
        return result


def make_live_evaluator(
    *,
    project_root: Path,
    spec: BenchmarkSpec,
    solver_client: LLMClient,
    grader_client: LLMClient,
    max_workers: int = 1,
) -> IMOEvaluator:
    return IMOEvaluator(
        project_root=project_root,
        spec=spec,
        solver_client=solver_client,
        grader=LLMProofGrader(
            grader_client,
            spec,
            load_frozen_grader_prompt(project_root, spec),
        ),
        max_workers=max_workers,
    )


__all__ = [
    "AdmissionGate",
    "AdmissionIssue",
    "AdmissionReport",
    "BudgetExceeded",
    "BudgetedLLM",
    "CandidateExecutionError",
    "GraderOutcome",
    "IMOEvaluator",
    "LLMProofGrader",
    "ModelBrokerError",
    "ProblemRecord",
    "ProofDataset",
    "ProofGrader",
    "SubprocessCandidateExecutor",
    "load_frozen_grader_prompt",
    "make_live_evaluator",
]
