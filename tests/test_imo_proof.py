"""Acceptance tests for the self-contained IMO proof experiment."""

from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

import pytest

from evoharness import ScorableTask, WorkspaceGradeFnGrader
from evoharness.evocore.workspace import GitWorkspace
from evoharness.evoserve import GradeContext, InfraError
from evoharness.evocore import (
    Candidate,
    LLMClient,
    LLMProtocolError,
    LLMResponse,
    LLMStopReason,
    LLMToolCall,
)
from experiments.imo_proof.evaluation.contract import (
    CandidateEvaluation,
    EvaluationProtocolError,
    EvaluationRouter,
    EvaluationUnavailable,
    ModelUsage,
    ProblemResult,
)
from experiments.imo_proof.evaluation.engine import (
    AdmissionGate,
    BudgetExceeded,
    BudgetedLLM,
    CandidateExecutionError,
    GraderOutcome,
    IMOEvaluator,
    LLMProofGrader,
    ProblemRecord,
    SubprocessCandidateExecutor,
    load_frozen_grader_prompt,
)
from experiments.imo_proof.evaluation.service import (
    PROTOCOL_VERSION,
    EvaluationService,
    HttpEvaluationBackend,
    HttpReply,
    idempotency_key,
    serve,
    workspace_tree_hash,
)
from experiments.imo_proof.evaluation.worker import LocalProcessBackend
from experiments.imo_proof.evolution import make_task, run_experiment
from experiments.imo_proof.grade import make_grade_func
from experiments.imo_proof.protocol import BenchmarkSpec, load_default_spec
from experiments.imo_proof.result import RunManifest
from experiments.imo_proof.seed import materialize_seed, seed_sha256
from experiments.imo_proof.transport import SpecOpenAITransport


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _OfflineSolverTransport:
    def __call__(self, *, messages, model, **kwargs):
        prompt = messages[-1].content
        text = (
            "<verdict>PASS</verdict>"
            if prompt.startswith("Stage: review")
            else "A deterministic complete proof for offline evaluation."
        )
        return LLMResponse(
            text=text,
            model=model,
            prompt_tokens=3,
            completion_tokens=5,
        )


class _AlwaysCorrectGrader:
    def grade(self, record, proof):
        return GraderOutcome("correct", ModelUsage(calls=1))


class _OptimizerTransport:
    def __call__(self, *, messages, model, tools, **kwargs):
        if tools and messages[-1].role != "tool":
            return LLMResponse(
                text="",
                model=model,
                stop_reason=LLMStopReason.TOOL_CALLS,
                tool_calls=(
                    LLMToolCall(
                        call_id="edit-policy",
                        name="workspace_write",
                        arguments={
                            "path": "policy.py",
                            "content": (
                                "def needs_revision(review):\n"
                                "    return '<verdict>PASS</verdict>' not in "
                                "review.upper()\n"
                                "\n# offline mutation\n"
                            ),
                        },
                    ),
                ),
            )
        return LLMResponse(
            "TITLE: offline mutation\nSUMMARY: preserve solver behavior",
            model,
        )


def _solver_client(spec):
    return LLMClient(
        spec.solver.temperature,
        spec.solver.max_output_tokens,
        transport=_OfflineSolverTransport(),
        sleep=lambda _: None,
    )


def _sample_evaluation(candidate_id: str = "candidate") -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate_id=candidate_id,
        split="train",
        admitted=True,
        problems=(
            ProblemResult(
                problem_id="problem-1",
                label="correct",
                points=7,
                max_points=7,
                proof="proof",
                solver_usage=ModelUsage(calls=2, prompt_tokens=3),
                grader_usage=ModelUsage(calls=1, completion_tokens=4),
                elapsed_s=0.1,
            ),
        ),
    )


def test_default_spec_verifies_local_assets_and_frozen_splits():
    spec = load_default_spec()

    spec.verify_workspace(PROJECT_ROOT)

    assert spec.dataset.path.startswith("experiments/imo_proof/")
    assert spec.grader.prompt_path.startswith("experiments/imo_proof/")
    # Splits partition exactly the dataset (sizes are dev-tunable; the
    # invariant is full, disjoint coverage of expected_rows).
    all_ids = spec.splits.train + spec.splits.validation + spec.splits.test
    assert len(all_ids) == spec.dataset.expected_rows == 60
    assert len(set(all_ids)) == len(all_ids)
    assert spec.splits.train and spec.splits.validation and spec.splits.test
    assert len(spec.fingerprint) == 64


def test_spec_roundtrip_preserves_stable_fingerprint(tmp_path):
    original = load_default_spec()
    path = tmp_path / "spec.json"

    original.write(path)
    restored = BenchmarkSpec.load(path)

    assert restored == original
    assert restored.fingerprint == original.fingerprint


def test_spec_rejects_overlap_and_asset_drift():
    value = load_default_spec().to_dict()
    value["splits"]["test"][0] = value["splits"]["train"][0]
    with pytest.raises(ValueError, match="unique and disjoint"):
        BenchmarkSpec.from_dict(value)

    spec = load_default_spec()
    changed = replace(spec, dataset=replace(spec.dataset, sha256="0" * 64))
    with pytest.raises(ValueError, match="dataset SHA-256 mismatch"):
        changed.verify_workspace(PROJECT_ROOT)


def test_candidate_evaluation_is_the_single_serializable_result(tmp_path):
    original = _sample_evaluation()
    path = tmp_path / "evaluation.json"

    original.write(path)
    restored = CandidateEvaluation.read(path)

    assert restored == original
    assert restored.points_percentage == 1.0
    assert restored.solver_usage.calls == 2

    invalid = original.to_dict()
    invalid["admitted"] = "true"
    with pytest.raises(ValueError, match="admitted must be a boolean"):
        CandidateEvaluation.from_dict(invalid)


def test_evaluation_router_keeps_profiles_on_separate_backends(tmp_path):
    calls = []

    class _Backend:
        def __init__(self, expected):
            self.expected = expected

        def evaluate_directory(self, **kwargs):
            calls.append(self.expected)
            assert kwargs["split"] == self.expected
            return CandidateEvaluation(
                candidate_id=kwargs["candidate_id"],
                split=self.expected,
                admitted=False,
                admission_issues=("rejected",),
            )

    router = EvaluationRouter(
        {name: _Backend(name) for name in ("train", "validation", "test")}
    )

    result = router.evaluate_directory(
        candidate_id="candidate",
        candidate_root=tmp_path,
        split="validation",
    )

    assert result.split == "validation"
    assert calls == ["validation"]


def test_grade_adapter_never_scores_protocol_failure_as_candidate_failure(tmp_path):
    class _BrokenBackend:
        def evaluate_directory(self, **kwargs):
            raise EvaluationProtocolError("wrong schema")

    grade_func = make_grade_func(_BrokenBackend())

    with pytest.raises(InfraError, match="evaluation protocol error"):
        grade_func(
            tmp_path,
            GradeContext(candidate_id="candidate", workdir=tmp_path),
        )


def test_seed_directory_is_admitted_and_hash_is_stable(tmp_path):
    spec = load_default_spec()
    root = materialize_seed(spec.candidate, tmp_path / "seed")

    assert AdmissionGate(spec.candidate).check(root).ok
    assert seed_sha256(spec.candidate) == seed_sha256(spec.candidate)


def test_seed_review_policy_requires_one_structured_verdict():
    import sys

    seed_root = str(PROJECT_ROOT / "experiments/imo_proof/seed_agent")
    sys.path.insert(0, seed_root)
    try:
        from policy import needs_revision, review_verdict
    finally:
        sys.path.remove(seed_root)
        sys.modules.pop("policy", None)

    assert review_verdict("<verdict>PASS</verdict>") == "PASS"
    assert not needs_revision("<verdict>pass</verdict>")
    assert needs_revision("PASS")
    assert needs_revision("<verdict>PASS</verdict><verdict>PASS</verdict>")
    assert needs_revision("<verdict>unknown</verdict>")


def test_admission_rejects_extra_file_and_forbidden_access(tmp_path):
    spec = load_default_spec()
    root = materialize_seed(spec.candidate, tmp_path / "candidate")
    (root / "extra.py").write_text("x = 1\n")
    (root / "solver.py").write_text(
        "import requests\n\n"
        "def solve(problem, llm):\n"
        "    return open('/tmp/proofbench.csv').read()\n"
    )

    codes = {issue.code for issue in AdmissionGate(spec.candidate).check(root).issues}

    assert codes == {
        "file-outside-allowlist",
        "filesystem-or-dynamic-code-call",
        "network-or-process-import",
    }


def test_budgeted_llm_enforces_call_limit():
    spec = load_default_spec()
    budgeted = BudgetedLLM(
        _solver_client(spec),
        spec.solver,
        replace(spec.solver_budget, max_calls_per_problem=1),
    )

    assert budgeted.complete("solve", "problem")
    with pytest.raises(BudgetExceeded, match="call limit"):
        budgeted.complete("review", "proof")


def test_budgeted_llm_classifies_truncated_solver_output_as_budget_failure():
    spec = load_default_spec()

    def truncated_transport(**kwargs):
        return LLMResponse(
            text="unfinished proof",
            model=kwargs["model"],
            stop_reason=LLMStopReason.MAX_TOKENS,
            completion_tokens=5,
        )

    budgeted = BudgetedLLM(
        LLMClient(
            spec.solver.temperature,
            spec.solver.max_output_tokens,
            transport=truncated_transport,
        ),
        spec.solver,
        spec.solver_budget,
    )

    with pytest.raises(
        BudgetExceeded,
        match="solver solve did not complete within the output budget: max_tokens",
    ):
        budgeted.complete("solve", "problem")
    assert budgeted.usage.completion_tokens == 5


def test_candidate_executor_turns_solver_truncation_into_problem_failure(tmp_path):
    spec = load_default_spec()
    root = materialize_seed(spec.candidate, tmp_path / "candidate")

    def truncated_transport(**kwargs):
        return LLMResponse(
            text="unfinished proof",
            model=kwargs["model"],
            stop_reason=LLMStopReason.MAX_TOKENS,
            completion_tokens=5,
        )

    budgeted = BudgetedLLM(
        LLMClient(
            spec.solver.temperature,
            spec.solver.max_output_tokens,
            transport=truncated_transport,
        ),
        spec.solver,
        spec.solver_budget,
    )

    with pytest.raises(CandidateExecutionError, match="output budget"):
        SubprocessCandidateExecutor(
            spec.candidate.entrypoint,
            timeout_s=5,
        ).execute(root, "Prove P.", budgeted)


def test_proof_grader_retries_malformed_output_and_accounts_usage():
    spec = load_default_spec()
    responses = iter(("missing score", "<points>7 out of 7</points>"))

    def grader_transport(**kwargs):
        return LLMResponse(
            text=next(responses),
            model=kwargs["model"],
            prompt_tokens=3,
            completion_tokens=5,
        )

    grader = LLMProofGrader(
        LLMClient(
            spec.grader.model.temperature,
            spec.grader.model.max_output_tokens,
            transport=grader_transport,
        ),
        spec,
        "{student_answer}",
    )
    record = ProblemRecord("p", "problem", "solution", "guidelines", "", "")

    outcome = grader.grade(record, "proof")

    assert outcome.label == "correct"
    assert outcome.usage == ModelUsage(
        calls=2,
        prompt_tokens=6,
        completion_tokens=10,
    )


def test_subprocess_executor_runs_seed_through_model_broker(tmp_path):
    spec = load_default_spec()
    root = materialize_seed(spec.candidate, tmp_path / "candidate")
    budgeted = BudgetedLLM(_solver_client(spec), spec.solver, spec.solver_budget)

    result = SubprocessCandidateExecutor(
        spec.candidate.entrypoint,
        timeout_s=5,
    ).execute(root, "Prove P.", budgeted)

    assert result.proof.startswith("A deterministic complete proof")
    assert budgeted.usage.calls == 2


def test_local_process_backend_reads_worker_result(tmp_path):
    spec = load_default_spec()
    candidate_root = materialize_seed(spec.candidate, tmp_path / "candidate")

    def fake_runner(command, **kwargs):
        result_dir = Path(command[command.index("--result-dir") + 1])
        candidate_id = command[command.index("--candidate-id") + 1]
        _sample_evaluation(candidate_id).write(result_dir / "evaluation.json")
        return subprocess.CompletedProcess(command, 0, "", "")

    backend = LocalProcessBackend(
        project_root=PROJECT_ROOT,
        spec_path=PROJECT_ROOT / "experiments/imo_proof/benchmark.v1.json",
        runner=fake_runner,
    )

    result = backend.evaluate_directory(
        candidate_id="worker-candidate",
        candidate_root=candidate_root,
        split="train",
        output_dir=tmp_path / "result",
    )

    assert result.candidate_id == "worker-candidate"
    assert result.points_percentage == 1.0


def test_http_backend_serializes_full_workspace_and_reuses_domain_result(tmp_path):
    spec = load_default_spec()
    candidate_root = materialize_seed(spec.candidate, tmp_path / "candidate")
    seen = []

    class _Backend:
        def evaluate_directory(self, **kwargs):
            seen.append(
                {
                    path.relative_to(kwargs["candidate_root"]).as_posix()
                    for path in kwargs["candidate_root"].rglob("*")
                    if path.is_file()
                }
            )
            return _sample_evaluation(kwargs["candidate_id"])

    service = EvaluationService(
        _Backend(),
        spec,
        profile="train",
        base_dir=tmp_path,
    )

    def memory_transport(method, url, headers, payload, timeout_s):
        path = url.removeprefix("memory://evaluation")
        if method == "GET" and path == "/v2/meta":
            return HttpReply(200, service.meta())
        if method == "POST" and path == "/v2/evaluations":
            return HttpReply(202, {"job_id": service.submit(payload)})
        if method == "GET" and path.startswith("/v2/evaluations/"):
            job_id = path.rsplit("/", 1)[-1]
            return HttpReply(200, service.wait(job_id))
        return HttpReply(404, {"error": "unknown route"})

    backend = HttpEvaluationBackend(
        "memory://evaluation",
        spec,
        transport=memory_transport,
        sleep=lambda delay: time.sleep(min(delay, 0.001)),
    )
    try:
        result = backend.evaluate_directory(
            candidate_id="http-candidate",
            candidate_root=candidate_root,
            split="train",
            output_dir=tmp_path / "http-result",
        )
        cached = backend.evaluate_directory(
            candidate_id="same-content-new-id",
            candidate_root=candidate_root,
            split="train",
        )
    finally:
        service.close()

    assert result == _sample_evaluation("http-candidate")
    assert cached == _sample_evaluation("same-content-new-id")
    assert seen == [set(spec.candidate.mutable_files)]
    assert CandidateEvaluation.read(
        tmp_path / "http-result/evaluation.json"
    ) == result


def test_service_exposes_backend_contract_violation_as_protocol_error(tmp_path):
    spec = load_default_spec()
    candidate_root = materialize_seed(spec.candidate, tmp_path / "candidate")

    class _WrongSplitBackend:
        def evaluate_directory(self, **kwargs):
            return CandidateEvaluation(
                candidate_id=kwargs["candidate_id"],
                split="validation",
                admitted=True,
            )

    service = EvaluationService(
        _WrongSplitBackend(),
        spec,
        profile="train",
        base_dir=tmp_path,
    )
    files = {
        path.relative_to(candidate_root).as_posix(): path.read_text()
        for path in candidate_root.rglob("*")
        if path.is_file()
    }
    tree_hash = workspace_tree_hash(files)
    try:
        job_id = service.submit(
            {
                "protocol_version": PROTOCOL_VERSION,
                "candidate_id": "candidate",
                "protocol_fingerprint": spec.fingerprint,
                "tree_hash": tree_hash,
                "files": files,
                "idempotency_key": idempotency_key(
                    tree_hash, spec.fingerprint, "train"
                ),
            }
        )
        result = service.wait(job_id)
    finally:
        service.close()

    assert result["status"] == "protocol_error"
    assert "mismatched candidate or split" in result["error"]


def test_imo_evaluation_http_shell_roundtrip(tmp_path):
    spec = load_default_spec()
    candidate_root = materialize_seed(spec.candidate, tmp_path / "candidate")

    class _Backend:
        def evaluate_directory(self, **kwargs):
            return _sample_evaluation(kwargs["candidate_id"])

    service = EvaluationService(_Backend(), spec, base_dir=tmp_path)
    server = serve(service, port=0, token="test-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    backend = HttpEvaluationBackend(
        f"http://127.0.0.1:{server.server_address[1]}",
        spec,
        auth_token="test-token",
        poll_interval_s=0.001,
    )
    try:
        result = backend.evaluate_directory(
            candidate_id="real-http-candidate",
            candidate_root=candidate_root,
            split="train",
        )
    finally:
        server.shutdown()
        server.server_close()
        service.close()
        thread.join(timeout=2)

    assert result == _sample_evaluation("real-http-candidate")


def test_solver_provider_failure_is_infrastructure_not_zero_score(tmp_path):
    spec = load_default_spec()

    def broken_transport(**kwargs):
        raise ConnectionError("provider offline")

    evaluator = IMOEvaluator(
        project_root=PROJECT_ROOT,
        spec=spec,
        solver_client=LLMClient(
            spec.solver.temperature,
            spec.solver.max_output_tokens,
            transport=broken_transport,
            sleep=lambda _: None,
        ),
        grader=_AlwaysCorrectGrader(),
    )

    with pytest.raises(EvaluationUnavailable, match="model broker"):
        evaluator.evaluate_directory(
            candidate_id="seed",
            candidate_root=materialize_seed(
                spec.candidate,
                tmp_path / "candidate",
            ),
            split="test",
        )


def test_frozen_grader_prompt_is_loaded_from_experiment_asset():
    prompt = load_frozen_grader_prompt(PROJECT_ROOT, load_default_spec())

    assert "<points>7 out of 7</points>" in prompt
    assert "{student_answer}" in prompt


def test_imo_task_uses_generic_scorable_task_entry():
    spec = load_default_spec()
    evaluator = IMOEvaluator(
        project_root=PROJECT_ROOT,
        spec=spec,
        solver_client=_solver_client(spec),
        grader=_AlwaysCorrectGrader(),
    )

    task = make_task(spec, evaluator)

    assert isinstance(task, ScorableTask)
    assert isinstance(task.grader, WorkspaceGradeFnGrader)
    assert set(task.initial_workspace.texts()) == set(spec.candidate.mutable_files)
    assert task.initial_workspace.main_file == "solver.py"


def test_independent_experiment_runs_native_search_loop_offline(tmp_path):
    base = load_default_spec()
    spec = replace(
        base,
        optimizer_budget=replace(base.optimizer_budget, max_candidates=2),
    )
    evaluator = IMOEvaluator(
        project_root=PROJECT_ROOT,
        spec=spec,
        solver_client=_solver_client(spec),
        grader=_AlwaysCorrectGrader(),
    )
    optimizer = LLMClient(
        spec.optimizer.temperature,
        spec.optimizer.max_output_tokens,
        transport=_OptimizerTransport(),
        sleep=lambda _: None,
    )

    summary = run_experiment(
        spec=spec,
        evaluation_backend=evaluator,
        optimizer_client=optimizer,
        run_dir=tmp_path / "run",
        evolution_seed=0,
    )

    assert summary["experiment"] == spec.benchmark_id
    assert summary["test_points_percentage"] == 1.0
    assert summary["native_usage"]["total"]["calls"] > 0
    # num_islands=2 seeds one seed copy per island (seed_all_islands), so with
    # max_candidates=2 the count is 2 seeds + 1 offspring = 3.
    assert summary["evolution_metrics"]["candidate_count"] == 3
    assert (tmp_path / "run" / "experiment_manifest.json").is_file()
    manifest = RunManifest.load(tmp_path / "run" / "experiment_manifest.json")
    assert manifest.experiment_id == spec.benchmark_id
    assert not hasattr(manifest, "framework")
    assert manifest.actual["experience_mode"] == "lessons+scratchpad"


def test_run_experiment_experience_mode_arm_is_recorded(tmp_path):
    base = load_default_spec()
    spec = replace(
        base,
        optimizer_budget=replace(base.optimizer_budget, max_candidates=2),
    )
    evaluator = IMOEvaluator(
        project_root=PROJECT_ROOT,
        spec=spec,
        solver_client=_solver_client(spec),
        grader=_AlwaysCorrectGrader(),
    )
    optimizer = LLMClient(
        spec.optimizer.temperature,
        spec.optimizer.max_output_tokens,
        transport=_OptimizerTransport(),
        sleep=lambda _: None,
    )
    run_experiment(
        spec=spec,
        evaluation_backend=evaluator,
        optimizer_client=optimizer,
        run_dir=tmp_path / "run",
        evolution_seed=0,
        experience_mode="retrieval",
        lesson_directive=True,
        operator_bandit=True,
    )
    manifest = RunManifest.load(tmp_path / "run" / "experiment_manifest.json")
    assert manifest.actual["experience_mode"] == "retrieval"
    assert manifest.actual["lesson_directive"] is True
    assert manifest.actual["operator_bandit"] is True


def test_live_transport_freezes_thinking_and_computes_declared_cost(monkeypatch):
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "model": "m",
                    "choices": [
                        {
                            "message": {"content": "answer"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                }
            ).encode()

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    transport = SpecOpenAITransport(
        "https://example.test/v1",
        "secret",
        False,
        input_cost_per_million=2.0,
        output_cost_per_million=4.0,
    )
    client = LLMClient(0.0, 4096, transport=transport)

    response = client.query("system", "user", "openai/m")

    assert captured["payload"]["enable_thinking"] is False
    assert captured["payload"]["model"] == "m"
    assert response.cost == pytest.approx((100 * 2 + 20 * 4) / 1_000_000)


def test_minimax_transport_disables_and_removes_inline_thinking(monkeypatch):
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "model": "MiniMax-M3",
                    "choices": [
                        {
                            "message": {
                                "content": "<think>private</think>\nFINAL"
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                }
            ).encode()

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    transport = SpecOpenAITransport(
        "https://example.test/v1",
        "secret",
        False,
    )
    client = LLMClient(0.0, 4096, transport=transport)

    response = client.query("system", "user", "openai/MiniMax-M3")

    assert captured["payload"]["thinking"] == {"type": "disabled"}
    assert "enable_thinking" not in captured["payload"]
    assert response.text == "FINAL"


def test_transport_rejects_reasoning_without_final_answer(monkeypatch):
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "model": "MiniMax-M3",
                    "choices": [
                        {
                            "message": {
                                "content": "<think>unfinished</think>"
                            },
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {},
                }
            ).encode()

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: _Response(),
    )
    transport = SpecOpenAITransport(
        "https://example.test/v1",
        "secret",
        False,
    )
    client = LLMClient(0.0, 4096, transport=transport)

    with pytest.raises(LLMProtocolError, match="without a final answer"):
        client.query("system", "user", "openai/MiniMax-M3")


def test_manifest_records_the_code_version(tmp_path):
    """The protocol fingerprint pins the benchmark, not the engine. Without
    a code version a finished run cannot be attributed to the code that
    produced it, which silently invalidates comparisons spanning a change."""
    from experiments.imo_proof.result import code_version

    version = code_version()
    assert version == "unknown" or len(version.split("+")[0]) == 40

    manifest = RunManifest(
        schema_version=1,
        run_id="r1",
        experiment_id="e1",
        protocol_fingerprint="f" * 64,
        evolution_seed=0,
        seed_sha256="a" * 64,
        code_version=version,
    )
    manifest.write(tmp_path / "m.json")
    assert RunManifest.load(tmp_path / "m.json").code_version == version

    # older manifests without the field still load
    raw = json.loads((tmp_path / "m.json").read_text())
    del raw["code_version"]
    (tmp_path / "old.json").write_text(json.dumps(raw))
    assert RunManifest.load(tmp_path / "old.json").code_version == "unknown"


def test_evaluate_candidate_retries_are_wall_clock_bounded(tmp_path):
    """A retry count is not a bound: each attempt starts a fresh per-problem
    budget, so three retries of a 3000s budget let one stuck connection hold
    a run for 2.5h (observed live at 2h04m, process alive and silent)."""
    from experiments.imo_proof.evolution import _evaluate_candidate

    attempts = {"n": 0}
    clock = {"t": 0.0}

    class _Hanging:
        def evaluate_directory(self, **kwargs):
            attempts["n"] += 1
            clock["t"] += 2000.0          # each attempt burns its budget
            raise EvaluationUnavailable("connection stalled")

    import experiments.imo_proof.evolution as evo
    original = evo.monotonic
    evo.monotonic = lambda: clock["t"]
    try:
        candidate = Candidate(
            id="c1", code=GitWorkspace(
                base_files={"main.py": "x = 1\n"}, main_file="main.py",
            ).serialize(),
            generation=0, parent_id=None, island_idx=0, operator="seed",
            workspace_kind="git",
        )
        with pytest.raises(EvaluationUnavailable):
            _evaluate_candidate(
                _Hanging(), candidate, split="test",
                output_dir=tmp_path, retries=3, deadline_s=1000.0,
            )
    finally:
        evo.monotonic = original

    # second attempt is refused because the wall clock is already spent
    assert attempts["n"] == 1
