"""Host-native HyperAgents architecture evolution for the frozen IMO benchmark.

Docker is only an isolation mechanism in upstream HyperAgents. This runner
keeps the real MetaAgent, TaskAgent, patch lineage, archive, and parent
selection loop while executing candidates as a restricted operating-system
user. The privileged outer process alone owns reference answers and grading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pwd
import random
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from experiments.imo_proof.evaluation.engine import ProofDataset
from experiments.imo_proof.protocol import BenchmarkSpec, default_spec_path
from experiments.hyperagents_imo.scoring import _grader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HYPERAGENTS_ROOT = PROJECT_ROOT / "third_party" / "HyperAgents"
WORKER_SOURCE = Path(__file__).with_name("candidate_worker.py")
ALLOWED_SPLITS = ("train", "validation", "test")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        if (
            name in {".git", "outputs", ".experiment_deps", "__pycache__"}
            or name.startswith("._")
            or name.endswith((".pyc", ".pyo", ".csv"))
        ):
            ignored.add(name)
    return ignored


def _owner_ids(user: str) -> tuple[int, int]:
    record = pwd.getpwnam(user)
    return record.pw_uid, record.pw_gid


def _chown_tree(path: Path, user: str) -> None:
    uid, gid = _owner_ids(user)
    os.chown(path, uid, gid)
    for root, directories, files in os.walk(path):
        for name in directories:
            os.chown(Path(root) / name, uid, gid)
        for name in files:
            os.chown(Path(root) / name, uid, gid)


def _restricted_env() -> dict[str, str]:
    required = {
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "LLM_ENABLE_THINKING",
        "LLM_MAX_TOKENS",
        "LLM_REQUEST_TIMEOUT_S",
        "LLM_RETRY_MAX_TIME_S",
    }
    env = {
        key: value
        for key, value in os.environ.items()
        if key in required and value
    }
    missing = {"OPENAI_API_KEY", "OPENAI_API_BASE"} - set(env)
    if missing:
        raise RuntimeError(f"missing candidate runtime variables: {sorted(missing)}")
    env.setdefault("LLM_ENABLE_THINKING", "false")
    env.setdefault("LLM_MAX_TOKENS", "65536")
    env.setdefault("LLM_REQUEST_TIMEOUT_S", "300")
    env.setdefault("LLM_RETRY_MAX_TIME_S", "600")
    env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run_as_user(
    *,
    user: str,
    command: list[str],
    cwd: Path,
    timeout: float,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = _restricted_env()
    env["HOME"] = pwd.getpwnam(user).pw_dir
    env["USER"] = user
    env["LOGNAME"] = user
    return subprocess.run(
        ["/usr/sbin/runuser", "-u", user, "--", *command],
        cwd=cwd,
        text=True,
        capture_output=capture_output,
        timeout=timeout,
        check=False,
        env=env,
    )


def _git(
    workspace: Path,
    user: str,
    *arguments: str,
    timeout: float = 120,
) -> subprocess.CompletedProcess[str]:
    return _run_as_user(
        user=user,
        command=["git", "-C", str(workspace), *arguments],
        cwd=workspace,
        timeout=timeout,
    )


def _ensure_runtime_gitignore(workspace: Path) -> None:
    gitignore = workspace / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8") if gitignore.is_file() else ""
    rules = "\n# Experiment runtime artifacts\n__pycache__/\n*.py[cod]\n"
    if "__pycache__/" not in existing:
        gitignore.write_text(existing.rstrip() + rules, encoding="utf-8")


def _ensure_editor_tool_aliases(workspace: Path) -> None:
    path = workspace / "agent" / "llm_withtools.py"
    source = path.read_text(encoding="utf-8")
    marker = "def process_tool_call(tools_dict, tool_name, tool_input):\n"
    compatibility = (
        marker
        + "    # Normalize common direct editor actions emitted by OpenAI-style models.\n"
        + "    if tool_name in {\"view\", \"create\", \"str_replace\", \"insert\"}:\n"
        + "        tool_input = {\"command\": tool_name, **tool_input}\n"
        + "        tool_name = \"editor\"\n"
    )
    if "Normalize common direct editor actions" not in source:
        if marker not in source:
            raise ValueError("cannot locate HyperAgents process_tool_call")
        path.write_text(
            source.replace(marker, compatibility, 1),
            encoding="utf-8",
        )


def _initialize_workspace(workspace: Path, user: str) -> str:
    _ensure_runtime_gitignore(workspace)
    gitignore = workspace / ".gitignore"
    if gitignore.is_file():
        _chown_tree(gitignore, user)
    result = _git(workspace, user, "init")
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    for arguments in (
        ("config", "user.name", "HyperAgents Experiment"),
        ("config", "user.email", "hyperagents@example.invalid"),
        ("add", "--all"),
        ("commit", "-m", "candidate baseline"),
    ):
        result = _git(workspace, user, *arguments)
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
    result = _git(workspace, user, "rev-parse", "HEAD")
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


def _public_context(
    *,
    generation: int,
    parent_id: str | None,
    candidates: list[dict[str, Any]],
    iterations_left: int,
) -> str:
    compact = [
        {
            "candidate_id": item["candidate_id"],
            "parent_id": item["parent_id"],
            "train_points": item["train_points"],
            "train_max_points": item["train_max_points"],
            "train_points_percentage": item["train_points_percentage"],
            "valid": item["valid"],
        }
        for item in candidates
    ]
    return (
        "# Evolution context\n\n"
        "Improve the task-solving agent architecture in this repository. "
        "The protected evaluator uses 12 frozen IMO training problems. "
        "Do not attempt to access reference answers or hidden evaluation data.\n\n"
        f"- generation: {generation}\n"
        f"- parent: {parent_id}\n"
        f"- iterations left: {iterations_left}\n\n"
        "Previous candidate results:\n\n"
        f"```json\n{json.dumps(compact, indent=2, sort_keys=True)}\n```\n"
    )


def _select_parent(
    candidates: list[dict[str, Any]],
    *,
    generation: int,
    seed: int,
) -> str:
    valid = [item for item in candidates if item["valid"]]
    if not valid:
        return candidates[0]["candidate_id"]
    child_counts = {item["candidate_id"]: 0 for item in valid}
    for item in candidates:
        parent_id = item["parent_id"]
        if parent_id in child_counts:
            child_counts[parent_id] += 1
    scores = [item["train_points_percentage"] for item in valid]
    midpoint = sum(sorted(scores, reverse=True)[:3]) / min(3, len(scores))
    transformed = [1 / (1 + math.exp(-10 * (score - midpoint))) for score in scores]
    penalties = [
        math.exp(-((child_counts[item["candidate_id"]] / 8) ** 3))
        for item in valid
    ]
    weights = [score * penalty for score, penalty in zip(transformed, penalties)]
    rng = random.Random(seed + generation)
    return rng.choices(
        [item["candidate_id"] for item in valid],
        weights=weights,
        k=1,
    )[0]


def _prepare_initial_workspace(path: Path, user: str) -> str:
    shutil.copytree(HYPERAGENTS_ROOT, path, ignore=_copy_ignore)
    _chown_tree(path, user)
    return _initialize_workspace(path, user)


def _prepare_child_workspace(
    *,
    parent: Path,
    child: Path,
    user: str,
    context: str,
) -> str:
    shutil.copytree(parent, child, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    context_path = child / "EVOLUTION_CONTEXT.md"
    context_path.write_text(context, encoding="utf-8")
    _ensure_editor_tool_aliases(child)
    _chown_tree(child, user)
    return _initialize_workspace(child, user)


def _run_meta_agent(
    *,
    workspace: Path,
    public_run_dir: Path,
    output_dir: Path,
    model: str,
    agent_python: Path,
    user: str,
    base_commit: str,
    iterations_left: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    _chown_tree(output_dir, user)
    repo_instruction = (
        f"{workspace}. First read EVOLUTION_CONTEXT.md and inspect the public "
        f"evaluation archive at {public_run_dir}. Improve the actual TaskAgent "
        "architecture. Hidden reference answers and grader internals are not available."
    )
    result = _run_as_user(
        user=user,
        command=[
            str(agent_python),
            str(workspace / "run_meta_agent.py"),
            "--model",
            model,
            "--chat_history_file",
            str(output_dir / "meta_agent_chat_history.md"),
            "--repo_path",
            repo_instruction,
            "--evals_folder",
            str(public_run_dir),
            "--iterations_left",
            str(iterations_left),
            "--git_dir",
            str(workspace),
            "--base_commit",
            base_commit,
            "--outdir",
            str(output_dir),
        ],
        cwd=workspace,
        timeout=3600,
    )
    (output_dir / "meta_agent.stdout.txt").write_text(
        result.stdout + "\n" + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"MetaAgent exited with {result.returncode}: {result.stderr[-2000:]}"
        )
    patch = output_dir / "model_patch.diff"
    if not patch.is_file() or not patch.read_text(encoding="utf-8").strip():
        raise RuntimeError("MetaAgent produced an empty model_patch.diff")
    changed = _git(
        workspace,
        user,
        "diff",
        "--name-only",
        base_commit,
    )
    if changed.returncode != 0:
        raise RuntimeError(f"cannot inspect MetaAgent patch: {changed.stderr}")
    source_files = [
        value
        for value in changed.stdout.splitlines()
        if value.endswith(".py")
        and "__pycache__" not in Path(value).parts
        and not value.startswith("domains/")
    ]
    if not source_files:
        raise RuntimeError(
            "MetaAgent patch contains no Python source architecture change"
        )
    _write_json(
        output_dir / "patch_manifest.json",
        {
            "base_commit": base_commit,
            "changed_files": changed.stdout.splitlines(),
            "source_architecture_files": source_files,
        },
    )
    return patch


def _compile_candidate(
    workspace: Path,
    *,
    user: str,
    agent_python: Path,
) -> None:
    result = _run_as_user(
        user=user,
        command=[
            str(agent_python),
            "-c",
            "from task_agent import TaskAgent; print(TaskAgent.__name__)",
        ],
        cwd=workspace,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"candidate import failed: {result.stderr[-2000:]}")


def _run_problem(
    *,
    workspace: Path,
    problem_file: Path,
    output: Path,
    log: Path,
    model: str,
    user: str,
    agent_python: Path,
    worker_path: Path,
    timeout_s: float,
    max_calls: int,
) -> None:
    result = _run_as_user(
        user=user,
        command=[
            str(agent_python),
            str(worker_path),
            "--workspace",
            str(workspace),
            "--problem-file",
            str(problem_file),
            "--output",
            str(output),
            "--log",
            str(log),
            "--model",
            model,
            "--max-calls",
            str(max_calls),
        ],
        cwd=workspace,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"candidate worker failed: {result.stderr[-2000:]} {result.stdout[-1000:]}"
        )


def _grade_proof(
    *,
    grader: Any,
    record: Any,
    proof: str,
    private_item: Path,
    spec: BenchmarkSpec,
) -> dict[str, Any]:
    proof_hash = _sha256_text(proof)
    if private_item.is_file():
        value = json.loads(private_item.read_text(encoding="utf-8"))
        if value.get("proof_sha256") != proof_hash:
            raise ValueError(f"saved grade hash mismatch for {record.problem_id}")
        return value
    outcome = grader.grade(record, proof)
    value = {
        "problem_id": record.problem_id,
        "proof_sha256": proof_hash,
        "proof_characters": len(proof),
        "label": outcome.label,
        "points": spec.scoring.points[outcome.label],
        "max_points": spec.scoring.max_points,
        "grader_usage": {
            "calls": outcome.usage.calls,
            "prompt_tokens": outcome.usage.prompt_tokens,
            "completion_tokens": outcome.usage.completion_tokens,
            "cost_usd": outcome.usage.cost_usd,
        },
        "grader_response": outcome.raw_response,
    }
    _write_json(private_item, value)
    return value


def _evaluate_workspace(
    *,
    candidate_id: str,
    workspace: Path,
    split: str,
    public_eval_dir: Path,
    private_eval_dir: Path,
    spec: BenchmarkSpec,
    dataset: ProofDataset,
    solver_model: str,
    user: str,
    agent_python: Path,
    worker_path: Path,
    solver_workers: int,
    grader_workers: int,
) -> dict[str, Any]:
    ids = spec.splits.ids(split)
    public_eval_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = public_eval_dir / "inputs"
    outputs_dir = public_eval_dir / "outputs"
    logs_dir = public_eval_dir / "logs"
    for directory in (inputs_dir, outputs_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _chown_tree(public_eval_dir, user)

    records = {record.problem_id: record for record in dataset.select(ids)}
    pending = []
    for problem_id in ids:
        input_path = inputs_dir / f"{problem_id}.json"
        if not input_path.is_file():
            _write_json(
                input_path,
                {
                    "problem_id": problem_id,
                    "problem": records[problem_id].problem,
                },
            )
            _chown_tree(input_path, user)
        output_path = outputs_dir / f"{problem_id}.json"
        if not output_path.is_file():
            pending.append(problem_id)

    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=solver_workers) as pool:
        futures = {
            pool.submit(
                _run_problem,
                workspace=workspace,
                problem_file=inputs_dir / f"{problem_id}.json",
                output=outputs_dir / f"{problem_id}.json",
                log=logs_dir / f"{problem_id}.md",
                model=solver_model,
                user=user,
                agent_python=agent_python,
                worker_path=worker_path,
                timeout_s=spec.solver_budget.timeout_s_per_problem,
                max_calls=spec.solver_budget.max_calls_per_problem,
            ): problem_id
            for problem_id in pending
        }
        for future in as_completed(futures):
            problem_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures[problem_id] = f"{type(exc).__name__}: {exc}"

    proofs: dict[str, str] = {}
    for problem_id in ids:
        output_path = outputs_dir / f"{problem_id}.json"
        if not output_path.is_file():
            failures.setdefault(problem_id, "missing candidate output")
            continue
        value = json.loads(output_path.read_text(encoding="utf-8"))
        proof = str(value.get("proof", "")).strip()
        if proof:
            proofs[problem_id] = proof
        else:
            failures[problem_id] = "empty candidate proof"

    grader = _grader(spec)
    graded: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=grader_workers) as pool:
        futures = {
            pool.submit(
                _grade_proof,
                grader=grader,
                record=records[problem_id],
                proof=proof,
                private_item=private_eval_dir / "items" / f"{problem_id}.json",
                spec=spec,
            ): problem_id
            for problem_id, proof in proofs.items()
        }
        for future in as_completed(futures):
            problem_id = futures[future]
            graded[problem_id] = future.result()

    items = []
    for problem_id in ids:
        if problem_id in graded:
            value = graded[problem_id]
            items.append(
                {
                    key: value[key]
                    for key in (
                        "problem_id",
                        "proof_sha256",
                        "proof_characters",
                        "label",
                        "points",
                        "max_points",
                        "grader_usage",
                    )
                }
            )
        else:
            items.append(
                {
                    "problem_id": problem_id,
                    "proof_sha256": None,
                    "proof_characters": 0,
                    "label": "incorrect",
                    "points": 0,
                    "max_points": spec.scoring.max_points,
                    "grader_usage": {
                        "calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cost_usd": 0.0,
                    },
                    "failure": failures[problem_id],
                }
            )
    earned = sum(item["points"] for item in items)
    maximum = sum(item["max_points"] for item in items)
    correct = sum(item["points"] == item["max_points"] for item in items)
    report = {
        "candidate_id": candidate_id,
        "split": split,
        "problem_count": len(items),
        "points": earned,
        "max_points": maximum,
        "points_percentage": earned / maximum,
        "correct_count": correct,
        "correct_percentage": correct / len(items),
        "failure_count": len(failures),
        "items": items,
    }
    _write_json(public_eval_dir / "report.json", report)
    return report


def _candidate_record(
    *,
    candidate_id: str,
    parent_id: str | None,
    generation: int,
    workspace: Path,
    report: dict[str, Any],
    valid: bool,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "parent_id": parent_id,
        "generation": generation,
        "workspace": str(workspace),
        "train_points": report["points"],
        "train_max_points": report["max_points"],
        "train_points_percentage": report["points_percentage"],
        "train_correct_percentage": report["correct_percentage"],
        "valid": valid,
        "error": error,
    }


def evolve(
    *,
    public_run_dir: Path,
    private_run_dir: Path,
    candidates_budget: int,
    solver_workers: int,
    grader_workers: int,
    seed: int,
    sandbox_user: str,
    agent_python: Path,
    stop_after: int | None = None,
) -> dict[str, Any]:
    spec = BenchmarkSpec.load(default_spec_path())
    spec.verify_workspace(PROJECT_ROOT)
    if candidates_budget != spec.optimizer_budget.max_candidates:
        raise ValueError(
            "candidate budget must equal BenchmarkSpec.optimizer_budget.max_candidates"
        )
    if len(spec.splits.train) != 12:
        raise ValueError("frozen train split must contain exactly 12 problems")
    dataset = ProofDataset.load(PROJECT_ROOT, spec)
    public_run_dir.mkdir(parents=True, exist_ok=True)
    private_run_dir.mkdir(parents=True, exist_ok=True)
    worker_path = public_run_dir / "infrastructure" / "candidate_worker.py"
    worker_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(WORKER_SOURCE, worker_path)
    _chown_tree(public_run_dir, sandbox_user)
    manifest = {
        "protocol": "hyperagents-host-native-v1",
        "benchmark_id": spec.benchmark_id,
        "benchmark_fingerprint": spec.fingerprint,
        "dataset_sha256": spec.dataset.sha256,
        "train_ids": list(spec.splits.train),
        "optimizer_model": spec.optimizer.name,
        "solver_model": spec.solver.name,
        "grader_model": spec.grader.model.name,
        "grader_prompt_sha256": spec.grader.prompt_sha256,
        "candidate_budget": candidates_budget,
        "parent_selection": "score_child_prop",
        "seed": seed,
        "sandbox_user": sandbox_user,
        "docker_used": False,
    }
    manifest_path = public_run_dir / "manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("run manifest does not match requested protocol")
    else:
        _write_json(manifest_path, manifest)

    candidates: list[dict[str, Any]] = []
    archive_path = public_run_dir / "archive.jsonl"
    if archive_path.is_file():
        candidates = [
            json.loads(line)
            for line in archive_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    target_count = candidates_budget if stop_after is None else min(
        candidates_budget, stop_after
    )
    if target_count < 1:
        raise ValueError("stop_after must be at least 1")
    for generation in range(len(candidates), target_count):
        candidate_id = "initial" if generation == 0 else f"{generation:03d}"
        candidate_dir = public_run_dir / "candidates" / candidate_id
        workspace = candidate_dir / "workspace"
        parent_id = None
        error = None
        valid = True
        try:
            if generation == 0:
                if not workspace.is_dir():
                    workspace.parent.mkdir(parents=True, exist_ok=True)
                    base_commit = _prepare_initial_workspace(workspace, sandbox_user)
                else:
                    base_commit = _git(
                        workspace, sandbox_user, "rev-parse", "HEAD"
                    ).stdout.strip()
            else:
                parent_id = _select_parent(
                    candidates,
                    generation=generation,
                    seed=seed,
                )
                parent_workspace = Path(
                    next(
                        item["workspace"]
                        for item in candidates
                        if item["candidate_id"] == parent_id
                    )
                )
                if not workspace.is_dir():
                    workspace.parent.mkdir(parents=True, exist_ok=True)
                    base_commit = _prepare_child_workspace(
                        parent=parent_workspace,
                        child=workspace,
                        user=sandbox_user,
                        context=_public_context(
                            generation=generation,
                            parent_id=parent_id,
                            candidates=candidates,
                            iterations_left=candidates_budget - generation - 1,
                        ),
                    )
                    _run_meta_agent(
                        workspace=workspace,
                        public_run_dir=public_run_dir,
                        output_dir=candidate_dir / "agent_output",
                        model=spec.optimizer.name,
                        agent_python=agent_python,
                        user=sandbox_user,
                        base_commit=base_commit,
                        iterations_left=candidates_budget - generation - 1,
                    )
            _compile_candidate(
                workspace,
                user=sandbox_user,
                agent_python=agent_python,
            )
            report = _evaluate_workspace(
                candidate_id=candidate_id,
                workspace=workspace,
                split="train",
                public_eval_dir=candidate_dir / "imo_proof_eval",
                private_eval_dir=private_run_dir / "candidates" / candidate_id / "train",
                spec=spec,
                dataset=dataset,
                solver_model=spec.solver.name,
                user=sandbox_user,
                agent_python=agent_python,
                worker_path=worker_path,
                solver_workers=solver_workers,
                grader_workers=grader_workers,
            )
        except Exception as exc:
            valid = False
            error = f"{type(exc).__name__}: {exc}"
            report = {
                "points": 0,
                "max_points": len(spec.splits.train) * spec.scoring.max_points,
                "points_percentage": 0.0,
                "correct_percentage": 0.0,
            }
        record = _candidate_record(
            candidate_id=candidate_id,
            parent_id=parent_id,
            generation=generation,
            workspace=workspace,
            report=report,
            valid=valid,
            error=error,
        )
        with archive_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        candidates.append(record)
        print(
            f"candidate {candidate_id}: "
            f"{record['train_points']}/{record['train_max_points']} "
            f"valid={record['valid']}",
            flush=True,
        )

    valid_candidates = [item for item in candidates if item["valid"]]
    if not valid_candidates:
        raise RuntimeError("evolution produced no valid architecture")
    if len(candidates) < candidates_budget:
        return {
            "status": "in_progress",
            "candidate_count": len(candidates),
            "candidate_budget": candidates_budget,
            "latest_candidate": candidates[-1],
        }
    best = max(
        valid_candidates,
        key=lambda item: (item["train_points_percentage"], -item["generation"]),
    )
    lineage = []
    cursor: str | None = best["candidate_id"]
    by_id = {item["candidate_id"]: item for item in candidates}
    while cursor is not None:
        lineage.append(cursor)
        cursor = by_id[cursor]["parent_id"]
    lineage.reverse()
    result = {
        "best_candidate_id": best["candidate_id"],
        "best_workspace": best["workspace"],
        "best_train_points": best["train_points"],
        "best_train_max_points": best["train_max_points"],
        "best_train_points_percentage": best["train_points_percentage"],
        "lineage": lineage,
        "candidate_count": len(candidates),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    _write_json(public_run_dir / "final_architecture.json", result)
    return result


def evaluate_final(
    *,
    public_run_dir: Path,
    private_run_dir: Path,
    split: str,
    solver_workers: int,
    grader_workers: int,
    sandbox_user: str,
    agent_python: Path,
) -> dict[str, Any]:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"invalid split {split!r}")
    spec = BenchmarkSpec.load(default_spec_path())
    spec.verify_workspace(PROJECT_ROOT)
    dataset = ProofDataset.load(PROJECT_ROOT, spec)
    final = json.loads(
        (public_run_dir / "final_architecture.json").read_text(encoding="utf-8")
    )
    worker_path = public_run_dir / "infrastructure" / "candidate_worker.py"
    return _evaluate_workspace(
        candidate_id=final["best_candidate_id"],
        workspace=Path(final["best_workspace"]),
        split=split,
        public_eval_dir=public_run_dir / "final_evaluation" / split,
        private_eval_dir=private_run_dir / "final_evaluation" / split,
        spec=spec,
        dataset=dataset,
        solver_model=spec.solver.name,
        user=sandbox_user,
        agent_python=agent_python,
        worker_path=worker_path,
        solver_workers=solver_workers,
        grader_workers=grader_workers,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-run-dir", type=Path, required=True)
    parser.add_argument("--private-run-dir", type=Path, required=True)
    parser.add_argument("--sandbox-user", default="hyperagent")
    parser.add_argument("--agent-python", type=Path, required=True)
    parser.add_argument("--solver-workers", type=int, default=5)
    parser.add_argument("--grader-workers", type=int, default=3)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evolve_parser = subparsers.add_parser("evolve")
    evolve_parser.add_argument("--candidates", type=int, default=15)
    evolve_parser.add_argument("--seed", type=int, default=0)
    evolve_parser.add_argument("--stop-after", type=int)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--split", choices=ALLOWED_SPLITS, required=True)
    args = parser.parse_args(argv)

    if args.command == "evolve":
        result = evolve(
            public_run_dir=args.public_run_dir,
            private_run_dir=args.private_run_dir,
            candidates_budget=args.candidates,
            solver_workers=args.solver_workers,
            grader_workers=args.grader_workers,
            seed=args.seed,
            sandbox_user=args.sandbox_user,
            agent_python=args.agent_python,
            stop_after=args.stop_after,
        )
    else:
        result = evaluate_final(
            public_run_dir=args.public_run_dir,
            private_run_dir=args.private_run_dir,
            split=args.split,
            solver_workers=args.solver_workers,
            grader_workers=args.grader_workers,
            sandbox_user=args.sandbox_user,
            agent_python=args.agent_python,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
