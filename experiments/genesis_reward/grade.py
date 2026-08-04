"""Grade a candidate reward function by training a Go2 walking policy with it.

One evaluation = one full PPO training run in HyperAgents' Genesis harness
(`rl_trainer`), followed by one rollout of the trained policy (`rl_eval`).
Fitness is the task fitness the environment measures itself — tracking of the
commanded forward velocity — NOT the reward the candidate wrote. A candidate
that inflates its own reward therefore gains nothing; only a reward that
actually teaches the robot to walk at the commanded speed moves the score.

Both phases run as subprocesses of a dedicated Genesis interpreter, because
Genesis initialises CUDA and Taichi at import time and cannot be re-entered
inside a long-lived loop process.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
from pathlib import Path

from evoharness.evoserve import Grade, GradeContext, InfraError

# Wall-clock ceiling for one phase. HyperAgents uses 3600s for both; a
# candidate whose reward makes the simulation crawl is a bad candidate, but
# a machine-level stall is infrastructure, so the two are reported apart.
_DEFAULT_TIMEOUT_S = 3600.0

# The environment attribute the fitness is defined on. Used only for the
# failure message when a candidate's reward tensor has the wrong shape.
_TASK = "Go2WalkingCommand-v0"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def _hyperagents_root() -> Path:
    raw = os.environ.get("HYPERAGENTS_ROOT")
    if raw:
        root = Path(raw)
    else:
        root = Path(__file__).resolve().parents[2] / "third_party" / "HyperAgents"
    if not (root / "domains" / "genesis").is_dir():
        raise InfraError(f"HyperAgents Genesis domain not found under {root}")
    return root


def _genesis_python() -> str:
    python = os.environ.get("GENESIS_PYTHON")
    if not python:
        raise InfraError(
            "GENESIS_PYTHON is unset: Genesis needs its own interpreter "
            "(torch + genesis-world + rsl-rl-lib==2.2.4)"
        )
    if not Path(python).exists():
        raise InfraError(f"GENESIS_PYTHON={python} does not exist")
    return python


def _subprocess_env() -> dict:
    env = dict(os.environ)
    # Genesis renders headless; without this the OpenGL import goes looking
    # for GLX and dies on a machine with no X server.
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("MPLBACKEND", "Agg")
    # HyperAgents' modules are imported as `domains.genesis...` from its root.
    env["PYTHONPATH"] = str(_hyperagents_root())
    return env


def _run_phase(command: list[str], *, cwd: Path, timeout_s: float) -> tuple[int, str]:
    """Run one Genesis phase, returning (returncode, tail of combined output)."""

    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            env=_subprocess_env(),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return 124, f"phase exceeded {timeout_s:.0f}s"
    except OSError as exc:  # interpreter missing, fork failure, ...
        raise InfraError(f"could not launch Genesis phase: {exc}") from exc
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode, output


def _read_eval_stats(eval_dir: Path) -> dict | None:
    files = sorted(eval_dir.glob("*.json"))
    if not files:
        return None
    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _drop_truncated_tail(scores: list[float]) -> list[float]:
    """Discard the partial episode the rollout is cut off in the middle of.

    `rl_eval` stops after a fixed number of steps and flushes whatever has
    accumulated since the last reset, so the recorded list ends with a stub —
    0.004 against ~0.78 for the seed. Averaging it in is not merely a constant
    offset: a policy that survives longer resets less often, so it records
    fewer complete episodes and the stub carries MORE weight in its mean.
    Keeping it would pay candidates to fall over. The stub is identified by
    being far below the rest, never by position alone, so a genuinely bad
    final episode is kept.
    """

    if len(scores) < 2:
        return list(scores)
    body = scores[:-1]
    reference = statistics.median(body)
    if reference > 0 and scores[-1] < 0.25 * reference:
        return body
    return list(scores)


def _failure_excerpt(output: str, limit: int = 1200) -> str:
    """The last real lines of a failed phase — this is what the mutation
    prompt gets to see, so it must carry the reason, not just the tail of a
    progress bar."""

    lines = [line for line in output.splitlines() if line.strip()]
    interesting = [
        line
        for line in lines
        if any(
            marker in line
            for marker in ("Error", "error", "Traceback", "Exception", "assert")
        )
    ]
    chosen = interesting[-12:] if interesting else lines[-12:]
    return "\n".join(chosen)[-limit:]


def grade_workspace(candidate_dir: Path, ctx: GradeContext) -> Grade:
    root = _hyperagents_root()
    python = _genesis_python()
    num_envs = _env_int("GENESIS_NUM_ENVS", 4096)
    max_iterations = _env_int("GENESIS_MAX_ITERS", 101)
    max_steps = _env_int("GENESIS_EVAL_STEPS", 1000)
    timeout_s = _env_float("GENESIS_TIMEOUT_S", _DEFAULT_TIMEOUT_S)
    vel_low = _env_float("GENESIS_VEL_LOW", 0.2)
    vel_high = _env_float("GENESIS_VEL_HIGH", 0.8)

    reward_file = Path(candidate_dir) / "reward_function.py"
    if not reward_file.is_file():
        return Grade(
            fitness=0.0,
            passed=False,
            fault="genome is missing reward_function.py",
            visible_metrics={"structural_doa": True},
            structured_feedback={
                "schema_version": 1,
                "summary": "no reward_function.py in the candidate workspace",
            },
        )

    work = Path(ctx.workdir) / "genesis"
    work.mkdir(parents=True, exist_ok=True)
    # rl_trainer wipes and rewrites its own log dir, so hand it a clean one.
    exp_name = f"{_TASK}/speed_episode_00"
    run_dir = work / "rl"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    train_cmd = [
        python, "-m", "domains.genesis.genesis_train.rl_trainer",
        "-e", exp_name,
        "--num_envs", str(num_envs),
        "--max_iterations", str(max_iterations),
        "--output_dir", str(run_dir),
        "--rwd_func_path", str(reward_file),
        "--episode_idx", "00",
        "--lin_vel_x_range", str(vel_low), str(vel_high),
    ]
    train_rc, train_out = _run_phase(train_cmd, cwd=root, timeout_s=timeout_s)
    if train_rc != 0:
        excerpt = _failure_excerpt(train_out)
        return Grade(
            fitness=0.0,
            passed=False,
            fault=f"training failed (rc={train_rc})",
            visible_metrics={"train_failed": True, "structural_doa": False},
            structured_feedback={
                "schema_version": 1,
                "phase": "train",
                "summary": (
                    "the policy never trained: the reward function raised, or "
                    "produced a tensor the trainer could not use"
                ),
                "error_excerpt": excerpt,
            },
            stderr_log=excerpt,
            stage_reached=1,
        )

    eval_cmd = [
        python, "-m", "domains.genesis.genesis_eval.rl_eval",
        "-e", exp_name,
        "--output_dir", str(run_dir),
        "--ckpt", str(max_iterations - 1),
        "--num_envs", str(num_envs),
        "--max_steps", str(max_steps),
        "--rwd_func_path", str(reward_file),
        "--no-record_video",
        "--episode_idx", "00",
    ]
    eval_rc, eval_out = _run_phase(eval_cmd, cwd=root, timeout_s=timeout_s)
    stats = _read_eval_stats(run_dir / "genesis_eval_00")
    if eval_rc != 0 or stats is None:
        excerpt = _failure_excerpt(eval_out)
        return Grade(
            fitness=0.0,
            passed=False,
            fault=f"rollout failed (rc={eval_rc})",
            visible_metrics={"eval_failed": True, "structural_doa": False},
            structured_feedback={
                "schema_version": 1,
                "phase": "eval",
                "summary": "the trained policy could not be rolled out",
                "error_excerpt": excerpt,
            },
            stderr_log=excerpt,
            stage_reached=2,
        )

    raw_scores = [float(value) for value in stats.get("fitness_score", [])]
    scores = _drop_truncated_tail(raw_scores)
    if not scores:
        # Training and rollout both ran, but no episode ever finished — a
        # policy that falls over instantly still completes episodes, so this
        # means the rollout produced nothing measurable.
        return Grade(
            fitness=0.0,
            passed=False,
            fault="no episode completed during the rollout",
            visible_metrics={"episodes": 0, "structural_doa": False},
            structured_feedback={
                "schema_version": 1,
                "phase": "eval",
                "summary": "rollout finished but recorded zero completed episodes",
            },
            stage_reached=2,
        )

    fitness = statistics.fmean(scores)
    stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    sem = stdev / (len(scores) ** 0.5) if scores else 0.0

    components = {
        name: (statistics.fmean([float(v) for v in values]) if values else 0.0)
        for name, values in (stats.get("reward_component") or {}).items()
    }
    rewards = [float(value) for value in stats.get("total_reward", [])]

    return Grade(
        fitness=fitness,
        passed=True,
        visible_metrics={
            "average_fitness": fitness,
            "fitness_stdev": stdev,
            "episodes": len(scores),
            # HyperAgents' own summariser averages every recorded entry,
            # truncated tail included. Kept for comparability with numbers
            # published against their harness.
            "average_fitness_untrimmed": (
                statistics.fmean(raw_scores) if raw_scores else 0.0
            ),
            "episodes_recorded": len(raw_scores),
            "total_episodes_played": stats.get("total_episodes_played", 0),
            "average_total_reward": statistics.fmean(rewards) if rewards else 0.0,
            "num_envs": stats.get("num_envs", num_envs),
            "structural_doa": False,
        },
        structured_feedback={
            "schema_version": 1,
            "summary": (
                f"average fitness {fitness:.4f} over {len(scores)} completed "
                f"episodes (stdev {stdev:.4f})"
            ),
            "reward_component_means": components,
            "average_total_reward": (
                statistics.fmean(rewards) if rewards else 0.0
            ),
            # The single most useful signal for the next mutation: how much of
            # the reward the robot actually collected came from each term.
            "note": (
                "fitness is exp(-4 * (commanded_vx - measured_vx)^2) integrated "
                "over the episode and normalised by resampling time; it is "
                "independent of the reward this genome defines"
            ),
        },
        n_units=len(scores),
        sem=sem,
    )


__all__ = ["grade_workspace"]
