"""Task bundle for Robotics Reward Design (HyperAgents Genesis go2walking).

The genome is a single file, `reward_function.py`, defining `compute_reward(env)`.
Everything else — the robot, the simulator, the PPO recipe, and the fitness
measure — is fixed by HyperAgents, so the only thing that varies between
candidates is the reward design. That is exactly the domain's claim.
"""

from __future__ import annotations

import ast
from pathlib import Path

from evoharness import ScorableTask
from evoharness.evocore import PreflightIssue, PreflightResult
from evoharness.evocore.preflight import PreflightContext
from evoharness.evocore.workspace import GitWorkspace

_HERE = Path(__file__).resolve().parent
_SEEDS = _HERE / "seeds"
_GENOME_FILES = ("reward_function.py",)


class RewardContractValidator:
    """Reject a genome that cannot possibly train, without paying for a GPU.

    One PPO run costs minutes of L40S time; a missing `compute_reward` or a
    syntax error costs microseconds to catch here. Nothing about the reward's
    *quality* is judged — that is the grader's job and only the robot can
    answer it.
    """

    name = "reward-contract"

    def validate(self, ctx: PreflightContext) -> PreflightResult:
        path = Path(ctx.workdir) / "reward_function.py"
        if not path.is_file():
            return PreflightResult(
                self.name,
                (
                    PreflightIssue(
                        self.name,
                        "reward-file-missing",
                        "the genome must contain reward_function.py",
                        path="reward_function.py",
                    ),
                ),
            )
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename="reward_function.py")
        except SyntaxError as exc:
            return PreflightResult(
                self.name,
                (
                    PreflightIssue(
                        self.name,
                        "reward-syntax-error",
                        f"reward_function.py does not parse: {exc.msg}",
                        path="reward_function.py",
                        line=exc.lineno,
                        column=exc.offset,
                    ),
                ),
            )
        defined = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if "compute_reward" not in defined:
            return PreflightResult(
                self.name,
                (
                    PreflightIssue(
                        self.name,
                        "reward-entrypoint-missing",
                        "reward_function.py must define a top-level "
                        "compute_reward(env) returning "
                        "(total_reward, reward_components, reward_scales)",
                        path="reward_function.py",
                    ),
                ),
            )
        return PreflightResult(self.name)


def make_task() -> ScorableTask:
    from genesis_reward.grade import grade_workspace

    return ScorableTask.from_directory(
        _SEEDS / "naive",
        grade_workspace,
        main_file="reward_function.py",
        include_files=_GENOME_FILES,
        task_sys_msg=(_HERE / "sys_msg.md").read_text(encoding="utf-8"),
        research_brief=(_HERE / "research_msg.md").read_text(encoding="utf-8"),
        preflight_validators=(RewardContractValidator(),),
    )


def _workspace(name: str) -> GitWorkspace:
    return GitWorkspace.from_directory(
        _SEEDS / name,
        main_file="reward_function.py",
        include_files=_GENOME_FILES,
    )


__all__ = ["make_task", "RewardContractValidator"]
