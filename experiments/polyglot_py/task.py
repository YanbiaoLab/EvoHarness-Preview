"""Task bundle for Coding (Polyglot Python) — evolve the coding harness.

What evolves is the program that turns an exercise into source files, not the
solutions themselves: the same thing HyperAgents evolves in its coding domain,
and the same genome shape as this repository's IMO task (an entry point, a
control policy, and the prompts, in three files so a mutation can change one
without re-emitting the others).
"""

from __future__ import annotations

import ast
from pathlib import Path

from evoharness import ScorableTask
from evoharness.evocore import PreflightIssue, PreflightResult
from evoharness.evocore.preflight import PreflightContext

_HERE = Path(__file__).resolve().parent
_SEEDS = _HERE / "seeds"
_GENOME_FILES = ("solver.py", "policy.py", "prompts.py")


class SolverContractValidator:
    """Catch a genome that cannot run before spending a model budget on it."""

    name = "solver-contract"

    def validate(self, ctx: PreflightContext) -> PreflightResult:
        issues = []
        workdir = Path(ctx.workdir)
        for name in _GENOME_FILES:
            path = workdir / name
            if not path.is_file():
                issues.append(
                    PreflightIssue(
                        self.name,
                        "genome-file-missing",
                        f"the genome must contain {name}",
                        path=name,
                    )
                )
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=name)
            except SyntaxError as exc:
                issues.append(
                    PreflightIssue(
                        self.name,
                        "syntax-error",
                        f"{name} does not parse: {exc.msg}",
                        path=name,
                        line=exc.lineno,
                        column=exc.offset,
                    )
                )
                continue
            if name == "solver.py":
                entry = [
                    node for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "solve"
                ]
                if not entry:
                    issues.append(
                        PreflightIssue(
                            self.name,
                            "entrypoint-missing",
                            "solver.py must define solve(task, llm, tools)",
                            path=name,
                        )
                    )
                elif len(entry[0].args.args) != 3:
                    issues.append(
                        PreflightIssue(
                            self.name,
                            "entrypoint-arity",
                            "solve must take exactly (task, llm, tools); "
                            f"found {len(entry[0].args.args)} parameters",
                            path=name,
                            line=entry[0].lineno,
                        )
                    )
        return PreflightResult(self.name, tuple(issues))


def make_task() -> ScorableTask:
    from polyglot_py.grade import grade_workspace

    return ScorableTask.from_directory(
        _SEEDS / "base",
        grade_workspace,
        main_file="solver.py",
        include_files=_GENOME_FILES,
        task_sys_msg=(_HERE / "sys_msg.md").read_text(encoding="utf-8"),
        research_brief=(_HERE / "research_msg.md").read_text(encoding="utf-8"),
        preflight_validators=(SolverContractValidator(),),
    )


__all__ = ["make_task", "SolverContractValidator"]
