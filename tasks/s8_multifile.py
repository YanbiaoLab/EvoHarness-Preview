"""S8 multi-file coding smoke used by offline CI and optional live runs."""

from __future__ import annotations

import sys
from pathlib import Path

from evoharness.evocore import (
    EvalReport,
    LLMResponse,
    LLMStopReason,
    LLMToolCall,
    PreflightIssue,
    PreflightResult,
)
from evoharness.evocore.preflight import PreflightContext
from evoharness.evocore.workspace import GitWorkspace
from evoharness.evoguard import Sandbox

from recipes.common import TaskBundle


TASK_SYS_MSG = """Improve the small Python project so all project tests pass.
You must update both math_ops.py and metadata.py. Inspect and run the tests
before finishing. Keep the public run(values) interface in main.py unchanged.
"""

BASE_FILES = {
    "main.py": (
        "from math_ops import increment_all\n"
        "from metadata import VERSION\n\n"
        "def run(values):\n"
        "    return increment_all(values), VERSION\n"
    ),
    "math_ops.py": (
        "def increment_all(values):\n"
        "    return list(values)\n"
    ),
    "metadata.py": 'VERSION = "0"\n',
    "tests/test_solution.py": (
        "import unittest\n\n"
        "from main import run\n\n"
        "class SolutionTest(unittest.TestCase):\n"
        "    def test_contract(self):\n"
        "        self.assertEqual(run([1, 2, 3]), ([2, 3, 4], \"1\"))\n\n"
        "if __name__ == \"__main__\":\n"
        "    unittest.main()\n"
    ),
}

FINAL_FILES = {
    "math_ops.py": (
        "def increment_all(values):\n"
        "    return [value + 1 for value in values]\n"
    ),
    "metadata.py": 'VERSION = "1"\n',
}


class ProjectTestsValidator:
    """Task-specific validator; the framework does not guess project commands."""

    name = "project-tests"

    def __init__(self, runner: Sandbox, timeout_s: float = 5.0):
        self.runner = runner
        self.timeout_s = timeout_s

    def validate(self, ctx: PreflightContext) -> PreflightResult:
        command = (
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-q",
        )
        result = self.runner.run(
            list(command),
            workdir=ctx.workdir,
            timeout_s=self.timeout_s,
        )
        if result.ok:
            return PreflightResult(self.name)
        code = "project-tests-timeout" if result.timed_out else "project-tests-failed"
        return PreflightResult(
            self.name,
            (
                PreflightIssue(
                    validator=self.name,
                    code=code,
                    message=(
                        "Project tests exceeded their deadline"
                        if result.timed_out
                        else "Project tests failed"
                    ),
                    command=command,
                    stdout=result.stdout,
                    stderr=result.stderr,
                ),
            ),
        )


class MultiFileSmokeGrader:
    def __init__(self, validator: ProjectTestsValidator):
        self.validator = validator

    def grade(self, cand, workdir: Path) -> EvalReport:
        candidate_dir = cand.workspace.materialize(workdir / "candidate")
        result = self.validator.validate(
            PreflightContext(
                parent=cand,
                operator=cand.operator,
                workdir=candidate_dir,
            )
        )
        solved = result.ok
        return EvalReport(
            fitness=1.0 if solved else 0.0,
            # A failing project test remains an evaluable parent. The Agent
            # proposal admission gate is stricter and requires validator.ok.
            passed=True,
            fault=None if solved else result.issues[0].code,
            visible_metrics={"project_tests_passed": solved},
            stderr_log="" if solved else result.issues[0].stderr,
        )


def _offline_transport_factory():
    def transport(messages, model, **kwargs):
        tools = kwargs.get("tools", ())
        if tools:
            if messages[-1].role != "tool":
                calls = tuple(
                    LLMToolCall(
                        call_id=f"s8-write-{index}",
                        name="workspace_write",
                        arguments={"path": path, "content": content},
                    )
                    for index, (path, content) in enumerate(
                        FINAL_FILES.items(),
                        start=1,
                    )
                )
                return LLMResponse(
                    text="",
                    model=model,
                    cost=0.001,
                    stop_reason=LLMStopReason.TOOL_CALLS,
                    tool_calls=calls,
                )
            return LLMResponse(
                text=(
                    "TITLE: satisfy project contract\n"
                    "SUMMARY: update implementation and version metadata"
                ),
                model=model,
                cost=0.001,
            )

        blocks = "".join(
            f"### FILE: {path}\n```python\n{content}```\n"
            for path, content in FINAL_FILES.items()
        )
        return LLMResponse(
            text=(
                "TITLE: satisfy project contract\n"
                "SUMMARY: update implementation and version metadata\n"
                + blocks
            ),
            model=model,
            cost=0.001,
        )

    return transport


def make_task() -> TaskBundle:
    runner = Sandbox(allow_network=False)
    validator = ProjectTestsValidator(runner)
    workspace = GitWorkspace(base_files=dict(BASE_FILES))
    return TaskBundle(
        grader=MultiFileSmokeGrader(validator),
        initial_code=workspace.main_text(),
        initial_workspace=workspace,
        task_sys_msg=TASK_SYS_MSG,
        transport=_offline_transport_factory(),
        preflight_validators=(validator,),
        runner=runner,
    )
