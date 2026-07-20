"""Ordered, fail-closed proposal preflight execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Iterable, Protocol, runtime_checkable

from .workspace import Workspace, WorkspaceError

if TYPE_CHECKING:
    from .population import Candidate


@dataclass(frozen=True)
class PreflightIssue:
    """Provider-neutral diagnostic IR produced by preflight validators."""

    validator: str
    code: str
    message: str

    repairable: bool = True
    path: str | None = None
    line: int | None = None
    column: int | None = None

    command: tuple[str, ...] | None = None
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class PreflightResult:
    """Provider-neutral result from one validator or pipeline stage."""

    stage: str
    issues: tuple[PreflightIssue, ...] = ()
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def repairable(self) -> bool:
        return bool(self.issues) and all(
            issue.repairable for issue in self.issues
        )

    def __post_init__(self) -> None:
        if self.elapsed_s < 0:
            raise ValueError("preflight elapsed_s cannot be negative")


@dataclass(frozen=True)
class PreflightReport:
    """Pure IR for one ordered pipeline execution."""

    results: tuple[PreflightResult, ...] = ()

    @property
    def issues(self) -> tuple[PreflightIssue, ...]:
        return tuple(
            issue
            for result in self.results
            for issue in result.issues
        )

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def repairable(self) -> bool:
        issues = self.issues
        return bool(issues) and all(issue.repairable for issue in issues)

    @property
    def failed_stage(self) -> str | None:
        return next(
            (result.stage for result in self.results if not result.ok),
            None,
        )

    @property
    def elapsed_s(self) -> float:
        return sum(result.elapsed_s for result in self.results)


@dataclass(frozen=True)
class PreflightContext:
    """Context available to universal and task-specific validators."""

    parent: Candidate
    operator: str
    workdir: Path


@runtime_checkable
class PreflightValidator(Protocol):
    """Fast local check executed before the expensive Grader."""

    name: str

    def validate(
        self,
        ctx: PreflightContext,
    ) -> PreflightResult:
        ...


@dataclass(frozen=True)
class ProposalCheckResult:
    """Authoritative result of checking one materialized proposal."""

    child_workspace: Workspace | None
    report: PreflightReport

    @property
    def ok(self) -> bool:
        return self.child_workspace is not None and self.report.ok

    def __post_init__(self) -> None:
        if self.report.ok != (self.child_workspace is not None):
            raise ValueError(
                "child_workspace must be present exactly when preflight passes"
            )


class PreflightPipeline:
    """Run validators in dependency order and stop at the first failure."""

    def __init__(self, validators: Iterable[PreflightValidator] = ()):
        self.validators = tuple(validators)
        names = [validator.name for validator in self.validators]
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("preflight validator names must be non-empty strings")
        if len(names) != len(set(names)):
            raise ValueError("preflight validator names must be unique")

    def run(self, ctx: PreflightContext) -> PreflightReport:
        results: list[PreflightResult] = []

        for validator in self.validators:
            result = self._run_validator(validator, ctx)
            results.append(result)
            if not result.ok:
                break

        return PreflightReport(tuple(results))

    @staticmethod
    def _run_validator(
        validator: PreflightValidator,
        ctx: PreflightContext,
    ) -> PreflightResult:
        started = monotonic()
        try:
            result = validator.validate(ctx)
            if not isinstance(result, PreflightResult):
                raise TypeError(
                    "validate() must return PreflightResult, got "
                    f"{type(result).__name__}"
                )
            if result.stage != validator.name:
                raise ValueError(
                    f"validator {validator.name!r} returned stage "
                    f"{result.stage!r}"
                )
        except Exception as exc:  # validator failures must fail closed
            elapsed_s = max(0.0, monotonic() - started)
            return PreflightResult(
                stage=validator.name,
                issues=(
                    PreflightIssue(
                        validator=validator.name,
                        code="validator-error",
                        message=f"{type(exc).__name__}: {exc}",
                        repairable=False,
                    ),
                ),
                elapsed_s=elapsed_s,
            )

        elapsed_s = max(0.0, monotonic() - started)
        return replace(result, elapsed_s=elapsed_s)


class ProposalPreflight:
    """Capture and validate a materialized proposal workspace."""

    WORKSPACE_STAGE = "workspace"

    def __init__(self, pipeline: PreflightPipeline):
        if any(
            validator.name == self.WORKSPACE_STAGE
            for validator in pipeline.validators
        ):
            raise ValueError("'workspace' reserved for ProposalPreflight")

        self.pipeline = pipeline

    def check(self, ctx: PreflightContext) -> ProposalCheckResult:
        started = monotonic()
        try:
            parent_workspace = ctx.parent.workspace
            child_workspace = parent_workspace.capture_child(ctx.workdir)
        except WorkspaceError as exc:
            return self._fail(
                code="workspace-invalid",
                message=str(exc),
                elapsed_s=max(0.0, monotonic() - started),
            )

        if child_workspace.serialize() == parent_workspace.serialize():
            return self._fail(
                code="no-changes",
                message="No changes were made to this workspace",
                elapsed_s=max(0.0, monotonic() - started),
            )

        workspace_result = PreflightResult(
            stage=self.WORKSPACE_STAGE,
            elapsed_s=max(0.0, monotonic() - started),
        )
        task_report = self.pipeline.run(ctx)
        report = PreflightReport(
            results=(workspace_result, *task_report.results)
        )

        if not task_report.ok:
            return ProposalCheckResult(
                child_workspace=None,
                report=report,
            )
        return ProposalCheckResult(
            child_workspace=child_workspace,
            report=report,
        )

    @classmethod
    def _fail(
        cls,
        *,
        code: str,
        message: str,
        elapsed_s: float,
    ) -> ProposalCheckResult:
        issue = PreflightIssue(
            validator=cls.WORKSPACE_STAGE,
            code=code,
            message=message,
            repairable=True,
        )
        report = PreflightReport(
            results=(
                PreflightResult(
                    stage=cls.WORKSPACE_STAGE,
                    issues=(issue,),
                    elapsed_s=elapsed_s,
                ),
            )
        )
        return ProposalCheckResult(
            child_workspace=None,
            report=report,
        )
