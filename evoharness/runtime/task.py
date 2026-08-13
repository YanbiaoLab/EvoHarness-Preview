"""Resolved runtime view of a frozen TaskSpec."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from evoharness.contracts import (
    ComponentSpec,
    CriterionSpec,
    FeedbackSpec,
    MeasurementSpec,
    TaskSpec,
    WorkspaceSpec,
)
from evoharness.contracts.component import implementation_sha256, qualified_name
from evoharness.core.workspace import FileWorkspace, GitWorkspace, Workspace

from .grading import WorkspaceGradeFn, WorkspaceGradeFnGrader

if TYPE_CHECKING:
    from evoharness.core.agent import Runner
    from evoharness.core.interfaces import Grader
    from evoharness.core.llm import LLMTransport
    from evoharness.core.preflight import PreflightValidator


@dataclass
class ResolvedTask:
    """TaskSpec plus process-local services; not itself a public contract."""

    spec: TaskSpec
    grader: "Grader"
    initial_workspace: Workspace
    default_transport: "LLMTransport | None" = None
    preflight_validators: tuple["PreflightValidator", ...] = ()
    runner: "Runner | None" = None
    extra_seeds: tuple[Workspace, ...] = ()

    @staticmethod
    def _component_config(value: object) -> dict:
        """Return explicit, stable component configuration when available."""

        provider = getattr(value, "component_config", None)
        if provider is not None:
            config = provider() if callable(provider) else provider
            if not isinstance(config, dict):
                raise TypeError("component_config must be a dict")
            return config
        grade_func = getattr(value, "_grade_func", None)
        if callable(grade_func):
            config = {"grade_func": qualified_name(grade_func)}
            source_hash = implementation_sha256(grade_func)
            if source_hash is not None:
                config["grade_func_sha256"] = source_hash
            return config
        return {}

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        version: str,
        grader: "Grader",
        initial_workspace: Workspace,
        domain_prompt: str = "",
        knowledge: tuple[str, ...] = (),
        criterion: CriterionSpec | None = None,
        measurement: MeasurementSpec | None = None,
        feedback: FeedbackSpec | None = None,
        default_transport: "LLMTransport | None" = None,
        preflight_validators: tuple["PreflightValidator", ...] = (),
        runner: "Runner | None" = None,
        extra_seeds: tuple[Workspace, ...] = (),
        success_policy: ComponentSpec | None = None,
        domain_tools: tuple[ComponentSpec, ...] = (),
        metadata_json: str = "{}",
    ) -> "ResolvedTask":
        grader_spec = ComponentSpec.for_object(
            "grader",
            grader,
            version=version,
            config=cls._component_config(grader),
        )
        validator_specs = tuple(
            ComponentSpec.for_object(
                "preflight_validator",
                validator,
                version=version,
                name=(
                    f"{validator.__class__.__module__}."
                    f"{validator.__class__.__qualname__}:{validator.name}"
                ),
                config=cls._component_config(validator),
            )
            for validator in preflight_validators
        )
        tool_specs = tuple(domain_tools)
        if runner is not None:
            tool_specs = (
                *tool_specs,
                ComponentSpec.for_object(
                    "domain_tool",
                    runner,
                    version=version,
                    name=(
                        f"{runner.__class__.__module__}."
                        f"{runner.__class__.__qualname__}:runner"
                    ),
                ),
            )
        spec = TaskSpec(
            task_id=task_id,
            version=version,
            initial_workspace=WorkspaceSpec.from_workspace(initial_workspace),
            criterion=criterion or CriterionSpec("fitness"),
            measurement=measurement or MeasurementSpec("domain-grader"),
            feedback=feedback or FeedbackSpec(),
            grader=grader_spec,
            domain_prompt=domain_prompt,
            knowledge=knowledge,
            preflight_validators=validator_specs,
            success_policy=success_policy,
            domain_tools=tool_specs,
            extra_seeds=tuple(
                WorkspaceSpec.from_workspace(seed) for seed in extra_seeds
            ),
            metadata_json=metadata_json,
        )
        return cls(
            spec=spec,
            grader=grader,
            initial_workspace=initial_workspace,
            default_transport=default_transport,
            preflight_validators=preflight_validators,
            runner=runner,
            extra_seeds=extra_seeds,
        )

    @classmethod
    def from_source(
        cls,
        *,
        task_id: str,
        version: str,
        source: str,
        grader: "Grader",
        filename: str = "main.py",
        **kwargs,
    ) -> "ResolvedTask":
        return cls.create(
            task_id=task_id,
            version=version,
            grader=grader,
            initial_workspace=FileWorkspace(source, filename=filename),
            **kwargs,
        )

    @classmethod
    def from_directory(
        cls,
        seed_dir: Path,
        grade_func: WorkspaceGradeFn,
        *,
        task_id: str,
        version: str,
        main_file: str = "main.py",
        include_files: Collection[str] | None = None,
        **kwargs,
    ) -> "ResolvedTask":
        """Resolve a multi-file seed and workspace grader into a TaskSpec."""

        workspace = GitWorkspace.from_directory(
            Path(seed_dir),
            main_file=main_file,
            include_files=include_files,
        )
        return cls.create(
            task_id=task_id,
            version=version,
            grader=WorkspaceGradeFnGrader(grade_func),
            initial_workspace=workspace,
            **kwargs,
        )

    @property
    def task_sys_msg(self) -> str:
        return self.spec.domain_prompt

    @property
    def research_brief(self) -> str:
        return self.spec.research_brief

    @property
    def initial_code(self) -> str:
        return self.initial_workspace.main_text()

    @property
    def transport(self):
        return self.default_transport

    @transport.setter
    def transport(self, value) -> None:
        self.default_transport = value
