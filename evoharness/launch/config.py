"""Everything a run needs to know about itself, without a command line.

The assembly that turns configuration into a `SearchLoop` used to read an
`argparse.Namespace` directly, which tied it to the one caller that had a
parser. A detached run started from a tool call has no parser, and neither
does a resume — but both need exactly the same values, which is why they live
in a dataclass that can round-trip through the run directory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


class LaunchConfigError(ValueError):
    """The configuration cannot produce a runnable search."""


def _path(value) -> Path | None:
    return None if value is None else Path(value)


@dataclass(frozen=True)
class LaunchConfig:
    """One run's invocation, independent of how it was requested."""

    recipe: str
    run_dir: Path
    task: str = "demo_counter"
    #: An authored task directory, used instead of the registry entry named by
    #: `task` when present.
    task_dir: Path | None = None
    config_path: Path | None = None
    budget_usd: float | None = None
    brief: Path | None = None
    live: bool = False
    dsh_config: Path | None = None
    dsh_runtime: Path | None = None
    #: The provider route the dsh config declares. It has to match, because
    #: the runtime resolves a request's model against that route's catalog and
    #: a name it does not know fails the request rather than falling back.
    dsh_provider: str = "deepseek-official"
    overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_dir", Path(self.run_dir))
        for name in ("task_dir", "config_path", "brief", "dsh_config", "dsh_runtime"):
            object.__setattr__(self, name, _path(getattr(self, name)))
        object.__setattr__(self, "overrides", tuple(self.overrides))
        if not self.recipe.strip():
            raise LaunchConfigError("recipe must be non-empty")
        if not self.dsh_provider.strip():
            raise LaunchConfigError("dsh_provider must be non-empty")
        # Both or neither: a dsh config without its runtime entry silently
        # falls back to the in-process agent, which is a different experiment
        # wearing the same name.
        if (self.dsh_config is None) != (self.dsh_runtime is None):
            raise LaunchConfigError(
                "dsh_config and dsh_runtime must be given together"
            )

    @property
    def task_label(self) -> str:
        """How this run names its task in the manifest."""

        return f"dir:{self.task_dir}" if self.task_dir is not None else self.task

    def to_json(self) -> dict:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, Path):
                payload[key] = str(value)
        payload["overrides"] = list(self.overrides)
        return payload

    @classmethod
    def from_json(cls, payload: dict) -> "LaunchConfig":
        known = {field_name for field_name in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            # A key nobody reads is a setting the author believes is in
            # effect. Silently dropping it is how a run ends up not being the
            # experiment its record claims.
            raise LaunchConfigError(
                f"unknown launch settings: {sorted(unknown)}"
            )
        return cls(**payload)
