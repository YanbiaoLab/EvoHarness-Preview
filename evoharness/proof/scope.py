"""What a proof graph was built under, and the refusal to open it under anything else.

A goal node carries its proof status on its own row, and `upsert_goal` hands an
existing node back "with whatever progress it has already accumulated". So a
graph that silently changes its premises between two commands lets a PROVED
earned under one set of premises stand under another. Before this module that
happened with nothing more than a forgotten flag: the identity hasher was chosen
per invocation by `--lean-identity`, and `open --preamble` overwrote the
graph's preamble file. Two hashers' keys are not interchangeable, and a
different preamble changes what the same statement text means.

The rule for every field: **does changing it make an existing PROVED possibly
wrong?** If so it is in the scope, and a mismatch refuses to open the graph.

Deliberately NOT in the scope: the verifier's runtime fingerprint. It changes
whenever the verifier is rebuilt, and putting it here would void every graph on
every rebuild. It is bound per certification instead, which is the right
granularity.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .policy import POLICY_VERSION, AxiomPolicy

#: This table's own shape.
SCOPE_VERSION = 1
#: Version of the source-envelope contract (how a candidate file maps onto what
#: the verifier checks). Owned by the contract, recorded here because changing
#: it changes what a PROVED certifies.
ENVELOPE_SCHEMA_VERSION = 1
#: The verifier's goal-identity schema version: what counts as "the same goal"
#: on the verifier's side.
GOALKEY_SCHEMA_VERSION = 1

DEFAULT_MINIMUM_TRUST = "audited"
DEFAULT_IDENTITY_HASHER = "exact-text"
IDENTITY_HASHERS = ("exact-text", "lean-expr")


class ScopeError(ValueError):
    """The graph cannot be opened under the premises asked for."""


class ScopeMismatch(ScopeError):
    """The graph was built under different premises. Lists every difference."""

    def __init__(self, diffs: list[tuple[str, object, object]]):
        self.diffs = diffs
        lines = [f"{name}: graph has {old!r}, this command has {new!r}"
                 for name, old, new in diffs]
        super().__init__(
            "this graph was built under a different scope; refusing to open it "
            "(a PROVED earned under one scope need not hold under another). "
            + "; ".join(lines)
            + ". Match the graph's settings, or start over with --force-new-graph."
        )


class ScopeMissing(ScopeError):
    """A graph with goals but no recorded scope: built before scopes existed."""


def preamble_sha256(text: str) -> str:
    """Hash of the preamble as the graph uses it: stripped, like every reader."""

    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def local_environment(project: str | Path | None) -> str:
    """The Lean a local runner compiles against, as a scope string.

    A lake project is identified by its manifest (which pins every dependency's
    revision) together with its toolchain file (which pins Lean). A bare `lean`
    is `local:bare`: good enough for the core-only tests it serves, and it does
    not shell out to ask Lean its version on every open -- most commands never
    compile at all.

    Once verification goes through the verifier the environment becomes
    `verifier:<base fingerprint>`, and moving from `local:` to `verifier:` is a
    scope change like any other: a PROVED judged by a local compile is not a
    verifier certification.
    """

    if not project:
        return "local:bare"
    root = Path(project)
    digest = hashlib.sha256()
    for name in ("lake-manifest.json", "lean-toolchain"):
        path = root / name
        digest.update(name.encode() + b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
        digest.update(b"\0")
    return f"local:lake:{digest.hexdigest()}"


@dataclass(frozen=True)
class GraphScope:
    environment: str
    minimum_trust: str = DEFAULT_MINIMUM_TRUST
    identity_hasher: str = DEFAULT_IDENTITY_HASHER
    preamble_sha256: str = preamble_sha256("")
    #: For people only. Comparison goes by `environment`.
    base_id: str = ""
    axiom_policy_version: int = POLICY_VERSION
    envelope_schema_version: int = ENVELOPE_SCHEMA_VERSION
    goalkey_schema_version: int = GOALKEY_SCHEMA_VERSION
    scope_version: int = SCOPE_VERSION

    def __post_init__(self) -> None:
        AxiomPolicy(self.minimum_trust)  # validates the value
        if self.identity_hasher not in IDENTITY_HASHERS:
            raise ValueError(f"unknown identity hasher {self.identity_hasher!r}")

    @property
    def policy(self) -> AxiomPolicy:
        return AxiomPolicy(self.minimum_trust)

    def diff(self, other: "GraphScope") -> list[tuple[str, object, object]]:
        """Every compared field that differs, as (field, self's, other's)."""

        return [
            (f.name, getattr(self, f.name), getattr(other, f.name))
            for f in fields(self)
            if f.name != "base_id"
            and getattr(self, f.name) != getattr(other, f.name)
        ]

    def to_row(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row) -> "GraphScope":
        return cls(**{f.name: row[f.name] for f in fields(cls)})

    @classmethod
    def resolve(
        cls,
        *,
        stored: "GraphScope | None",
        environment: str,
        preamble_sha256: str,
        identity_hasher: str | None = None,
        minimum_trust: str | None = None,
        base_id: str = "",
    ) -> "GraphScope":
        """The scope a command asks for.

        `environment` and `preamble_sha256` are always what the command will
        actually use, so they are always compared. The settings a command only
        states by flag -- the hasher and the trust floor -- are inherited from
        the graph when the flag is left out, and compared when it is given.
        Inheriting is not a courtesy: a command that hashes with the default
        hasher on a `lean-expr` graph is exactly the mixing this module stops.

        The versions are always this code's, so upgrading a policy or a
        contract refuses graphs built under the old one.
        """

        return cls(
            environment=environment,
            minimum_trust=minimum_trust
            or (stored.minimum_trust if stored else DEFAULT_MINIMUM_TRUST),
            identity_hasher=identity_hasher
            or (stored.identity_hasher if stored else DEFAULT_IDENTITY_HASHER),
            preamble_sha256=preamble_sha256,
            base_id=base_id or (stored.base_id if stored else ""),
        )


__all__ = [
    "ENVELOPE_SCHEMA_VERSION",
    "GOALKEY_SCHEMA_VERSION",
    "SCOPE_VERSION",
    "GraphScope",
    "ScopeError",
    "ScopeMismatch",
    "ScopeMissing",
    "local_environment",
    "preamble_sha256",
]
