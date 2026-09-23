"""Which axioms a finished proof may rest on. One object, not two copied sets.

There used to be two `ALLOWED_AXIOMS` sets, one in `grade.py` and one in
`assembly.py`, identical and independent. Changing the policy in one place and
not the other is how the per-candidate grader and the final certificate come to
disagree about the same file.

The classification mirrors the verifier's trust classification, four levels,
lowest axiom wins:

    trusted  3   propext, Quot.sound, Classical.choice -- Mathlib's foundations
    audited  2   Lean.ofReduceBool / ofReduceNat / trustCompiler, and
                 native_decide axioms whose content has been rechecked
    claimed  1   anything else, including every user-declared `axiom`
    tainted  0   sorryAx

The two implementations have to agree case by case. The cases are part of
the verifier's contract; this repo keeps a pinned copy in
`tests/fixtures/trust_classification.json` and tests against it.

**A native_decide axiom is not trusted by its name.** In Lean v4.33 the tactic
adds one axiom per declaration, named `<decl>._native.native_decide.ax_N`. The
name shape alone does not show that the asserted `Bool` really evaluates to
`true`; only a content recheck does, and that is the verifier's job, reported
per axiom. This side has no Lean to recheck with, so without a positive
recheck result a native_decide axiom counts as `claimed`: an unverifiable
proof is not a proof, it fails closed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

#: Bumped whenever the classification below changes. Recorded in a graph's
#: scope, because a PROVED node earned under one version need not hold under
#: another. Matches `policy_version` in the shared fixture.
POLICY_VERSION = 1

LEVELS: Mapping[str, int] = {"tainted": 0, "claimed": 1, "audited": 2, "trusted": 3}

STANDARD_AXIOMS = frozenset({"propext", "Quot.sound", "Classical.choice"})

#: Real constants in Lean's core library. A candidate cannot redeclare them --
#: the names are taken -- so unlike the native_decide axioms, their names can
#: be believed.
COMPILER_AXIOMS = frozenset(
    {"Lean.ofReduceBool", "Lean.ofReduceNat", "Lean.trustCompiler"}
)

SORRY = "sorryAx"


def is_native_decide_axiom(name: str) -> bool:
    """The name shape of a v4.33 native_decide axiom. A candidate, not a verdict."""

    parts = name.split(".")
    return "_native" in parts and "native_decide" in parts


@dataclass(frozen=True)
class AxiomPolicy:
    """The lowest trust level a proof may have and still count as proved.

    `minimum_trust` takes the same three values as the verifier's own trust
    floor. `tainted` is not among them: no policy may accept `sorry`.
    """

    minimum_trust: str = "audited"

    def __post_init__(self) -> None:
        if self.minimum_trust not in ("trusted", "audited", "claimed"):
            raise ValueError(f"unknown minimum_trust={self.minimum_trust!r}")

    def level_of(
        self, axiom: str, native_holds: Mapping[str, bool] | None = None
    ) -> int:
        if axiom in STANDARD_AXIOMS:
            return LEVELS["trusted"]
        if axiom in COMPILER_AXIOMS:
            return LEVELS["audited"]
        if is_native_decide_axiom(axiom):
            # Only a recheck that ran and said yes. Absent means nobody could
            # check, and that is not the same as checked.
            held = (native_holds or {}).get(axiom) is True
            return LEVELS["audited"] if held else LEVELS["claimed"]
        if axiom == SORRY:
            return LEVELS["tainted"]
        return LEVELS["claimed"]

    def classify(
        self,
        axioms: Iterable[str],
        native_holds: Mapping[str, bool] | None = None,
    ) -> str:
        level = min(
            (self.level_of(a, native_holds) for a in axioms),
            default=LEVELS["trusted"],
        )
        return next(name for name, value in LEVELS.items() if value == level)

    def accepts(self, trust: str) -> bool:
        return LEVELS[trust] >= LEVELS[self.minimum_trust]

    def blocking(
        self,
        axioms: Iterable[str],
        native_holds: Mapping[str, bool] | None = None,
    ) -> frozenset[str]:
        """Every axiom that on its own puts the proof below the policy."""

        floor = LEVELS[self.minimum_trust]
        return frozenset(
            a for a in axioms if self.level_of(a, native_holds) < floor
        )

    def unverified_native(
        self,
        axioms: Iterable[str],
        native_holds: Mapping[str, bool] | None = None,
    ) -> frozenset[str]:
        """native_decide axioms nobody has shown to hold.

        Kept apart from the rest of `blocking` because the two mean different
        things to a caller: a user-declared axiom is the candidate's doing, an
        unrechecked native_decide axiom may be a perfectly good proof that
        this side simply has no way to confirm.
        """

        return frozenset(
            a
            for a in axioms
            if is_native_decide_axiom(a) and (native_holds or {}).get(a) is not True
        )


__all__ = [
    "COMPILER_AXIOMS",
    "LEVELS",
    "POLICY_VERSION",
    "SORRY",
    "STANDARD_AXIOMS",
    "AxiomPolicy",
    "is_native_decide_axiom",
]
