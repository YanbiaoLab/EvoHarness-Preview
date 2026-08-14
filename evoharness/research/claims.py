from dataclasses import dataclass
from enum import Enum

from evoharness.contracts import spec_hash


class ClaimKind(str, Enum):
    CANDIDATE_VALID = "candidate_valid"
    OBJECTIVE_MET = "objective_met"
    HAS_ANY_GAIN = "has_any_gain"
    BETTER_THAN_REFERENCE = "better_than_reference"
    NO_REGRESSION = "no_regression"


_COMPARATIVE = {
    ClaimKind.HAS_ANY_GAIN,
    ClaimKind.BETTER_THAN_REFERENCE,
    ClaimKind.NO_REGRESSION,
}

@dataclass(frozen=True)
class Claim:
    kind: ClaimKind
    subject_id: str
    reference_id: str = ""
    margin: float = 0.0

    def __post_init__(self) -> None:
        if not self.subject_id.strip():
            raise ValueError("subject_id must be non-empty")
        if self.kind in _COMPARATIVE and not self.reference_id.strip():
            raise ValueError(f"{self.kind.value} requires reference_id")
        if self.margin < 0:
            raise ValueError("margin must be non-negative")


    @property
    def hash(self) -> str:
        return spec_hash(
            {"kind":"claim",
             "claim_kind": self.kind.value,
            "subject_id": self.subject_id,
            "reference_id": self.reference_id,
            "margin": self.margin,
             }
        )

