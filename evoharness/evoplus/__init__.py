"""evoplus: EvoHarness research extensions (C1 feedback, C2 behavior, C3
experience), plugged into evocore via its extension interfaces."""

from .behavior import BehavioralNoveltyPolicy, CoverageStats, SignatureRecorder
from .brief import StaticBriefContributor
from .directives import (
    Directive,
    DirectiveBook,
    HumanDirectiveContributor,
    LineageVetoPolicy,
    append_directive,
)
from .experience import (
    Cheatsheet,
    ExperienceContributor,
    ExperienceEntry,
    ExperienceStore,
)
from .feedback import (
    BehaviorSignature,
    FeedbackContributor,
    ItemResult,
    StructuredFeedback,
)
from .reflection import MutationReflector

__all__ = [
    "BehaviorSignature",
    "BehavioralNoveltyPolicy",
    "Cheatsheet",
    "CoverageStats",
    "ExperienceContributor",
    "ExperienceEntry",
    "Directive",
    "DirectiveBook",
    "ExperienceStore",
    "FeedbackContributor",
    "HumanDirectiveContributor",
    "LineageVetoPolicy",
    "MutationReflector",
    "append_directive",
    "ItemResult",
    "SignatureRecorder",
    "StaticBriefContributor",
    "StructuredFeedback",
]
