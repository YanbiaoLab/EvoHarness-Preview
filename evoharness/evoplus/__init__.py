"""evoplus: EvoHarness research extensions (C1 feedback, C2 behavior, C3
experience), plugged into evocore via its extension interfaces."""

from .bandit import OperatorBandit
from .behavior import (
    BehavioralNoveltyPolicy,
    CoverageStats,
    RegressionSoftPenalty,
    SignatureRecorder,
)
from .brief import StaticBriefContributor
from .capability import CapabilityLedger
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
    LessonDirectiveContributor,
)
from .feedback import (
    BehaviorSignature,
    FeedbackContributor,
    ItemResult,
    StructuredFeedback,
)
from .inspiration import ComplementaryInspiration
from .islands import IslandHealth, IslandHealthMonitor, IslandRestart
from .merge import (
    Complementarity,
    MergePlan,
    StateMergePlanner,
    complementarity,
)
from .reflection import MutationReflector

__all__ = [
    "BehaviorSignature",
    "BehavioralNoveltyPolicy",
    "CapabilityLedger",
    "Cheatsheet",
    "Complementarity",
    "ComplementaryInspiration",
    "CoverageStats",
    "ExperienceContributor",
    "ExperienceEntry",
    "Directive",
    "DirectiveBook",
    "ExperienceStore",
    "FeedbackContributor",
    "HumanDirectiveContributor",
    "IslandHealth",
    "IslandHealthMonitor",
    "IslandRestart",
    "LessonDirectiveContributor",
    "LineageVetoPolicy",
    "MergePlan",
    "MutationReflector",
    "OperatorBandit",
    "RegressionSoftPenalty",
    "StateMergePlanner",
    "append_directive",
    "complementarity",
    "ItemResult",
    "SignatureRecorder",
    "StaticBriefContributor",
    "StructuredFeedback",
]
