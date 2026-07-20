"""E3r — the full system: E2 + experience store in RETRIEVAL mode.

Diff vs E3g: experience is retrieved per-mutation by the parent's top error
categories instead of one shared global cheatsheet. E3r - E3g isolates the
value of failure-mode routing, this project's core C3 claim.
"""

from evoharness.evocore import SearchLoop
from evoharness.evoplus import (
    BehavioralNoveltyPolicy,
    ExperienceContributor,
    ExperienceStore,
    FeedbackContributor,
)

from .behavior_stack import SignatureRecorder
from .common import RecipeContext, assemble

NAME = "e3r"
DESCRIPTION = "full system: E2 + failure-mode-retrieved experience"


def build(ctx: RecipeContext) -> SearchLoop:
    policy = BehavioralNoveltyPolicy(
        hamming_threshold=ctx.plus.hamming_threshold,
        duplicate_penalty=ctx.plus.duplicate_penalty,
    )
    xstore = ExperienceStore(ctx.run_dir / "experience.jsonl")
    ctx.extras["behavior_policy"] = policy
    ctx.extras["experience_store"] = xstore
    return assemble(
        ctx,
        contributors=[
            FeedbackContributor(
                top_k=ctx.plus.feedback_top_k,
                examples_per_cat=ctx.plus.feedback_examples_per_cat,
            ),
            ExperienceContributor(
                xstore,
                mode="retrieval",
                max_bytes=ctx.plus.experience_max_bytes,
                top_m=ctx.plus.experience_top_m,
                top_n=ctx.plus.experience_top_n,
            ),
        ],
        observers=[SignatureRecorder(), policy, xstore],
        weight_policies=[policy],
    )
