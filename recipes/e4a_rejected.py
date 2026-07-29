"""E4a — E3r plus the negative-experience section.

Diff vs E3r: ExperienceContributor runs in "retrieval+rejected" mode, so
proposals also see recent pre-evaluation rejections (novelty duplicates,
failed proposals) and unredeemed regressions from the sliding window,
same-island first. E4a - E3r isolates the value of negative experience.

Why this graduated from the experiment matrix to a production recipe
(2026-07-29, modmul r12): in one generation, four proposals independently
raised the same constant and died identically, and four more independently
invented the same padding scheme and died identically — eight of sixteen
mutations were re-purchases of two already-bought lessons. The negative
section is where those lessons reach the next proposal.
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

NAME = "e4a"
DESCRIPTION = "E3r + negative experience (rejections and regressions)"


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
                mode="retrieval+rejected",
                max_bytes=ctx.plus.experience_max_bytes,
                top_m=ctx.plus.experience_top_m,
                top_n=ctx.plus.experience_top_n,
            ),
        ],
        observers=[SignatureRecorder(), policy, xstore],
        weight_policies=[policy],
    )
