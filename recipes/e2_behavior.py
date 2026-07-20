"""E2 — E1 + behavioral-signature novelty (plan C2).

Diff vs E1: BehavioralNoveltyPolicy marks behavior duplicates (excluded from
the archive) and down-weights them in parent selection.
"""

from evoharness.evocore import SearchLoop
from evoharness.evoplus import BehavioralNoveltyPolicy, FeedbackContributor

from .behavior_stack import SignatureRecorder
from .common import RecipeContext, assemble

NAME = "e2"
DESCRIPTION = "E1 + behavioral novelty (C2)"


def build(ctx: RecipeContext) -> SearchLoop:
    policy = BehavioralNoveltyPolicy(
        hamming_threshold=ctx.plus.hamming_threshold,
        duplicate_penalty=ctx.plus.duplicate_penalty,
    )
    ctx.extras["behavior_policy"] = policy
    return assemble(
        ctx,
        contributors=[
            FeedbackContributor(
                top_k=ctx.plus.feedback_top_k,
                examples_per_cat=ctx.plus.feedback_examples_per_cat,
            )
        ],
        observers=[SignatureRecorder(), policy],  # recorder must precede policy
        weight_policies=[policy],
    )
