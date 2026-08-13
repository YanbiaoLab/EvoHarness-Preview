"""E1 — E0 + structured verifier feedback (plan C1).

Diff vs E0: FeedbackContributor injects the parent's failure summary
(top-k error categories + examples) into every mutation prompt.
"""

from evoharness.core import SearchLoop
from evoharness.evoplus import FeedbackContributor

from .behavior_stack import SignatureRecorder
from .common import RecipeContext, assemble

NAME = "e1"
DESCRIPTION = "E0 + structured verifier feedback (C1)"


def build(ctx: RecipeContext) -> SearchLoop:
    return assemble(
        ctx,
        contributors=[
            FeedbackContributor(
                top_k=ctx.plus.feedback_top_k,
                examples_per_cat=ctx.plus.feedback_examples_per_cat,
            )
        ],
        observers=[SignatureRecorder()],
    )
