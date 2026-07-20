"""B0 — manual baseline: evaluate the hand-tuned harness once, no evolution.

Implemented as a zero-generation SearchLoop so B0 shares the exact same
grading path, metric log and manifest as every other group.
"""

import dataclasses

from evoharness.evocore import SearchLoop

from .behavior_stack import SignatureRecorder
from .common import RecipeContext, assemble

NAME = "b0"
DESCRIPTION = "manual harness, evaluated once, no evolution"


def build(ctx: RecipeContext) -> SearchLoop:
    ctx = dataclasses.replace(ctx, search=dataclasses.replace(ctx.search, num_generations=0))
    return assemble(ctx, observers=[SignatureRecorder()])
