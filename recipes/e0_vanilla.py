"""E0 — ported engine as-is: all three extensions off.

Diff vs B0: evolution actually runs (num_generations from config).
"""

from evoharness.evocore import SearchLoop

from .behavior_stack import SignatureRecorder
from .common import RecipeContext, assemble

NAME = "e0"
DESCRIPTION = "evocore vanilla, no extensions"


def build(ctx: RecipeContext) -> SearchLoop:
    return assemble(ctx, observers=[SignatureRecorder()])
