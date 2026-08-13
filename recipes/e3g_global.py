"""E3g — E2 + experience store in GLOBAL mode (upstream-style control arm).

Diff vs E2: one LLM-distilled cheatsheet shared by all mutations, refreshed
every plus.experience_interval generations — reproducing the upstream
meta-recommendation flow as the comparison arm for E3r.
"""

from evoharness.core import SearchLoop
from evoharness.evoplus import (
    BehavioralNoveltyPolicy,
    ExperienceContributor,
    ExperienceStore,
    FeedbackContributor,
)

from .behavior_stack import SignatureRecorder
from .common import RecipeContext, assemble

NAME = "e3g"
DESCRIPTION = "E2 + experience cheatsheet, global mode (upstream-style arm)"


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
                mode="global",
                llm=ctx.llm,
                model=ctx.search.llm_models[0],
                interval=ctx.plus.experience_interval,
                max_bytes=ctx.plus.experience_max_bytes,
            ),
        ],
        observers=[SignatureRecorder(), policy, xstore],
        weight_policies=[policy],
    )
