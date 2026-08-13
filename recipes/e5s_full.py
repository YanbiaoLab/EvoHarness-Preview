"""E5s — the full three-layer experience stack, as a registry recipe.

Diff vs E4a: adds L1 batched LLM attribution (MutationReflector writes
verdict/why/advice/tags back onto evidence rows), retrieval switches from
the mechanical ledger to lessons matched on the parent's failure tags, one
sampled scratchpad direction rides along, and a qualifying lesson is
promoted to the mutation's explicit directive with probability 0.5.

This stack has been running on the IMO line (experiments/imo_proof/
evolution.py) since the E5 experiments; what was missing was a registry
recipe, so every recipes/-driven domain — modmul included — silently got
the mechanical ledger while the full stack sat one import away. The
wiring below is that entry point's, ported verbatim where it applies.

Order is load-bearing twice: SignatureRecorder must precede the policy
that reads its output, and the store must record a graded entry before
the reflector counts pending work.
"""

from evoharness.core import SearchLoop
from evoharness.evoplus import (
    BehavioralNoveltyPolicy,
    ExperienceContributor,
    ExperienceStore,
    FeedbackContributor,
    LessonDirectiveContributor,
    MutationReflector,
)

from .behavior_stack import SignatureRecorder
from .common import RecipeContext, assemble

NAME = "e5s"
DESCRIPTION = "full experience stack: lessons + scratchpad + directive"


def build(ctx: RecipeContext) -> SearchLoop:
    policy = BehavioralNoveltyPolicy(
        hamming_threshold=ctx.plus.hamming_threshold,
        duplicate_penalty=ctx.plus.duplicate_penalty,
    )
    xstore = ExperienceStore(ctx.run_dir / "experience.jsonl")
    model = ctx.search.llm_models[0]
    # batch_size=4, per the IMO entry's measured note: with the default 8,
    # the first lessons arrive too late to inform most of a small run.
    reflector = MutationReflector(
        xstore,
        llm=ctx.llm,
        model=model,
        budget=ctx.budget,
        batch_size=4,
    )
    ctx.extras["behavior_policy"] = policy
    ctx.extras["experience_store"] = xstore
    ctx.extras["reflector"] = reflector
    return assemble(
        ctx,
        contributors=[
            LessonDirectiveContributor(xstore),
            FeedbackContributor(
                top_k=ctx.plus.feedback_top_k,
                examples_per_cat=ctx.plus.feedback_examples_per_cat,
            ),
            ExperienceContributor(
                xstore,
                mode="lessons+scratchpad",
                llm=ctx.llm,
                model=model,
                reflector=reflector,
                max_bytes=ctx.plus.experience_max_bytes,
                top_m=ctx.plus.experience_top_m,
                top_n=ctx.plus.experience_top_n,
            ),
        ],
        observers=[SignatureRecorder(), policy, xstore, reflector],
        weight_policies=[policy],
    )
