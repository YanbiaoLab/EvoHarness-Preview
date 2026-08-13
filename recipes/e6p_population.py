"""E6p — the full experience stack plus two population-level mechanisms.

`diff recipes/e6p_population.py recipes/e5s_full.py` IS this arm's definition:
everything E5s does, and additionally

* STATE MERGING. The loop's only way of using two candidates at once was to
  show one to the model as inspiration. But candidates that fail DIFFERENT
  items each hold something the other lacks, and where a domain carries
  trained state beside the genome that can be combined outright. Measured on
  the modular-arithmetic domain: the run champion missed one top-tier problem
  and a sibling missed three, with no overlap; a blend of their weights
  solved all four and beat both sources on five independently seeded test
  sets. Seven generations of search had not moved that number.

* ISLAND REVIVAL. Islands exist to keep several attacks alive, and the loop
  could not tell when one had died. On run r15 the third island returned a
  median fitness of 0.23-0.29 for thirty consecutive generations while the
  others reached 0.95 -- roughly a third of a three-day run spent breeding
  from a lineage that never contributed, with nothing able to say so.

Both change WHICH CANDIDATES EXIST rather than how they are ranked, so a run
using them is not comparable to one that does not. That is the reason this is
a separate recipe rather than a flag on E5s: the paired experiment needs an
untouched baseline to sit beside.
"""

from evoharness.core import SearchLoop
from evoharness.evoplus import (
    BehavioralNoveltyPolicy,
    ComplementaryInspiration,
    ExperienceContributor,
    ExperienceStore,
    FeedbackContributor,
    IslandHealthMonitor,
    LessonDirectiveContributor,
    MutationReflector,
    StateMergePlanner,
)

from .behavior_stack import SignatureRecorder
from .common import RecipeContext, assemble

NAME = "e6p"
DESCRIPTION = (
    "experience stack + state merging + island revival "
    "+ complementary inspiration"
)


def build(ctx: RecipeContext) -> SearchLoop:
    policy = BehavioralNoveltyPolicy(
        hamming_threshold=ctx.plus.hamming_threshold,
        duplicate_penalty=ctx.plus.duplicate_penalty,
    )
    xstore = ExperienceStore(ctx.run_dir / "experience.jsonl")
    model = ctx.search.llm_models[0]
    reflector = MutationReflector(
        xstore,
        llm=ctx.llm,
        model=model,
        budget=ctx.budget,
        batch_size=4,
    )
    merge_planner = StateMergePlanner(
        probability=ctx.plus.merge_probability,
        ratio_choices=tuple(ctx.plus.merge_ratios),
        min_gain=ctx.plus.merge_min_gain,
        donor_fitness_floor=ctx.plus.merge_donor_fitness_floor,
    )
    island_health = IslandHealthMonitor(
        patience=ctx.plus.island_patience,
        relative_floor=ctx.plus.island_relative_floor,
        min_generation=ctx.plus.island_min_generation,
        max_restarts_per_island=ctx.plus.island_max_restarts,
    )
    # Deterministic and rng-free: mounting it does not move the seeded
    # random stream, so this arm stays replayable against e5s draws.
    inspiration_policy = ComplementaryInspiration()
    ctx.extras["behavior_policy"] = policy
    ctx.extras["experience_store"] = xstore
    ctx.extras["reflector"] = reflector
    ctx.extras["merge_planner"] = merge_planner
    ctx.extras["island_health"] = island_health
    ctx.extras["inspiration_policy"] = inspiration_policy
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
        # The monitor is an observer AND is held directly by the loop: it
        # learns each island's high-water mark from grading, and acts once a
        # generation when the whole generation is visible.
        observers=[SignatureRecorder(), policy, xstore, reflector, island_health],
        weight_policies=[policy],
        merge_planner=merge_planner,
        island_health=island_health,
        inspiration_policy=inspiration_policy,
    )
