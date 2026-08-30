"""The hand-written route P-3 uses in place of a model.

Core Lean only, no Mathlib: the whole chain checks in a couple of seconds, so
the graph tests stay interactive. Three conjuncts, three lemmas, one term that
puts them back together -- verified to compile with `sorry` confined to the
lemmas.
"""

from evoharness.proof.sketch import Sketch, SubgoalSpec

FIXTURE_SIGNATURE = (
    "theorem fixture_main (a b c : Nat) :\n"
    "    (a + b) * c = a * c + b * c\n"
    "    ∧ (a + b) + c = a + (b + c)\n"
    "    ∧ a * 0 = 0"
)

FIXTURE_SKETCH = Sketch(
    parent_name="fixture_main",
    parent_signature=FIXTURE_SIGNATURE,
    parent_body="⟨lemma_distrib a b c, lemma_assoc a b c, lemma_mul_zero a⟩",
    subgoals=(
        SubgoalSpec(
            name="lemma_distrib",
            identity="text:lemma_distrib",
            signature=(
                "theorem lemma_distrib (a b c : Nat) : "
                "(a + b) * c = a * c + b * c"
            ),
        ),
        SubgoalSpec(
            name="lemma_assoc",
            identity="text:lemma_assoc",
            signature=(
                "theorem lemma_assoc (a b c : Nat) : (a + b) + c = a + (b + c)"
            ),
        ),
        SubgoalSpec(
            name="lemma_mul_zero",
            identity="text:lemma_mul_zero",
            signature="theorem lemma_mul_zero (a : Nat) : a * 0 = 0",
        ),
    ),
)

#: What a solver returns when it closes each lemma.
FIXTURE_PROOFS = {
    "text:lemma_distrib": "Nat.add_mul a b c",
    "text:lemma_assoc": "Nat.add_assoc a b c",
    "text:lemma_mul_zero": "Nat.mul_zero a",
}

#: A sketch that compiles but leaves `sorry` in the parent body: it moves the
#: goal instead of reducing it, and only the sorry-position check catches it.
LAZY_SKETCH = Sketch(
    parent_name="fixture_main",
    parent_signature=FIXTURE_SIGNATURE,
    parent_body="by\n  refine ⟨lemma_distrib a b c, ?_, ?_⟩ <;> sorry",
    subgoals=(FIXTURE_SKETCH.subgoals[0],),
)
