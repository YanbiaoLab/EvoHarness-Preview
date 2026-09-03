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

#: The same route, but carrying an import. `import Init` is real -- bare `lean`
#: resolves it -- so a file that repeats it mid-way fails exactly the way a
#: Mathlib one does, and the check does not depend on a lake environment.
NESTED_PREAMBLE = "import Init"

NESTED_ROOT_SKETCH = Sketch(
    parent_name=FIXTURE_SKETCH.parent_name,
    parent_signature=FIXTURE_SKETCH.parent_signature,
    parent_body=FIXTURE_SKETCH.parent_body,
    subgoals=FIXTURE_SKETCH.subgoals,
    preamble=NESTED_PREAMBLE,
)

#: The second level: `lemma_distrib` is not proved directly but decomposed
#: again. Until this existed, every graph anyone built was one level deep, and
#: the branch in `assemble` that splices a subtree had never run -- not in a
#: session and not in a test.
NESTED_CHILD_SKETCH = Sketch(
    parent_name="lemma_distrib",
    parent_signature=FIXTURE_SKETCH.subgoals[0].signature,
    parent_body="lemma_distrib_step a b c",
    subgoals=(
        SubgoalSpec(
            name="lemma_distrib_step",
            identity="text:lemma_distrib_step",
            signature=(
                "theorem lemma_distrib_step (a b c : Nat) : "
                "(a + b) * c = a * c + b * c"
            ),
        ),
    ),
    preamble=NESTED_PREAMBLE,
)

NESTED_PROOFS = {
    "text:lemma_distrib_step": "Nat.add_mul a b c",
    "text:lemma_assoc": FIXTURE_PROOFS["text:lemma_assoc"],
    "text:lemma_mul_zero": FIXTURE_PROOFS["text:lemma_mul_zero"],
}

#: A second route for the same root, shaped like the one PB-Basic-008 needed:
#: flat, citing an already-proved lemma and closing the other two conjuncts
#: inline. Deliberately NOT a copy of the first -- two identical routes would
#: render identically, and a test could not then tell which one was used.
ALTERNATE_ROOT_SKETCH = Sketch(
    parent_name=FIXTURE_SKETCH.parent_name,
    parent_signature=FIXTURE_SKETCH.parent_signature,
    parent_body=(
        "⟨lemma_distrib a b c, Nat.add_assoc a b c, Nat.mul_zero a⟩"
    ),
    subgoals=(FIXTURE_SKETCH.subgoals[0],),
    preamble=NESTED_PREAMBLE,
)

#: A sketch that compiles but leaves `sorry` in the parent body: it moves the
#: goal instead of reducing it, and only the sorry-position check catches it.
LAZY_SKETCH = Sketch(
    parent_name="fixture_main",
    parent_signature=FIXTURE_SIGNATURE,
    parent_body="by\n  refine ⟨lemma_distrib a b c, ?_, ?_⟩ <;> sorry",
    subgoals=(FIXTURE_SKETCH.subgoals[0],),
)
