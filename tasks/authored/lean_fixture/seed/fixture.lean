-- EvoHarness P-0a fixture.
--
-- Core Lean 4 only, no Mathlib import: a full check costs about a second, so
-- the development loop stays interactive. That is the whole point of this
-- task -- it exists to exercise the plumbing (grading, three states, graph
-- propagation in P-3), not to be hard.
--
-- The goal is three independent conjuncts on purpose. P-3 needs a goal with
-- an obvious three-lemma AND-decomposition to test graph propagation against,
-- and each conjunct closes in one line from core `Nat` lemmas.
--
-- Everything between the EDIT-REGION markers is yours. The LOCKED FOOTER is
-- not: it restates the goal and applies your theorem to it, so weakening
-- `fixture_main` does not make the file compile -- it makes the footer fail
-- to typecheck. The proposition is pinned by Lean, not by comparing bytes.

-- EDIT-REGION-BEGIN
theorem fixture_main (a b c : Nat) :
    (a + b) * c = a * c + b * c
    ∧ (a + b) + c = a + (b + c)
    ∧ a * 0 = 0 := by
  sorry
-- EDIT-REGION-END

-- LOCKED FOOTER: do not edit or delete.
example : ∀ (a b c : Nat),
    (a + b) * c = a * c + b * c
    ∧ (a + b) + c = a + (b + c)
    ∧ a * 0 = 0 := fixture_main

#print axioms fixture_main
