import Mathlib

-- EDIT-REGION-BEGIN
theorem pbadvanced006_negcase_finite_range (f : ℤ → ℤ) (h0 : f 0 = 0) (hshift : ∀ u v c, f u = f v → f (u + c) = f (v + c)) (a : ℤ) (ha0 : f (a - 2) = 0) (ha2 : a ≠ 2) : (Set.range f).Finite := by
  let p : ℤ := a - 2
  have hp : p ≠ 0 := sub_ne_zero.mpr ha2
  have hper : Function.Periodic f p := by
    intro x
    simpa [p, add_comm] using hshift (a - 2) 0 x (ha0.trans h0.symm)
  let g : Fin p.natAbs → ℤ := fun r => f r
  refine Set.Finite.subset (Set.finite_range g) ?_
  rintro y ⟨x, rfl⟩
  let r : Fin p.natAbs := ⟨(x % p).natAbs, by
    rw [Int.natAbs_lt]
    exact ⟨Int.emod_nonneg x hp, Int.emod_lt_of_pos x (Int.natAbs_pos.mp hp)⟩⟩
  refine ⟨g r, ⟨r, rfl⟩, ?_⟩
  dsimp [g, r]
  have hm := hper.zsmul (x / p) (x % p)
  rw [zsmul_eq_mul] at hm
  rw [← hm]
  congr 1
  rw [Int.emod_add_ediv]
  ring
-- EDIT-REGION-END

#print axioms pbadvanced006_negcase_finite_range