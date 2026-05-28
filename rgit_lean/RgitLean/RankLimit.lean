/-
Copyright (c) 2026 Joseph Rich. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Joseph Rich
-/
import RgitLean.Posterior

/-!
# Proposition 6 — Rank limit (the recoverable dimension is capped)

`main.tex` **Proposition 6 (Rank limit)** states
`rank(Σ_g − Σ_{g|x}) = rank(A Σ_g Aᵀ) ≤ min(d, p, rank A, rank Σ_g)`,
so at most `min(d, p, rank A, rank Σ_g)` canonical correlations are nonzero.

Here we machine-check the conceptual core — the **upper bound**

`rank(Σ_g − Σ_{g|x}) ≤ min(d, p, rank A, rank Σ_g)` —

which says the image-identifiable genomic subspace has dimension capped by the
biology–imaging channel, not the number of measured genes.  The recoverable
matrix is `Σ_g − Σ_{g|x} = Σ_g Aᵀ Σ_x⁻¹ A Σ_g` (definitionally, from
`Sgcx`), a product routed through the `d`-dimensional imaging space, and rank
sub-multiplicativity does the rest.
-/

namespace RgitLean

open Matrix

variable {p d : ℕ}
variable (Sg : Matrix (Fin p) (Fin p) ℝ) (A : Matrix (Fin d) (Fin p) ℝ) (σ2 : ℝ)

/-- The recoverable ("explained") covariance is `Σ_g Aᵀ Σ_x⁻¹ A Σ_g`
(`main.tex` eq. `explained`). -/
lemma explained_eq :
    Explained Sg A σ2 = Sg * Aᵀ * (Sx Sg A σ2)⁻¹ * (A * Sg) := by
  unfold Explained Sgcx
  abel

theorem rank_explained_le_rankA : (Explained Sg A σ2).rank ≤ A.rank := by
  rw [explained_eq]
  calc (Sg * Aᵀ * (Sx Sg A σ2)⁻¹ * (A * Sg)).rank
      ≤ (A * Sg).rank := rank_mul_le_right _ _
    _ ≤ A.rank := rank_mul_le_left _ _

theorem rank_explained_le_rankSg : (Explained Sg A σ2).rank ≤ Sg.rank := by
  rw [explained_eq]
  calc (Sg * Aᵀ * (Sx Sg A σ2)⁻¹ * (A * Sg)).rank
      ≤ (Sg * Aᵀ * (Sx Sg A σ2)⁻¹).rank := rank_mul_le_left _ _
    _ ≤ (Sg * Aᵀ).rank := rank_mul_le_left _ _
    _ ≤ Sg.rank := rank_mul_le_left _ _

theorem rank_explained_le_d : (Explained Sg A σ2).rank ≤ d := by
  rw [explained_eq]
  calc (Sg * Aᵀ * (Sx Sg A σ2)⁻¹ * (A * Sg)).rank
      ≤ (Sg * Aᵀ * (Sx Sg A σ2)⁻¹).rank := rank_mul_le_left _ _
    _ ≤ Fintype.card (Fin d) := rank_le_card_width _
    _ = d := Fintype.card_fin d

theorem rank_explained_le_p : (Explained Sg A σ2).rank ≤ p := by
  calc (Explained Sg A σ2).rank
      ≤ Fintype.card (Fin p) := rank_le_card_width _
    _ = p := Fintype.card_fin p

/-- **Proposition 6, rank limit** (`main.tex`):
`rank(Σ_g − Σ_{g|x}) ≤ min(d, p, rank A, rank Σ_g)`. -/
theorem rank_limit :
    (Explained Sg A σ2).rank ≤ min (min d p) (min A.rank Sg.rank) :=
  le_min
    (le_min (rank_explained_le_d Sg A σ2) (rank_explained_le_p Sg A σ2))
    (le_min (rank_explained_le_rankA Sg A σ2) (rank_explained_le_rankSg Sg A σ2))

end RgitLean
