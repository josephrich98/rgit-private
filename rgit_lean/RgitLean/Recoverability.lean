/-
Copyright (c) 2026 Joseph Rich. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Joseph Rich
-/
import RgitLean.RankLimit

/-!
# Recoverability score and the canonical identity `R = ρ²`

This file covers the **Recoverability score** definition (`main.tex` eq. `Rv`),
the explained-covariance identity (`main.tex` eq. `explained`), and the scalar
identities underlying **Proposition 4 (Canonical decomposition)** — in
particular the boxed identity

`R(aᵢ) = 1 − Var(Sᵢ ∣ X) = 1 − 1/(1+dᵢ²) = dᵢ²/(1+dᵢ²) = ρᵢ²`  (`main.tex` eq. `R-rho`).

The spectral content of Proposition 4 (existence of the SVD `M = U D Vᵀ` and the
per-direction posterior variance `Var(Sᵢ ∣ X) = 1/(1+dᵢ²)`) rests on the
classical singular-value decomposition cited in the manuscript; granting
`Var(Sᵢ ∣ X) = 1/(1+dᵢ²)` and the canonical correlation
`ρᵢ = dᵢ/√(1+dᵢ²)`, the boxed chain of equalities is the elementary real-number
computation verified here as `recoverability_canonical`.

`Rscore_eq` records the algebraic identity that the recoverability score equals
the population `R²` of the explained covariance, and `explained_eq_cca` records
that the explained covariance is the CCA form `Σ_gx Σ_x⁻¹ Σ_xg`.
-/

namespace RgitLean

open Matrix

variable {p d : ℕ}
variable (Sg : Matrix (Fin p) (Fin p) ℝ) (A : Matrix (Fin d) (Fin p) ℝ) (σ2 : ℝ)

/-- **Recoverability score** (`main.tex` eq. `Rv`):
`R(v) = 1 − (vᵀ Σ_{g|x} v)/(vᵀ Σ_g v)`. -/
noncomputable def Rscore (v : Fin p → ℝ) : ℝ :=
  1 - (v ⬝ᵥ (Sgcx Sg A σ2 *ᵥ v)) / (v ⬝ᵥ (Sg *ᵥ v))

/-- **Recoverability score, explained form** (`main.tex` eq. `Rv`, middle term):
`R(v) = (vᵀ (Σ_g − Σ_{g|x}) v)/(vᵀ Σ_g v)`, the population `R²` of the
Bayes-optimal predictor of `vᵀG` from `X`. -/
theorem Rscore_eq (v : Fin p → ℝ) (hv : v ⬝ᵥ (Sg *ᵥ v) ≠ 0) :
    Rscore Sg A σ2 v = (v ⬝ᵥ (Explained Sg A σ2 *ᵥ v)) / (v ⬝ᵥ (Sg *ᵥ v)) := by
  unfold Rscore Explained
  rw [Matrix.sub_mulVec, dotProduct_sub]
  field_simp

/-- **Explained covariance is the CCA form** (`main.tex` eq. `explained`):
for symmetric `Σ_g`, `Σ_g − Σ_{g|x} = Σ_gx Σ_x⁻¹ Σ_xg`. -/
theorem explained_eq_cca (hsymm : Sgᵀ = Sg) :
    Explained Sg A σ2 = Sgx Sg A * (Sx Sg A σ2)⁻¹ * (Sgx Sg A)ᵀ := by
  rw [explained_eq, Sgx_def, Matrix.transpose_mul, Matrix.transpose_transpose, hsymm]

/-! ### Scalar identities for Proposition 4 (`R = ρ²`) -/

variable (di : ℝ)

/-- `1 + dᵢ² > 0`. -/
private lemma one_add_sq_pos : (0 : ℝ) < 1 + di ^ 2 := by positivity

/-- Posterior-variance retention vs. recoverability:
`1 − 1/(1+dᵢ²) = dᵢ²/(1+dᵢ²)`. -/
theorem recoverability_eq : 1 - 1 / (1 + di ^ 2) = di ^ 2 / (1 + di ^ 2) := by
  have h := (one_add_sq_pos di).ne'
  field_simp
  ring

/-- Squared canonical correlation: with `ρᵢ = dᵢ/√(1+dᵢ²)`,
`ρᵢ² = dᵢ²/(1+dᵢ²)`. -/
theorem rho_sq : (di / Real.sqrt (1 + di ^ 2)) ^ 2 = di ^ 2 / (1 + di ^ 2) := by
  rw [div_pow, Real.sq_sqrt (one_add_sq_pos di).le]

/-- The complement `1 − ρᵢ² = 1/(1+dᵢ²)` (posterior variance retained). -/
theorem one_sub_rho_sq : 1 - (di / Real.sqrt (1 + di ^ 2)) ^ 2 = 1 / (1 + di ^ 2) := by
  have h := (one_add_sq_pos di).ne'
  rw [rho_sq]
  field_simp
  ring

/-- **Proposition 4, boxed identity** (`main.tex` eq. `R-rho`):
`R(aᵢ) = 1 − 1/(1+dᵢ²) = dᵢ²/(1+dᵢ²) = ρᵢ²`, where `ρᵢ = dᵢ/√(1+dᵢ²)`.
Granting the classical `Var(Sᵢ ∣ X) = 1/(1+dᵢ²)`, the left side is the
recoverability of the `i`-th canonical genomic direction. -/
theorem recoverability_canonical :
    1 - 1 / (1 + di ^ 2) = (di / Real.sqrt (1 + di ^ 2)) ^ 2 := by
  rw [recoverability_eq, rho_sq]

/-- The inverse relation `dᵢ² = ρᵢ²/(1−ρᵢ²)` (`main.tex` Prop. 4(ii)). -/
theorem di_sq_eq_rho :
    di ^ 2 = (di / Real.sqrt (1 + di ^ 2)) ^ 2 / (1 - (di / Real.sqrt (1 + di ^ 2)) ^ 2) := by
  have h := (one_add_sq_pos di).ne'
  rw [one_sub_rho_sq, rho_sq]
  field_simp

end RgitLean
