/-
Copyright (c) 2026 Joseph Rich. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Joseph Rich
-/
import RgitLean.Model

/-!
# Proposition 2 — Bayes-optimal recovery (posterior precision and read-out map)

This file verifies the algebraic identities of `main.tex`
**Proposition 2 (Posterior; standard Gaussian conditioning)**.

The probabilistic step — that Gaussian conditioning gives
`G ∣ X ~ N(B X, Σ_{g|x})` — is the classical Gaussian-conditioning fact cited in
the manuscript.  What the manuscript then *derives* algebraically, and what we
machine-check here, is:

* the **precision form** `Σ_{g|x}⁻¹ = Σ_g⁻¹ + σ⁻² Aᵀ A`
  (`main.tex` eq. `precision`), proved as the product identity
  `Prec · Σ_{g|x} = I`; and
* the **read-out identity** `B = σ⁻² Σ_{g|x} Aᵀ`
  (`main.tex` eq. `precision`).

Everything is a finite-dimensional matrix identity over `ℝ`, requiring only that
`Σ_g` and `Σ_x = A Σ_g Aᵀ + σ² I` be invertible and `σ² ≠ 0`.
-/

namespace RgitLean

open Matrix

variable {p d : ℕ}
variable {Sg : Matrix (Fin p) (Fin p) ℝ} {A : Matrix (Fin d) (Fin p) ℝ} {σ2 : ℝ}

/-- `Prec · Σ_g = I + σ⁻² Aᵀ A Σ_g`. -/
private lemma prec_mul_Sg (hSg : IsUnit Sg.det) :
    Prec Sg A σ2 * Sg = 1 + σ2⁻¹ • (Aᵀ * A * Sg) := by
  have hSgi : Sg⁻¹ * Sg = 1 := nonsing_inv_mul Sg hSg
  unfold Prec
  rw [add_mul, hSgi, smul_mul_assoc]

/-- **Proposition 2, precision form** (`main.tex` eq. `precision`):
the posterior precision `Σ_g⁻¹ + σ⁻² Aᵀ A` is a left inverse of the posterior
covariance `Σ_{g|x}`.  Combined with `nonsing_inv` lemmas this is exactly
`Σ_{g|x}⁻¹ = Σ_g⁻¹ + σ⁻² Aᵀ A`. -/
theorem prec_mul_postcov_eq_one
    (hSg : IsUnit Sg.det) (hSx : IsUnit (Sx Sg A σ2).det) (hσ : σ2 ≠ 0) :
    Prec Sg A σ2 * Sgcx Sg A σ2 = 1 := by
  have hSxi : Sx Sg A σ2 * (Sx Sg A σ2)⁻¹ = 1 := mul_nonsing_inv _ hSx
  have hσi : σ2⁻¹ * σ2 = 1 := inv_mul_cancel₀ hσ
  -- `(I + σ⁻² Aᵀ A Σ_g) Aᵀ = Aᵀ (σ⁻² Σ_x)`, the key push-through step.
  have hfac : (1 + σ2⁻¹ • (Aᵀ * A * Sg)) * Aᵀ = Aᵀ * (σ2⁻¹ • Sx Sg A σ2) := by
    rw [Sx_def]
    simp only [Matrix.add_mul, Matrix.mul_add, Matrix.mul_smul, smul_mul, Matrix.one_mul,
      Matrix.mul_one, smul_add, smul_smul, hσi, one_smul, Matrix.mul_assoc]
    abel
  -- `Aᵀ Σ_x Σ_x⁻¹ = Aᵀ`.
  have hcancel : Aᵀ * Sx Sg A σ2 * (Sx Sg A σ2)⁻¹ = Aᵀ := by
    rw [Matrix.mul_assoc Aᵀ (Sx Sg A σ2) (Sx Sg A σ2)⁻¹, hSxi, Matrix.mul_one]
  -- The "off-diagonal" term collapses to `σ⁻² Aᵀ A Σ_g`.
  have hPT : Prec Sg A σ2 * (Sg * Aᵀ * (Sx Sg A σ2)⁻¹ * (A * Sg))
      = σ2⁻¹ • (Aᵀ * A * Sg) := by
    calc Prec Sg A σ2 * (Sg * Aᵀ * (Sx Sg A σ2)⁻¹ * (A * Sg))
        = Prec Sg A σ2 * Sg * Aᵀ * (Sx Sg A σ2)⁻¹ * (A * Sg) := by
          simp only [Matrix.mul_assoc]
      _ = (1 + σ2⁻¹ • (Aᵀ * A * Sg)) * Aᵀ * (Sx Sg A σ2)⁻¹ * (A * Sg) := by
          rw [prec_mul_Sg hSg]
      _ = Aᵀ * (σ2⁻¹ • Sx Sg A σ2) * (Sx Sg A σ2)⁻¹ * (A * Sg) := by
          rw [hfac]
      _ = σ2⁻¹ • (Aᵀ * A * Sg) := by
          rw [Matrix.mul_smul, smul_mul, smul_mul]
          congr 1
          rw [hcancel, ← Matrix.mul_assoc]
  -- Assemble: `Prec · (Σ_g − off-diagonal) = (I + term) − term = I`.
  unfold Sgcx
  rw [mul_sub, prec_mul_Sg hSg, hPT]
  abel

/-- **Proposition 2, precision form**, in the manuscript's stated direction:
`Σ_{g|x}⁻¹ = Σ_g⁻¹ + σ⁻² Aᵀ A` (`main.tex` eq. `precision`). -/
theorem postcov_inv_eq_prec
    (hSg : IsUnit Sg.det) (hSx : IsUnit (Sx Sg A σ2).det) (hσ : σ2 ≠ 0) :
    (Sgcx Sg A σ2)⁻¹ = Prec Sg A σ2 :=
  inv_eq_left_inv (prec_mul_postcov_eq_one hSg hSx hσ)

/-- The posterior covariance is the inverse of the precision matrix:
`Σ_{g|x} = (Σ_g⁻¹ + σ⁻² Aᵀ A)⁻¹`. -/
theorem postcov_eq_inv_prec
    (hSg : IsUnit Sg.det) (hSx : IsUnit (Sx Sg A σ2).det) (hσ : σ2 ≠ 0) :
    (Prec Sg A σ2)⁻¹ = Sgcx Sg A σ2 :=
  inv_eq_right_inv (prec_mul_postcov_eq_one hSg hSx hσ)

/-- **Proposition 2, read-out identity** (`main.tex` eq. `precision`):
the Bayes-optimal map satisfies `B = σ⁻² Σ_{g|x} Aᵀ`. -/
theorem Bopt_eq_smul_postcov
    (hSx : IsUnit (Sx Sg A σ2).det) (hσ : σ2 ≠ 0) :
    Bopt Sg A σ2 = σ2⁻¹ • (Sgcx Sg A σ2 * Aᵀ) := by
  have hSxi : (Sx Sg A σ2)⁻¹ * Sx Sg A σ2 = 1 := nonsing_inv_mul _ hSx
  have hσi : σ2⁻¹ * σ2 = 1 := inv_mul_cancel₀ hσ
  -- `A Σ_g Aᵀ = Σ_x − σ² I`.
  have hAGA : A * Sg * Aᵀ = Sx Sg A σ2 - σ2 • (1 : Matrix (Fin d) (Fin d) ℝ) := by
    rw [Sx_def]; abel
  -- `Σ_g Aᵀ Σ_x⁻¹ (A Σ_g Aᵀ) = Σ_g Aᵀ − σ² (Σ_g Aᵀ Σ_x⁻¹)`.
  have hcancel2 : Sg * Aᵀ * (Sx Sg A σ2)⁻¹ * Sx Sg A σ2 = Sg * Aᵀ := by
    rw [Matrix.mul_assoc (Sg * Aᵀ) (Sx Sg A σ2)⁻¹ (Sx Sg A σ2), hSxi, Matrix.mul_one]
  have hmid : Sg * Aᵀ * (Sx Sg A σ2)⁻¹ * (A * Sg * Aᵀ)
      = Sg * Aᵀ - σ2 • (Sg * Aᵀ * (Sx Sg A σ2)⁻¹) := by
    rw [hAGA, Matrix.mul_sub, hcancel2, Matrix.mul_smul, Matrix.mul_one]
  -- `Σ_{g|x} Aᵀ = σ² (Σ_g Aᵀ Σ_x⁻¹)`.
  have key : Sgcx Sg A σ2 * Aᵀ = σ2 • (Sg * Aᵀ * (Sx Sg A σ2)⁻¹) := by
    unfold Sgcx
    rw [Matrix.sub_mul]
    calc Sg * Aᵀ - Sg * Aᵀ * (Sx Sg A σ2)⁻¹ * (A * Sg) * Aᵀ
        = Sg * Aᵀ - Sg * Aᵀ * (Sx Sg A σ2)⁻¹ * (A * Sg * Aᵀ) := by
          simp only [Matrix.mul_assoc]
      _ = Sg * Aᵀ - (Sg * Aᵀ - σ2 • (Sg * Aᵀ * (Sx Sg A σ2)⁻¹)) := by
          rw [hmid]
      _ = σ2 • (Sg * Aᵀ * (Sx Sg A σ2)⁻¹) := by abel
  unfold Bopt
  rw [Sgx_def, key, smul_smul, hσi, one_smul]

end RgitLean
