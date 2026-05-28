/-
Copyright (c) 2026 Joseph Rich. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Joseph Rich
-/
import RgitLean.Posterior

/-!
# Proposition 3 — Mutual information and the determinant identity

`main.tex` **Proposition 3 (Mutual information; classical Gaussian MI identity)**
states
`I(G;X) = ½ log(det Σ_g / det Σ_{g|x}) = ½ log det(I_d + σ⁻² A Σ_g Aᵀ)`.

The identification `I(G;X) = ½ log(det Σ_g / det Σ_{g|x})` for a jointly Gaussian
pair is the classical Gaussian mutual-information identity cited in the
manuscript.  The *algebraic* content the manuscript derives from it — and what we
machine-check here — is:

* `det Σ_g = det(I + σ⁻² A Σ_g Aᵀ) · det Σ_{g|x}` (`det_Sg_eq`), equivalently
  `det Σ_g / det Σ_{g|x} = det(I_d + σ⁻² A Σ_g Aᵀ)`, which is the bridge between
  the two displayed forms of `I(G;X)`; and
* the **Sylvester / Weinstein–Aronszajn determinant identity**
  `det(I_d + σ⁻² A Σ_g Aᵀ) = det(I_p + σ⁻² Aᵀ A Σ_g)` (`det_mi_sylvester`), the
  `p ↔ d` swap used in the manuscript's proof.

Combining them yields the half-log identity `mi_logdet`, which is exactly the
manuscript's two displayed forms of `I(G;X)` once the classical
`I = ½ log(det Σ_g / det Σ_{g|x})` bridge is granted.
-/

namespace RgitLean

open Matrix

variable {p d : ℕ}
variable {Sg : Matrix (Fin p) (Fin p) ℝ} {A : Matrix (Fin d) (Fin p) ℝ} {σ2 : ℝ}

/-- `Σ_g · Prec = I + σ⁻² Σ_g Aᵀ A`. -/
private lemma Sg_mul_prec (hSg : IsUnit Sg.det) :
    Sg * Prec Sg A σ2 = 1 + σ2⁻¹ • (Sg * Aᵀ * A) := by
  unfold Prec
  rw [Matrix.mul_add, mul_nonsing_inv Sg hSg, Matrix.mul_smul, ← Matrix.mul_assoc]

/-- **Proposition 3, determinant bridge** (`main.tex` eq. `mi`):
`det Σ_g = det(I_d + σ⁻² A Σ_g Aᵀ) · det Σ_{g|x}`, i.e.
`det Σ_g / det Σ_{g|x} = det(I_d + σ⁻² A Σ_g Aᵀ)`. -/
theorem det_Sg_eq
    (hSg : IsUnit Sg.det) (hSx : IsUnit (Sx Sg A σ2).det) (hσ : σ2 ≠ 0) :
    Sg.det = (1 + σ2⁻¹ • (A * Sg * Aᵀ)).det * (Sgcx Sg A σ2).det := by
  -- `(I + σ⁻² Σ_g Aᵀ A) · Σ_{g|x} = Σ_g`.
  have key : (1 + σ2⁻¹ • (Sg * Aᵀ * A)) * Sgcx Sg A σ2 = Sg := by
    rw [← Sg_mul_prec hSg, Matrix.mul_assoc, prec_mul_postcov_eq_one hSg hSx hσ, Matrix.mul_one]
  -- Sylvester swap `det(I_p + σ⁻² Σ_g Aᵀ A) = det(I_d + σ⁻² A Σ_g Aᵀ)`.
  have hsyl : (1 + σ2⁻¹ • (Sg * Aᵀ * A)).det = (1 + σ2⁻¹ • (A * Sg * Aᵀ)).det := by
    have h := det_one_add_mul_comm (σ2⁻¹ • (Sg * Aᵀ)) A
    rw [smul_mul, Matrix.mul_smul, ← Matrix.mul_assoc] at h
    exact h
  have hprod : Sg.det = (1 + σ2⁻¹ • (Sg * Aᵀ * A)).det * (Sgcx Sg A σ2).det := by
    rw [← Matrix.det_mul, key]
  rw [hprod, hsyl]

/-- **Proposition 3, Sylvester identity** (`main.tex` proof of eq. `mi`):
`det(I_d + σ⁻² A Σ_g Aᵀ) = det(I_p + σ⁻² Aᵀ A Σ_g)`. -/
theorem det_mi_sylvester (Sg : Matrix (Fin p) (Fin p) ℝ) (A : Matrix (Fin d) (Fin p) ℝ)
    (σ2 : ℝ) :
    (1 + σ2⁻¹ • (A * Sg * Aᵀ)).det = (1 + σ2⁻¹ • (Aᵀ * A * Sg)).det := by
  have h := det_one_add_mul_comm (σ2⁻¹ • (A * Sg)) Aᵀ
  rw [smul_mul, Matrix.mul_smul, ← Matrix.mul_assoc] at h
  exact h

/-- **Proposition 3** in the manuscript's displayed half-log form: granting the
classical `I(G;X) = ½ log(det Σ_g / det Σ_{g|x})`, the mutual information equals
`½ log det(I_d + σ⁻² A Σ_g Aᵀ)`. -/
theorem mi_logdet
    (hSg : IsUnit Sg.det) (hSx : IsUnit (Sx Sg A σ2).det) (hσ : σ2 ≠ 0)
    (hne : (Sgcx Sg A σ2).det ≠ 0) :
    (1 / 2) * Real.log (Sg.det / (Sgcx Sg A σ2).det)
      = (1 / 2) * Real.log ((1 + σ2⁻¹ • (A * Sg * Aᵀ)).det) := by
  rw [det_Sg_eq hSg hSx hσ, mul_div_assoc, div_self hne, mul_one]

end RgitLean
