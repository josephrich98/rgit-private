/-
Copyright (c) 2026 Joseph Rich. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Joseph Rich
-/
import Mathlib

/-!
# The linear–Gaussian radiogenomic model

This file fixes the notation and the closed-form second moments of the
linear–Gaussian generative model of `main.tex` (Section "The linear–Gaussian
generative model"):

  `G ~ N(0, Σ_g)`,  `X = A G + ε`,  `ε ~ N(0, σ² I_d)`,  `ε ⟂ G`.

Genomics lives in `ℝ^p` (`Fin p`) and imaging in `ℝ^d` (`Fin d`).  The model is
parametrised by the transfer map `A : ℝ^{d×p}`, the genomic prior covariance
`Σ_g : ℝ^{p×p}` and the imaging-noise variance `σ² > 0`.

The probabilistic content of **Proposition 1 (Joint Gaussian law)** — that an
affine image of independent Gaussians is Gaussian with these moments — is the
classical affine-Gaussian fact cited in the manuscript.  Here we *define* the
induced second moments (`Sx`, `Sgx`, `Sgcx`) exactly as the manuscript states
them, and the later files verify every algebraic identity the manuscript derives
from them.

All matrices are over `ℝ`; multiplication `*` is rectangular matrix
multiplication and `ᵀ` is `Matrix.transpose`.
-/

namespace RgitLean

open Matrix

variable {p d : ℕ}

/-- Imaging marginal covariance `Σ_x = A Σ_g Aᵀ + σ² I_d`  (`main.tex` eq. for `Σ_x`). -/
def Sx (Sg : Matrix (Fin p) (Fin p) ℝ) (A : Matrix (Fin d) (Fin p) ℝ) (σ2 : ℝ) :
    Matrix (Fin d) (Fin d) ℝ :=
  A * Sg * Aᵀ + σ2 • (1 : Matrix (Fin d) (Fin d) ℝ)

/-- Cross-covariance `Σ_gx = Cov(G, X) = Σ_g Aᵀ`. -/
def Sgx (Sg : Matrix (Fin p) (Fin p) ℝ) (A : Matrix (Fin d) (Fin p) ℝ) :
    Matrix (Fin p) (Fin d) ℝ :=
  Sg * Aᵀ

/-- Posterior (conditional) covariance of genomics given imaging,
`Σ_{g|x} = Σ_g − Σ_g Aᵀ (A Σ_g Aᵀ + σ² I_d)⁻¹ A Σ_g`
(`main.tex` eq. `postcov`). -/
noncomputable def Sgcx (Sg : Matrix (Fin p) (Fin p) ℝ) (A : Matrix (Fin d) (Fin p) ℝ) (σ2 : ℝ) :
    Matrix (Fin p) (Fin p) ℝ :=
  Sg - Sg * Aᵀ * (Sx Sg A σ2)⁻¹ * (A * Sg)

/-- Bayes-optimal (posterior-mean) read-out matrix `B = Σ_gx Σ_x⁻¹`
(`main.tex` eq. `postmean`); the optimal estimator is `ĝ(X) = B X`. -/
noncomputable def Bopt (Sg : Matrix (Fin p) (Fin p) ℝ) (A : Matrix (Fin d) (Fin p) ℝ) (σ2 : ℝ) :
    Matrix (Fin p) (Fin d) ℝ :=
  Sgx Sg A * (Sx Sg A σ2)⁻¹

/-- Posterior precision `Σ_{g|x}⁻¹ = Σ_g⁻¹ + σ⁻² Aᵀ A`  (`main.tex` eq. `precision`). -/
noncomputable def Prec (Sg : Matrix (Fin p) (Fin p) ℝ) (A : Matrix (Fin d) (Fin p) ℝ) (σ2 : ℝ) :
    Matrix (Fin p) (Fin p) ℝ :=
  Sg⁻¹ + σ2⁻¹ • (Aᵀ * A)

/-- The "explained" genomic covariance `Σ_g − Σ_{g|x} = Σ_gx Σ_x⁻¹ Σ_xg`
(`main.tex` eq. `explained`); its quadratic forms drive the recoverability score. -/
noncomputable def Explained (Sg : Matrix (Fin p) (Fin p) ℝ)
    (A : Matrix (Fin d) (Fin p) ℝ) (σ2 : ℝ) :
    Matrix (Fin p) (Fin p) ℝ :=
  Sg - Sgcx Sg A σ2

@[simp] lemma Sx_def (Sg : Matrix (Fin p) (Fin p) ℝ) (A : Matrix (Fin d) (Fin p) ℝ) (σ2 : ℝ) :
    Sx Sg A σ2 = A * Sg * Aᵀ + σ2 • (1 : Matrix (Fin d) (Fin d) ℝ) := rfl

@[simp] lemma Sgx_def (Sg : Matrix (Fin p) (Fin p) ℝ) (A : Matrix (Fin d) (Fin p) ℝ) :
    Sgx Sg A = Sg * Aᵀ := rfl

end RgitLean
