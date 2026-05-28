# rgit_lean — machine-checked proofs of the `main.tex` propositions

A [Lean 4](https://leanprover.github.io/) + [Mathlib](https://github.com/leanprover-community/mathlib4)
formalization of the algebraic propositions in the manuscript `main.tex`
(the linear–Gaussian radiogenomic model). Every theorem here is checked by the
Lean kernel: each of the headline results depends only on Lean's three standard
axioms (`propext`, `Classical.choice`, `Quot.sound`) and contains no `sorry`.

The model is

> `G ~ N(0, Σ_g)`, `X = A G + ε`, `ε ~ N(0, σ² I_d)`, `ε ⟂ G`,

with genomics `G ∈ ℝ^p`, imaging `X ∈ ℝ^d`, transfer map `A ∈ ℝ^{d×p}`.

## What is proved (and what is assumed)

The manuscript's propositions split into a *probabilistic layer* (jointly
Gaussian laws, Gaussian conditioning, the Gaussian MI identity, the SVD /
CCA spectral theorem) and an *algebraic layer* (the closed-form matrix and
scalar identities the manuscript derives from the probabilistic layer). The
probabilistic layer is exactly the set of facts the manuscript itself cites as
classical (Cover & Thomas; Hotelling; Bach & Jordan). **This development takes
those classical facts as the interface and machine-checks the entire algebraic
layer** — which is where the manuscript's derivations actually live.

| Manuscript | Lean file | Status |
|---|---|---|
| Prop. 1 — joint Gaussian law (`Σ_x`, `Σ_gx` blocks) | `Model.lean` | second moments **defined** as stated; jointly-Gaussian step is the cited classical affine-Gaussian fact |
| Prop. 2 — posterior precision `Σ_{g\|x}⁻¹ = Σ_g⁻¹ + σ⁻²AᵀA` and read-out `B = σ⁻²Σ_{g\|x}Aᵀ` | `Posterior.lean` | **proved** (`postcov_inv_eq_prec`, `postcov_eq_inv_prec`, `Bopt_eq_smul_postcov`) |
| Prop. 3 — `I(G;X) = ½log(det Σ_g/det Σ_{g\|x}) = ½log det(I+σ⁻²AΣ_gAᵀ)` | `MutualInformation.lean` | determinant bridge + Sylvester swap + half-log form **proved** (`det_Sg_eq`, `det_mi_sylvester`, `mi_logdet`); the `I = ½log` ratio bridge is the cited classical Gaussian-MI identity |
| Prop. 4 — canonical decomposition, boxed `R(aᵢ) = ρᵢ²` | `Recoverability.lean` | scalar chain `1 − 1/(1+dᵢ²) = dᵢ²/(1+dᵢ²) = ρᵢ²` **proved** (`recoverability_canonical`, `rho_sq`, `di_sq_eq_rho`); the SVD / per-direction `Var(Sᵢ\|X)=1/(1+dᵢ²)` is the cited classical SVD step |
| Def. — recoverability score `R(v)` | `Recoverability.lean` | `R(v)` **defined**; explained-`R²` form **proved** (`Rscore_eq`) |
| Prop. 5 — recoverability eigenproblem = CCA | `Recoverability.lean` | explained covariance = CCA form `Σ_gx Σ_x⁻¹ Σ_xg` **proved** (`explained_eq_cca`); the variational/eigen identification is the cited Hotelling CCA result |
| Prop. 6 — rank limit `rank(Σ_g − Σ_{g\|x}) ≤ min(d,p,rank A,rank Σ_g)` | `RankLimit.lean` | bound **proved** (`rank_limit` and the four component bounds) |

This complements `../scripts/verify_propositions.py`, which checks the *same*
identities numerically (random instances + Monte-Carlo) to machine precision;
the Lean development checks them symbolically, for all dimensions and all
admissible parameters.

## Layout

- `RgitLean/Model.lean` — model parameters and the closed-form second moments (`Sx`, `Sgx`, `Sgcx`, `Bopt`, `Prec`, `Explained`).
- `RgitLean/Posterior.lean` — Proposition 2.
- `RgitLean/MutualInformation.lean` — Proposition 3.
- `RgitLean/RankLimit.lean` — Proposition 6.
- `RgitLean/Recoverability.lean` — the recoverability score, Proposition 4 (scalar), Proposition 5 (CCA form).
- `RgitLean.lean` — imports the whole library.

## Building

```bash
cd rgit_lean
lake exe cache get      # download prebuilt Mathlib oleans (first time only)
lake build              # check every proof
```

To re-confirm that nothing is proved by `sorry`:

```bash
echo 'import RgitLean
open RgitLean
#print axioms postcov_inv_eq_prec' | lake env lean --stdin
# expect: depends on axioms: [propext, Classical.choice, Quot.sound]
```

### Note for this machine (RHEL 8 / glibc 2.28)

The bundled Lean toolchain's `clang` requires `GLIBC_2.29`, which is newer than
the system glibc, and the Mathlib cache tool ships a `curl` whose OpenSSL config
is incompatible with the system one. The `./check.sh` wrapper sets the few
environment variables that work around both (use the system C compiler `cc`, put
the toolchain's shared libraries on the loader path, and neutralize the system
`OPENSSL_CONF`) and then runs `lake build`. Use it instead of bare `lake build`:

```bash
./check.sh            # builds/checks the library
./check.sh cache      # one-time: fetch the Mathlib olean cache
```
