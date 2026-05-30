"""Fixed-working-dimension downsampling sweep on the synthetic instance.

Companion to the cohort downsample sweeps (ADNI/NSCLC) but on the synthetic
linear--Gaussian instance of `rgit.make_synthetic_radiogenomics`, where the
leading recoverability is known in closed form via `rgit.true_recoverability`.
We hold the working dimension FIXED (p*=d*=DOWN_DIM) and vary only n, so the
plot isolates "how much data is needed" from "bigger model", and shows the
honest cross-validated estimate converging to the planted truth as n grows.

Saves synthetic_downsample_recoverability.{pdf} and a stats block alongside the
other synthetic figures. Run from repo root: python scripts/synthetic_downsample_sweep.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))
from rgit import (
    make_synthetic_radiogenomics,
    true_recoverability,
    cross_validated_recoverability,
    permutation_test,
)

REPO = Path(__file__).parent.parent
OUT = REPO / "notebooks" / "figures" / "synthetic" / "gene_expression_synthetic" / "imaging_synthetic"
OUT.mkdir(parents=True, exist_ok=True)

# Match the notebook's synthetic instance so this aligns with the other
# synthetic figures (Sections on estimation regimes / honest recoverability).
N_FULL, P, D, K = 800, 80, 40, 6
SEED = 0
DOWN_DIM = 20            # fixed working dimension p*=d* across all subsample sizes
FRACS = [0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.6, 0.8, 1.0]
N_FOLDS = 5
N_PERM = 200


def fixed_cvr1(G, X, sel):
    """Leading CV recoverability and the permutation 95% null at fixed working dim."""
    n_sub = len(sel)
    Gs = PCA(min(DOWN_DIM, P, n_sub - 1), random_state=SEED).fit_transform(
        StandardScaler().fit_transform(G[sel]))
    Xs = PCA(min(DOWN_DIM, D, n_sub - 1), random_state=SEED).fit_transform(
        StandardScaler().fit_transform(X[sel]))
    cv = cross_validated_recoverability(
        Gs, Xs, n_components=3, n_folds=min(N_FOLDS, max(2, n_sub // 10)),
        random_state=SEED).mean(0)
    _, nulls, _ = permutation_test(Gs, Xs, n_components=1, n_perm=N_PERM, random_state=SEED)
    return float(cv[0]), float(np.quantile(nulls[:, 0], 0.95) ** 2)


def main():
    G, X, truth = make_synthetic_radiogenomics(
        n=N_FULL, p=P, d=D, k=K, random_state=SEED, genomics_type="gaussian")
    _, R_true = true_recoverability(truth)
    r1_true = float(R_true[0])
    print(f"planted leading recoverability rho_1^2 = {r1_true:.3f}  "
          f"(true spectrum: {np.round(R_true[:K], 3)})")

    rng = np.random.default_rng(SEED + 7)
    sizes = sorted({max(40, int(round(f * N_FULL))) for f in FRACS})
    sizes = [n for n in sizes if n <= N_FULL]

    rows = []
    for n_sub in sizes:
        sel = rng.choice(N_FULL, size=n_sub, replace=False)
        r1, thr = fixed_cvr1(G, X, sel)
        rows.append({"n": n_sub, "frac": round(n_sub / N_FULL, 3),
                     "R1_cv": r1, "null95": thr})
        print(f"  n={n_sub:4d}  CV R1={r1:.3f}  perm-null95={thr:.3f}")

    ns = np.array([r["n"] for r in rows])
    r1s = np.array([r["R1_cv"] for r in rows])
    thrs = np.array([r["null95"] for r in rows])

    # smallest n recovering 90% of the planted truth
    reach = ns[r1s >= 0.9 * r1_true]
    n_needed = int(reach[0]) if len(reach) else None
    # smallest n clearing the permutation null
    clears = ns[r1s > thrs]
    n_clear = int(clears[0]) if len(clears) else None

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.axhline(r1_true, ls="-", lw=1.5, color="#55A868",
               label=r"planted truth $\rho_1^2 = %.2f$" % r1_true)
    ax.plot(ns, r1s, "s-", color="#4C72B0", label=r"CV $\hat R_1$ (estimate)")
    ax.plot(ns, thrs, "^--", color="#C44E52", label=r"perm. null 95% $\rho_1^2$")
    if n_needed is not None:
        ax.axvline(n_needed, ls=":", color="gray")
        ax.annotate(f"90% of truth\nat n$\\approx${n_needed}", (n_needed, 0.02),
                    fontsize=8, color="gray", ha="left")
    ax.set_xlabel(r"subsample size $n$  (fixed working dim $p^\star=d^\star=%d$)" % DOWN_DIM)
    ax.set_ylabel(r"leading cross-validated $\hat R_1$")
    ax.set_title("Synthetic data: CV recoverability vs. subsample size, known truth")
    ax.set_ylim(-0.02, max(0.4, r1_true * 1.25, float(r1s.max()) * 1.15))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT / "synthetic_downsample_recoverability.pdf", bbox_inches="tight")
    print(f"saved figure: {(OUT / 'synthetic_downsample_recoverability.pdf').relative_to(REPO)}")

    # merge into the synthetic stats.json
    stats_path = OUT / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    stats["synthetic_downsample_sweep"] = {
        "working_dim": DOWN_DIM, "n_full": N_FULL, "p": P, "d": D, "k": K,
        "rho1_sq_true": r1_true,
        "rows": rows,
        "n_for_90pct_truth": n_needed,
        "n_clears_perm_null": n_clear,
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"n for 90% of truth: {n_needed};  n clearing perm null: {n_clear}")


if __name__ == "__main__":
    main()
