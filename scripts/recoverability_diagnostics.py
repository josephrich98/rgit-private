"""Diagnostic harness for the linear-Gaussian recoverability model.

Loads a (genomics, imaging) AnnData pair, applies configurable preprocessing,
and reports how well the data meet the model's assumptions plus honest
(CV / permutation) recoverability. Intended to be imported from the notebook
or run standalone for quick iteration.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # prefer local rgit-private/rgit

import numpy as np
import scipy.sparse as sp
import anndata as ad
from scipy import stats

from rgit import (
    fit_recoverability,
    cross_validated_recoverability,
    permutation_test,
    imaging_variance_explained,
    mutual_information,
)


def _dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def load_pair(gen_path, img_path, gen_layer=None):
    g = ad.read_h5ad(gen_path)
    x = ad.read_h5ad(img_path)
    common = [p for p in g.obs_names if p in set(x.obs_names)]
    g = g[common].copy()
    x = x[common].copy()
    G = _dense(g.layers[gen_layer]) if gen_layer else _dense(g.X)
    X = _dense(x.X)
    return G, X, np.array(common), np.array(g.var_names), np.array(x.var_names)


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------
def prep_expression(G, n_hvg=2000, log=True):
    """CPM-ish log1p + HVG (by variance) + z-score."""
    Gp = G.astype(np.float64)
    if log:
        # library-size normalize to median depth then log1p
        libs = Gp.sum(1, keepdims=True)
        libs[libs == 0] = 1.0
        Gp = np.log1p(Gp / libs * np.median(libs))
    var = Gp.var(0)
    keep = np.argsort(var)[::-1][:n_hvg]
    Gp = Gp[:, keep]
    Gp = (Gp - Gp.mean(0)) / (Gp.std(0) + 1e-8)
    return Gp, keep


def prep_radiomics(X, signed_log=True):
    """Signed-log1p to tame heavy tails, then z-score; drop constant cols."""
    Xp = X.astype(np.float64)
    keep = Xp.std(0) > 0
    Xp = Xp[:, keep]
    if signed_log:
        Xp = np.sign(Xp) * np.log1p(np.abs(Xp))
    Xp = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-8)
    return Xp, keep


def gaussian_rank(M):
    """Column-wise rank -> inverse-normal (Gaussian copula marginal transform).

    Maps every feature to an exact standard-normal marginal, so the only thing
    the linear-Gaussian fit can use is the *dependence* structure -- the right
    move when the model's Gaussian-marginal assumption is the misfit.
    """
    M = np.asarray(M, dtype=np.float64)
    n = M.shape[0]
    R = np.argsort(np.argsort(M, axis=0), axis=0) + 1.0  # ranks 1..n
    return stats.norm.ppf(R / (n + 1.0))


def pca_reduce(M, k, seed=0):
    Mc = M - M.mean(0)
    U, s, Vt = np.linalg.svd(Mc, full_matrices=False)
    k = min(k, Vt.shape[0])
    return U[:, :k] * s[:k], Vt[:k]  # scores, components


# ---------------------------------------------------------------------------
# assumption diagnostics
# ---------------------------------------------------------------------------
def gaussianity(M, name):
    sk = stats.skew(M, axis=0)
    ku = stats.kurtosis(M, axis=0)  # excess
    print(f"  [{name}] skew  median={np.median(np.abs(sk)):.2f} "
          f"p95={np.percentile(np.abs(sk),95):.2f}")
    print(f"  [{name}] |kurt| median={np.median(np.abs(ku)):.2f} "
          f"p95={np.percentile(np.abs(ku),95):.2f}")


def run(name, G, X, p_star, d_star, n_perm=200, n_folds=5, seed=0):
    print(f"\n{'='*64}\n{name}: n={G.shape[0]}  p*={p_star} d*={d_star}")
    Gs, _ = pca_reduce(G, p_star, seed)
    Xs, _ = pca_reduce(X, d_star, seed)
    gaussianity(Gs, "G PCs")
    gaussianity(Xs, "X PCs")

    nc = min(p_star, d_star)
    fit = fit_recoverability(Gs, Xs, n_components=nc)
    cv = cross_validated_recoverability(Gs, Xs, n_components=min(nc, 10),
                                        n_folds=n_folds, random_state=seed)
    cv_mean = cv.mean(0)
    obs, null, pvals = permutation_test(Gs, Xs, n_components=3, n_perm=n_perm,
                                        random_state=seed)
    frac_in, _ = imaging_variance_explained(fit)

    print(f"  in-sample R1={fit.recoverability[0]:.3f}  rho1={fit.rho[0]:.3f}  "
          f"MI(in)={mutual_information(fit.rho)/np.log(2):.1f} bits over {nc} comps")
    print(f"  CV R: top5={np.round(cv_mean[:5],3)}")
    print(f"  perm p (top3)={np.round(pvals,3)}  null rho1 mean={null[:,0].mean():.3f} "
          f"q95={np.percentile(null[:,0],95):.3f}")
    print(f"  imaging var frac by genomics (in-sample)={frac_in:.3f}")
    # honest effective rank: permutation-significant AND positive out-of-sample R
    keff = int(np.sum((pvals < 0.05) & (cv_mean[:3] > 0)))
    print(f"  effective image-identifiable rank (perm p<.05 & CV R>0, top3) = {keff}")
    return dict(fit=fit, cv=cv_mean, pvals=pvals, null=null, frac_in=frac_in, keff=keff)


if __name__ == "__main__":
    G, X, pats, gv, xv = load_pair(
        "data/tcga_kirc/genomics/gene_expression.h5ad",
        "data/tcga_kirc/imaging/organ_radiomics.h5ad",
        gen_layer="tpm_unstranded",
    )
    print("KIRC raw:", G.shape, X.shape, "patients", len(pats))
    n = G.shape[0]
    print("\n##### z-score marginals #####")
    Gp, _ = prep_expression(G, n_hvg=2000)
    Xp, _ = prep_radiomics(X)
    for tau in (5, 8):
        ps = n // tau
        run(f"KIRC zscore tau={tau}", Gp, Xp, ps, min(ps, Xp.shape[1]))

    print("\n##### Gaussian-rank marginals (copula) #####")
    Gp2, keep = prep_expression(G, n_hvg=2000)
    Gp2 = gaussian_rank(Gp2)
    Xp2, _ = prep_radiomics(X, signed_log=False)
    Xp2 = gaussian_rank(Xp2)
    for tau in (5, 8):
        ps = n // tau
        run(f"KIRC grank tau={tau}", Gp2, Xp2, ps, min(ps, Xp2.shape[1]))
