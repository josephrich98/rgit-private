"""Consolidated baseline-vs-refined recoverability comparison on KIRC + NSCLC.

Reports, per cohort:
  baseline  = z-score marginals, in-sample-rho permutation eff-rank (paper-style)
  refined   = Gaussian-copula marginals, honest CV-permutation eff-rank
so the effect of each refinement is explicit.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # prefer local rgit-private/rgit

import numpy as np
from rgit import (fit_recoverability, cross_validated_recoverability,
                  permutation_test, cv_permutation_test, gaussian_rank_transform,
                  imaging_variance_explained, mutual_information)
from scripts.recoverability_diagnostics import load_pair, prep_radiomics
from scipy import stats


def prep_expression(G, n_hvg=2000):
    Gp = np.asarray(G, dtype=np.float64)
    # NSCLC matrix carries NaNs and mixed scaling: drop non-finite genes first
    finite = np.isfinite(Gp).all(0)
    Gp = Gp[:, finite]
    nz = (Gp > 0).mean(0) > 0.1                 # expressed in >10% of patients
    Gp = Gp[:, nz]
    libs = Gp.sum(1, keepdims=True); libs[libs == 0] = 1.0
    Gp = np.log1p(Gp / libs * np.median(libs))
    keep = np.argsort(Gp.var(0))[::-1][:n_hvg]
    Gp = Gp[:, keep]
    return (Gp - Gp.mean(0)) / (Gp.std(0) + 1e-8)


def pca(M, k, seed=0):
    Mc = M - M.mean(0)
    U, s, Vt = np.linalg.svd(Mc, full_matrices=False)
    k = min(k, Vt.shape[0])
    return U[:, :k] * s[:k]


def kurt95(M):
    return np.percentile(np.abs(stats.kurtosis(M, axis=0)), 95)


def analyze(name, Gp, Xp, tau=5, K=3, n_perm=200, seed=0):
    n = Gp.shape[0]; ps = min(n // tau, Xp.shape[1], Gp.shape[1])
    print(f"\n{'='*66}\n{name}   n={n}  p*=d*={ps}")
    for label, Gm, Xm in [
        ("baseline z-score", Gp, Xp),
        ("refined  copula ", gaussian_rank_transform(Gp), gaussian_rank_transform(Xp)),
    ]:
        Gs, Xs = pca(Gm, ps, seed), pca(Xm, ps, seed)
        fit = fit_recoverability(Gs, Xs, n_components=ps)
        cv = cross_validated_recoverability(Gs, Xs, n_components=K, n_folds=5, random_state=seed).mean(0)
        # paper-style eff-rank: CV-R vs in-sample-rho null
        _, null_rho, _ = permutation_test(Gs, Xs, n_components=K, n_perm=n_perm, random_state=seed)
        paper_rank = int(np.sum(cv > np.quantile(null_rho**2, 0.95, axis=0)))
        # honest eff-rank: CV-R vs CV-R null
        obs_cv, null_cv, p_cv = cv_permutation_test(Gs, Xs, n_components=K, n_perm=n_perm, random_state=seed)
        honest_rank = int(np.sum((obs_cv > np.quantile(null_cv, 0.95, axis=0)) & (p_cv < 0.05)))
        frac, _ = imaging_variance_explained(fit)
        print(f"  [{label}] X-PC |kurt|p95={kurt95(Xs):4.1f}  in-MI={mutual_information(fit.rho)/np.log(2):4.1f}b"
              f"  CV-R top3={np.round(cv,3)}")
        print(f"       paper eff-rank(CVR>inρ²null95)={paper_rank}   "
              f"HONEST eff-rank(CVR>CVRnull95 & p<.05)={honest_rank}   CV-perm p={np.round(p_cv,3)}")


if __name__ == "__main__":
    # KIRC
    G, X, *_ = load_pair("data/tcga_kirc/genomics/gene_expression.h5ad",
                         "data/tcga_kirc/imaging/organ_radiomics.h5ad", gen_layer="tpm_unstranded")
    analyze("TCGA-KIRC  expr x organ-radiomics", prep_expression(G), prep_radiomics(X, signed_log=False)[0])
    # NSCLC
    G2, X2, *_ = load_pair("data/nsclc/genomics/gene_expression.h5ad",
                           "data/nsclc/imaging/organ_radiomics_subset.h5ad")
    analyze("NSCLC      expr x organ-radiomics", prep_expression(G2), prep_radiomics(X2, signed_log=False)[0])
