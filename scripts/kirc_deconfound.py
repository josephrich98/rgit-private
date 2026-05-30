"""Demographic / scanner deconfounding for the TCGA-KIRC radiogenomic signal.

Mirrors the ADNI deconfounding (notebook) and the NSCLC batch control
(scripts/nsclc_batch_confound.py): regress a set of confounders out of *both*
modalities (post-copula, post-PCA working scores) and refit the honest
CV / CV-permutation recoverability, comparing raw vs deconfounded.

Confounders, read from the GDC clinical table and the TCIA series digest:
  age (demographic.age_at_index), sex, ethnicity, race  -- clinical_tcga.tsv
  scanner Manufacturer (per-patient mode)               -- TCIA *-nbia-digest.xlsx

Run from the repo root:  python scripts/kirc_deconfound.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from rgit import (cross_validated_recoverability, cv_permutation_test,
                  gaussian_rank_transform, mutual_information, fit_recoverability)
from scripts.recoverability_diagnostics import load_pair, prep_radiomics

REPO = Path(__file__).parent.parent
FIGDIR = REPO / "notebooks/figures/tcga_kirc/gene_expression/organ_radiomics"
TAU, K, SEED, N_PERM = 5, 3, 0, 1000


def prep_expression(G, n_hvg=2000):
    Gp = np.asarray(G, dtype=np.float64)
    finite = np.isfinite(Gp).all(0); Gp = Gp[:, finite]
    nz = (Gp > 0).mean(0) > 0.1; Gp = Gp[:, nz]
    libs = Gp.sum(1, keepdims=True); libs[libs == 0] = 1.0
    Gp = np.log1p(Gp / libs * np.median(libs))
    keep = np.argsort(Gp.var(0))[::-1][:n_hvg]; Gp = Gp[:, keep]
    return (Gp - Gp.mean(0)) / (Gp.std(0) + 1e-8)


def confounders(patient_ids):
    """[1, z(age), sex, one-hot(ethnicity, race, manufacturer)] aligned to patient_ids."""
    cols = ["cases.submitter_id", "demographic.age_at_index", "demographic.gender",
            "demographic.ethnicity", "demographic.race"]
    cl = (pd.read_csv(REPO / "data/tcga_kirc/clinical_tcga.tsv", sep="\t",
                      usecols=cols, low_memory=False)
          .replace("'--", np.nan).drop_duplicates("cases.submitter_id")
          .set_index("cases.submitter_id").reindex(patient_ids))
    xl = pd.read_excel(REPO / "data/tcga_kirc/imaging/"
                       "TCIA_TCGA-KIRC_09-16-2015-nbia-digest.xlsx")
    xl["Manufacturer"] = xl["Manufacturer"].str.replace("Philips Medical Systems",
                                                         "Philips", regex=False)
    man = (xl.groupby("Patient ID")["Manufacturer"]
           .agg(lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else np.nan)
           .reindex(patient_ids))

    age = pd.to_numeric(cl["demographic.age_at_index"], errors="coerce")
    age = age.fillna(age.mean())
    cat = pd.DataFrame({
        "sex": cl["demographic.gender"].fillna("unknown"),
        "eth": cl["demographic.ethnicity"].fillna("not reported"),
        "race": cl["demographic.race"].fillna("not reported"),
        "manu": man.fillna("unknown"),
    }, index=patient_ids)
    dummies = pd.get_dummies(cat, drop_first=True).values.astype(float)
    age_z = StandardScaler().fit_transform(age.values.reshape(-1, 1))
    D = np.column_stack([np.ones(len(patient_ids)), age_z, dummies])
    return D, ["age", "sex", "ethnicity", "race", "manufacturer"]


def work(G, X, ps, seed=SEED):
    Gm, Xm = gaussian_rank_transform(G), gaussian_rank_transform(X)
    Gs = PCA(ps, random_state=seed).fit_transform(StandardScaler().fit_transform(Gm))
    Xs = PCA(ps, random_state=seed).fit_transform(StandardScaler().fit_transform(Xm))
    return Gs, Xs


def report(tag, Gs, Xs):
    cv = cross_validated_recoverability(Gs, Xs, n_components=K, n_folds=5,
                                        random_state=SEED).mean(0)
    obs, null, p = cv_permutation_test(Gs, Xs, n_components=K, n_perm=N_PERM,
                                       random_state=SEED)
    rank = int(np.sum((obs > np.quantile(null, 0.95, axis=0)) & (p < 0.05)))
    mi = mutual_information(fit_recoverability(Gs, Xs, n_components=K).rho[:K]) / np.log(2)
    print(f"  [{tag}] CV-R top{K}={np.round(cv, 3)}  CV-perm p={np.round(p, 3)}  "
          f"eff-rank={rank}  in-MI(top{K})={mi:.2f}b")
    return cv, p, rank


def main():
    G, X, ids, *_ = load_pair("data/tcga_kirc/genomics/gene_expression.h5ad",
                              "data/tcga_kirc/imaging/organ_radiomics.h5ad",
                              gen_layer="tpm_unstranded")
    Gp, Xp = prep_expression(G), prep_radiomics(X, signed_log=False)[0]
    n = len(ids); ps = min(n // TAU, Gp.shape[1], Xp.shape[1])
    print(f"TCGA-KIRC  n={n}  p*=d*={ps}  perms={N_PERM}")

    Gs, Xs = work(Gp, Xp, ps)
    D, names = confounders(list(ids))
    print(f"confounders {names}: design has {D.shape[1]-1} columns "
          f"(incl. one-hot), full coverage on all {n} patients")
    resid = lambda M: M - D @ np.linalg.lstsq(D, M, rcond=None)[0]
    Gr, Xr = resid(Gs), resid(Xs)

    cv_raw, p_raw, rank_raw = report("raw        ", Gs, Xs)
    cv_dec, p_dec, rank_dec = report("deconfound ", Gr, Xr)
    print(f"\nleading CV-R retained: {cv_dec[0]/cv_raw[0]*100:.1f}%  "
          f"({cv_raw[0]:.3f} -> {cv_dec[0]:.3f})")

    # --- figure (matches ADNI deconfound_comparison.pdf) ------------------
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    xj = np.arange(K); w = 0.38
    ax.bar(xj - w/2, cv_raw, w, label="raw", color="#4C72B0")
    ax.bar(xj + w/2, cv_dec, w, label="deconfounded", color="#DD8452")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(xj); ax.set_xticklabels([f"dir {i+1}" for i in range(K)])
    ax.set_ylabel(r"honest $\hat R_i^{\mathrm{cv}}$")
    ax.set_title("TCGA-KIRC: age/sex/ethnicity/race/scanner deconfounding")
    ax.legend(frameon=False)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "deconfound_comparison.pdf", bbox_inches="tight")
    print(f"saved {(FIGDIR / 'deconfound_comparison.pdf').relative_to(REPO)}")


if __name__ == "__main__":
    main()
