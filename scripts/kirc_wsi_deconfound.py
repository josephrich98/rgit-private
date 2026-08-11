"""Demographic / tissue-source-site deconfounding for the TCGA-KIRC WSI signal.

The histopathology-vs-expression channel (scripts/kirc_wsi_recoverability.py) is
much stronger than the CT channel (R1_cv ~ 0.46 vs ~ 0.11), so it is exactly the
regime in which a demographic or batch confound could inflate the effect. We
apply the same skepticism used for CT (scripts/kirc_deconfound.py): residualize
*both* modalities (post-copula, post-PCA working scores) on a confounder design
and refit the honest CV / CV-permutation recoverability.

Confounders:
  age, sex, ethnicity, race            -- GDC clinical_tcga.tsv
  tissue source site (TSS, barcode)    -- the WSI analogue of CT scanner manufacturer
                                          (per-institution H&E staining / batch)

Run from the repo root:  python scripts/kirc_wsi_deconfound.py
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from rgit import (cross_validated_recoverability, cv_permutation_test,
                  gaussian_rank_transform, mutual_information, fit_recoverability)
from scripts.recoverability_diagnostics import prep_radiomics, _dense
from scripts.final_recoverability_compare import prep_expression
from scripts.kirc_wsi_recoverability import load_gen, align
import anndata as ad

REPO = Path(__file__).parent.parent
FIGDIR = REPO / "notebooks/figures/tcga_kirc/gene_expression/wsi"
TAU, K, SEED, N_PERM = 5, 3, 0, 1000


def confounders(case_ids):
    """[1, z(age), one-hot(sex, ethnicity, race, tissue-source-site)]."""
    cols = ["cases.submitter_id", "demographic.age_at_index", "demographic.gender",
            "demographic.ethnicity", "demographic.race"]
    cl = (pd.read_csv(REPO / "data/tcga_kirc/clinical_tcga.tsv", sep="\t",
                      usecols=cols, low_memory=False)
          .replace("'--", np.nan).drop_duplicates("cases.submitter_id")
          .set_index("cases.submitter_id").reindex(case_ids))
    age = pd.to_numeric(cl["demographic.age_at_index"], errors="coerce")
    age = age.fillna(age.mean())
    # tissue source site = 2nd barcode field (TCGA-<TSS>-<participant>)
    tss = [c.split("-")[1] for c in case_ids]
    cat = pd.DataFrame({
        "sex": cl["demographic.gender"].fillna("unknown").values,
        "eth": cl["demographic.ethnicity"].fillna("not reported").values,
        "race": cl["demographic.race"].fillna("not reported").values,
        "tss": tss,
    }, index=case_ids)
    dummies = pd.get_dummies(cat, drop_first=True).values.astype(float)
    age_z = StandardScaler().fit_transform(age.values.reshape(-1, 1))
    D = np.column_stack([np.ones(len(case_ids)), age_z, dummies])
    return D, ["age", "sex", "ethnicity", "race", "tissue_source_site"]


def work(Gp, Xp, ps, seed=SEED):
    Gs = PCA(ps, random_state=seed).fit_transform(
        StandardScaler().fit_transform(gaussian_rank_transform(Gp)))
    Xs = PCA(ps, random_state=seed).fit_transform(
        StandardScaler().fit_transform(gaussian_rank_transform(Xp)))
    return Gs, Xs


def report(tag, Gs, Xs):
    cv = cross_validated_recoverability(Gs, Xs, n_components=K, n_folds=5,
                                        random_state=SEED).mean(0)
    obs, null, p = cv_permutation_test(Gs, Xs, n_components=K, n_perm=N_PERM,
                                       random_state=SEED)
    rank = int(np.sum((obs > np.quantile(null, 0.95, axis=0)) & (p < 0.05)))
    mi = mutual_information(fit_recoverability(Gs, Xs, n_components=K).rho[:K]) / np.log(2)
    print(f"  [{tag}] CV-R top{K}={np.round(cv, 3)}  CV-perm p={np.round(p, 3)}  "
          f"eff-rank={rank}  MI(top{K})={mi:.2f}b")
    return [float(x) for x in cv], [float(x) for x in p], rank, float(mi)


def main():
    gen_ad, Graw = load_gen()
    wsi_t = ad.read_h5ad(REPO / "data/tcga_kirc/wsi/wsi_tumor.h5ad")
    G, X, cases = align(gen_ad, Graw, wsi_t)
    Gp, Xp = prep_expression(G), prep_radiomics(X, signed_log=False)[0]
    n = len(cases); ps = min(n // TAU, Gp.shape[1], Xp.shape[1])
    print(f"TCGA-KIRC WSI  n={n}  p*=d*={ps}  perms={N_PERM}")

    Gs, Xs = work(Gp, Xp, ps)
    D, names = confounders(list(cases))
    print(f"confounders {names}: design has {D.shape[1]-1} columns (incl. one-hot), "
          f"full coverage on all {n} cases")
    resid = lambda M: M - D @ np.linalg.lstsq(D, M, rcond=None)[0]
    Gr, Xr = resid(Gs), resid(Xs)

    cv_raw, p_raw, rank_raw, mi_raw = report("raw        ", Gs, Xs)
    cv_dec, p_dec, rank_dec, mi_dec = report("deconfound ", Gr, Xr)
    print(f"\nleading CV-R retained: {cv_dec[0]/cv_raw[0]*100:.1f}%  "
          f"({cv_raw[0]:.3f} -> {cv_dec[0]:.3f})")

    out = {"n": n, "ps": ps, "confounders": names,
           "raw": {"cv": cv_raw, "p": p_raw, "eff_rank": rank_raw, "mi_bits": mi_raw},
           "deconfounded": {"cv": cv_dec, "p": p_dec, "eff_rank": rank_dec, "mi_bits": mi_dec}}
    json.dump(out, open(FIGDIR / "deconfound_stats.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    xj = np.arange(K); w = 0.38
    ax.bar(xj - w / 2, cv_raw, w, label="raw", color="#55A868")
    ax.bar(xj + w / 2, cv_dec, w, label="deconfounded", color="#DD8452")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(xj); ax.set_xticklabels([f"dir {i+1}" for i in range(K)])
    ax.set_ylabel(r"honest $\hat R_i^{\mathrm{cv}}$")
    ax.set_title("TCGA-KIRC WSI: age/sex/ethnicity/race/source-site deconfounding")
    ax.legend(frameon=False)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "deconfound_comparison.pdf", bbox_inches="tight")
    print(f"saved {(FIGDIR / 'deconfound_comparison.pdf').relative_to(REPO)}")


if __name__ == "__main__":
    main()
