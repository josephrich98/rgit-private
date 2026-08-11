"""Is genomics more reflected in WSI histopathology than in CT radiology?

Digital-pathology arm of the TCGA-KIRC recoverability analysis. We swap the CT
``organ_radiomics`` imaging modality for the handcrafted whole-slide-image (WSI)
feature bank built by ``build_kirc_wsi_features.py`` and re-run the *same* honest
recoverability pipeline used for radiology (Gaussian-copula marginals, symmetric
PCA to a working dimension p*=d*=n/tau, 5-fold cross-validated recoverability,
and a cross-validated permutation null). This makes WSI vs CT a like-for-like
comparison against the identical genomic target (TCGA-KIRC HVG expression).

Two questions:
  (1) head-to-head -- WSI vs CT radiomics, same expression target, same n-budget;
  (2) tumour vs adjacent-normal -- is a patient's tumour-derived expression more
      reflected in their tumour slides than their normal slides (size-matched)?

Run from the repo root:  python scripts/kirc_wsi_recoverability.py
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rgit import (cross_validated_recoverability, cv_permutation_test,
                  gaussian_rank_transform, mutual_information, fit_recoverability)
from scripts.recoverability_diagnostics import load_pair, prep_radiomics, _dense
from scripts.final_recoverability_compare import prep_expression, pca

REPO = Path(__file__).parent.parent
FIGDIR = REPO / "notebooks/figures/tcga_kirc/gene_expression/wsi"
GEN = "data/tcga_kirc/genomics/gene_expression.h5ad"
TAU, K, SEED, N_PERM = 5, 3, 0, 1000


# ---------------------------------------------------------------------------
# Honest recoverability on a (genomics, imaging) working space
# ---------------------------------------------------------------------------
def recover(Gp, Xp, *, tau=TAU, K=K, n_perm=N_PERM, seed=SEED):
    """Copula -> symmetric PCA(p*=d*=n/tau) -> honest CV + CV-permutation."""
    n = Gp.shape[0]
    ps = min(n // tau, Xp.shape[1], Gp.shape[1])
    Gs = pca(gaussian_rank_transform(Gp), ps, seed)
    Xs = pca(gaussian_rank_transform(Xp), ps, seed)
    cv = cross_validated_recoverability(Gs, Xs, n_components=K, n_folds=5,
                                        random_state=seed).mean(0)
    mi = mutual_information(fit_recoverability(Gs, Xs, n_components=K).rho[:K]) / np.log(2)
    if n_perm:
        obs, null, p = cv_permutation_test(Gs, Xs, n_components=K, n_perm=n_perm,
                                           random_state=seed)
        rank = int(np.sum((obs > np.quantile(null, 0.95, axis=0)) & (p < 0.05)))
        p = [float(x) for x in p]
    else:                                  # subsampling pass: skip the perm null
        rank, p = -1, None
    return dict(n=int(n), ps=int(ps), cv=[float(x) for x in cv],
                cv1=float(cv[0]), p=p, eff_rank=rank, mi_bits=float(mi))


def load_gen():
    g = ad.read_h5ad(GEN)
    return g, _dense(g.layers["tpm_unstranded"])


def align(gen_ad, Graw, img_ad):
    """Intersect a genomics AnnData (+raw matrix) with a case-indexed imaging
    AnnData; return (G_raw, X_raw, cases) in a shared case order."""
    common = [c for c in gen_ad.obs_names if c in set(img_ad.obs_names)]
    gi = gen_ad.obs_names.get_indexer(common)
    G = Graw[gi]
    X = _dense(img_ad[common].X)
    return G, X, np.array(common)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    results = {}

    gen_ad, Graw = load_gen()
    wsi_t = ad.read_h5ad(REPO / "data/tcga_kirc/wsi/wsi_tumor.h5ad")
    wsi_n = ad.read_h5ad(REPO / "data/tcga_kirc/wsi/wsi_normal.h5ad")

    # ---- (1) head-to-head on the SAME expression target -------------------
    # WSI tumour slides vs CT organ radiomics, identical pipeline/budget.
    Gw, Xw, cw = align(gen_ad, Graw, wsi_t)
    r_wsi = recover(prep_expression(Gw), prep_radiomics(Xw, signed_log=False)[0])
    results["wsi_tumor_expr"] = {**r_wsi, "n_feat": int(Xw.shape[1]),
                                 "label": "WSI histopathology (tumour slides)"}
    print(f"WSI  tumour x expr : n={r_wsi['n']} p*={r_wsi['ps']}  "
          f"R1_cv={r_wsi['cv1']:.3f}  CV-perm p={np.round(r_wsi['p'],3)}  "
          f"eff-rank={r_wsi['eff_rank']}  MI(top3)={r_wsi['mi_bits']:.2f}b")

    # CT radiology on the same harness (re-derived here so the two share code).
    Gr, Xr, _ = load_pair(GEN, "data/tcga_kirc/imaging/organ_radiomics.h5ad",
                          gen_layer="tpm_unstranded")[:3]
    r_ct = recover(prep_expression(Gr), prep_radiomics(Xr, signed_log=False)[0])
    results["ct_radiomics_expr"] = {**r_ct, "n_feat": int(Xr.shape[1]),
                                    "label": "CT organ radiomics"}
    print(f"CT   radiomics x expr: n={r_ct['n']} p*={r_ct['ps']}  "
          f"R1_cv={r_ct['cv1']:.3f}  CV-perm p={np.round(r_ct['p'],3)}  "
          f"eff-rank={r_ct['eff_rank']}  MI(top3)={r_ct['mi_bits']:.2f}b")

    # ---- (2) tumour vs adjacent-normal histology, size-matched ------------
    Gt, Xt, ct = align(gen_ad, Graw, wsi_t)
    Gn, Xn, cn = align(gen_ad, Graw, wsi_n)
    rt = recover(prep_expression(Gt), prep_radiomics(Xt, signed_log=False)[0])
    rn = recover(prep_expression(Gn), prep_radiomics(Xn, signed_log=False)[0])

    nmin = min(rt["n"], rn["n"])

    def subsample(G, X, nmin, reps=20):
        vals = []
        for i in range(reps):
            rng = np.random.default_rng(SEED + i)
            sel = rng.choice(len(G), nmin, replace=False)
            vals.append(recover(prep_expression(G[sel]),
                                prep_radiomics(X[sel], signed_log=False)[0],
                                n_perm=0)["cv1"])
        return float(np.mean(vals)), float(np.std(vals))

    rt_m = (rt["cv1"], 0.0) if rt["n"] == nmin else subsample(Gt, Xt, nmin)
    rn_m = (rn["cv1"], 0.0) if rn["n"] == nmin else subsample(Gn, Xn, nmin)
    results["tumor_normal"] = {
        "tumor": {**rt, "label": "tumour slides"},
        "normal": {**rn, "label": "adjacent-normal slides"},
        "n_matched": int(nmin),
        "tumor_cv1_matched": rt_m[0], "tumor_cv1_matched_sd": rt_m[1],
        "normal_cv1_matched": rn_m[0], "normal_cv1_matched_sd": rn_m[1],
    }
    print(f"\ntumour  x expr: n={rt['n']}  R1_cv={rt['cv1']:.3f}  eff-rank={rt['eff_rank']}")
    print(f"normal  x expr: n={rn['n']}  R1_cv={rn['cv1']:.3f}  eff-rank={rn['eff_rank']}")
    print(f"size-matched (n={nmin}): tumour R1_cv={rt_m[0]:.3f}+/-{rt_m[1]:.3f}  "
          f"normal R1_cv={rn_m[0]:.3f}+/-{rn_m[1]:.3f}")

    json.dump(results, open(FIGDIR / "stats.json", "w"), indent=2)

    # ---- figures ----------------------------------------------------------
    _fig_headtohead(results, FIGDIR / "wsi_vs_ct.pdf")
    _fig_tumor_normal(results, FIGDIR / "tumor_vs_normal.pdf")
    print(f"\nWrote stats + figures to {FIGDIR.relative_to(REPO)}")


def _fig_headtohead(res, path):
    ct, ws = res["ct_radiomics_expr"], res["wsi_tumor_expr"]
    x = np.arange(K); w = 0.38
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.bar(x - w / 2, ct["cv"], w, label=f"CT organ radiomics "
           f"($R_1^{{cv}}$={ct['cv1']:.3f}, n={ct['n']})", color="#4C72B0")
    ax.bar(x + w / 2, ws["cv"], w, label=f"WSI histopathology "
           f"($R_1^{{cv}}$={ws['cv1']:.3f}, n={ws['n']})", color="#55A868")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"dir {i+1}" for i in range(K)])
    ax.set_ylabel(r"honest $\hat R_i^{\mathrm{cv}}$")
    ax.set_title("TCGA-KIRC: is expression more reflected in\nWSI histopathology or CT radiology?")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def _fig_tumor_normal(res, path):
    tn = res["tumor_normal"]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    labels = ["all cases", f"size-matched (n={tn['n_matched']})"]
    x = np.arange(2); w = 0.38
    tum = [tn["tumor"]["cv1"], tn["tumor_cv1_matched"]]
    nor = [tn["normal"]["cv1"], tn["normal_cv1_matched"]]
    ax.bar(x - w / 2, tum, w, label=f"tumour (n={tn['tumor']['n']})", color="#C44E52")
    ax.bar(x + w / 2, nor, w, label=f"adjacent-normal (n={tn['normal']['n']})",
           color="#4C72B0")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(r"leading $\hat R_1^{\mathrm{cv}}$ (expr from WSI)")
    ax.set_title("Is tumour-derived expression more reflected in\ntumour than adjacent-normal histology?")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
