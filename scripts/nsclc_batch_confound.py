"""Scanner/site batch-confound check for the NSCLC radiogenomic signal.

Tests whether the leading image-identifiable genomic direction merely encodes a
scanner/reconstruction batch, via three controls:
  (1) regress manufacturer+kernel out of both modalities, refit;
  (2) batch-stratified permutation (permute genomic labels within
      manufacturer x kernel blocks, preserving batch structure under the null);
  (3) restrict to the largest homogeneous scanner+kernel stratum.

Batch covariates are read from the per-series DICOM JSON sidecars under
data/nsclc/imaging/nifti/<series>/*.json (Manufacturer, ManufacturersModelName,
ConvolutionKernel). Run from the repo root: python scripts/nsclc_batch_confound.py
"""
from __future__ import annotations

import sys, json, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np, pandas as pd, anndata as ad, scanpy as sc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from rgit import cross_validated_recoverability, cv_permutation_test, gaussian_rank_transform, to_dense

REPO = Path(__file__).parent.parent


def load():
    gen = ad.read_h5ad(REPO / "data/nsclc/genomics/gene_expression.h5ad")
    img = ad.read_h5ad(REPO / "data/nsclc/imaging/organ_radiomics_subset.h5ad")
    common = sorted(set(gen.obs_names) & set(img.obs_names))
    gen, img = gen[common].copy(), img[common].copy()
    g = gen.copy(); M = np.nan_to_num(to_dense(g.X).astype(np.float32), nan=0.0); g.X = M
    g = g[:, (M > 0).sum(0) >= 3].copy()
    sc.pp.normalize_total(g, target_sum=1e6); g.X = np.log1p(g.X)
    sc.pp.highly_variable_genes(g, n_top_genes=2000, subset=True)
    G = gaussian_rank_transform(to_dense(g.X)); X = gaussian_rank_transform(to_dense(img.X))
    return G, X, np.array(common)


def batch_blocks(common):
    meta = pd.read_csv(REPO / "data/nsclc/imaging/metadata_subset.csv")
    sid = meta.drop_duplicates("patient_id").set_index("patient_id")["series_id"].to_dict()
    manu, kern = {}, {}
    for pid, s in sid.items():
        js = glob.glob(str(REPO / f"data/nsclc/imaging/nifti/{s}/*.json"))
        if not js:
            continue
        try:
            d = json.loads(open(js[0], encoding="utf-8", errors="replace").read().replace("\x00", ""))
            manu[pid] = (d.get("Manufacturer") or "NA")
            k = d.get("ConvolutionKernel") or "NA"
            kern[pid] = k[0] if isinstance(k, list) else k
        except Exception:
            pass
    man = np.array([manu.get(p, "NA") for p in common])
    kr = np.array([str(kern.get(p, "NA")) for p in common])
    kg = np.array([k if k in ("STANDARD", "BONEPLUS", "SOFT") else ("B" if k.startswith("B") else "o") for k in kr])
    return man, kg, np.array([f"{m}|{k}" for m, k in zip(man, kg)])


def cvr(G, X, seed=0, K=3):
    ps = max(3, min(len(G) // 5, X.shape[1], len(G) - 2))
    Gs = PCA(ps, random_state=0).fit_transform(StandardScaler().fit_transform(G))
    Xs = PCA(min(ps, X.shape[1]), random_state=0).fit_transform(StandardScaler().fit_transform(X))
    return cross_validated_recoverability(Gs, Xs, n_components=K, n_folds=5, random_state=seed).mean(0)


def main():
    G, X, common = load()
    man, kg, block = batch_blocks(common)
    ps = len(common) // 5
    Gs = PCA(ps, random_state=0).fit_transform(StandardScaler().fit_transform(G))
    Xs = PCA(min(ps, X.shape[1]), random_state=0).fit_transform(StandardScaler().fit_transform(X))

    D = pd.get_dummies(pd.DataFrame({"m": man, "k": kg}), drop_first=True).values.astype(float)
    D = np.column_stack([np.ones(len(D)), D])
    resid = lambda Mx: Mx - D @ np.linalg.lstsq(D, Mx, rcond=None)[0]
    obs, null, p = cv_permutation_test(resid(Gs), resid(Xs), n_components=3, n_perm=300, random_state=0)
    print(f"(1) both-residualized: CV-R1={obs[0]:.3f}  perm p={p[0]:.3f}")

    o1 = cvr(G, X)[0]
    rng = np.random.default_rng(0); NP = 300; nb = np.zeros(NP); idx = np.arange(len(common))
    for b in range(NP):
        perm = idx.copy()
        for blk in np.unique(block):
            m = np.where(block == blk)[0]; perm[m] = rng.permutation(m)
        nb[b] = cvr(G[perm], X)[0]
    print(f"(2) batch-stratified perm: obs={o1:.3f}  null mean={nb.mean():.3f} q95={np.quantile(nb,.95):.3f}  "
          f"p={(1+np.sum(nb>=o1))/(1+NP):.3f}")

    big = pd.Series(block).value_counts().idxmax()
    m = block == big
    print(f"(3) within {big} (n={m.sum()}): CV-R1={cvr(G[m], X[m])[0]:.3f}")


if __name__ == "__main__":
    main()
