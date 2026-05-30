"""Data loading, synthetic fallback, and preprocessing.

This module ports the notebook's "load" and "align + preprocess" sections
(`load_data`, the alignment/standardization/PCA cell, and the confounder cell)
into importable functions, so the matrices the recoverability model is fit on
can be reproduced from a :class:`~rgit.config.RecoverabilityConfig` alone.

The headline entry points are:

* :func:`load_modalities` -- read the two ``.h5ad`` files, or draw the synthetic
  probabilistic-CCA dataset when they are absent.
* :func:`preprocess` -- align patients, apply the (Bernoulli/Gaussian) copula
  transforms, standardize, and PCA-reduce each modality to its working
  dimension, returning a :class:`Preprocessed` bundle that downstream analysis
  consumes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as _st
from sklearn.preprocessing import StandardScaler

from rgit.config import RecoverabilityConfig
from rgit.model import (
    gaussian_rank_transform,
    make_synthetic_radiogenomics,
    to_dense,
)

logger = logging.getLogger("rgit")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _to_dense_numeric(X) -> np.ndarray:
    """Dense float32 with NaNs treated as unmeasured (-> 0), as the notebook does."""
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0)


def _select_hvg(M: np.ndarray, n_top: int) -> np.ndarray:
    """Indices of the ``n_top`` highest-variance columns (variance-based HVG)."""
    if M.shape[1] <= n_top:
        return np.arange(M.shape[1])
    return np.argsort(M.var(axis=0))[::-1][:n_top]


def _load_real(cfg: RecoverabilityConfig):
    """Read the two ``.h5ad`` files and apply expression preprocessing.

    Returns ``(genomics_adata, imaging_adata, None)``. Requires ``anndata``;
    ``scanpy`` is used for HVG/CPM only when available and requested.
    """
    import anndata as ad

    gen = ad.read_h5ad(cfg.genomics_h5ad)
    img = ad.read_h5ad(cfg.imaging_h5ad)

    common = gen.obs_names.intersection(img.obs_names)
    gen = gen[common].copy()
    img = img[common].copy()

    layer = cfg.expression_layer
    X = gen.layers[layer] if layer in getattr(gen, "layers", {}) else gen.X
    X = _to_dense_numeric(X)
    gen.X = X

    if cfg.min_expressed_patients > 0:
        keep = (X > 0).sum(axis=0) >= cfg.min_expressed_patients
        keep = np.asarray(keep).ravel()
        gen = gen[:, keep].copy()

    use_scanpy = cfg.hvg_method != "variance" and (cfg.do_cpm_normalization or cfg.do_hvg_selection)
    sc = None
    if use_scanpy:
        try:
            import scanpy as sc  # noqa: F401
        except ImportError:
            if cfg.hvg_method == "scanpy":
                raise ImportError(
                    "hvg_method='scanpy' requires scanpy; install rgit[processing] "
                    "or set hvg_method='variance'."
                )
            sc = None

    if cfg.do_cpm_normalization:
        if sc is not None:
            sc.pp.normalize_total(gen, target_sum=1e6, inplace=True)
        else:
            M = to_dense(gen.X)
            lib = M.sum(axis=1, keepdims=True)
            lib[lib == 0] = 1.0
            gen.X = M / lib * 1e6

    if cfg.do_log1p:
        gen.X = np.log1p(to_dense(gen.X))

    if cfg.do_hvg_selection:
        if sc is not None:
            sc.pp.highly_variable_genes(gen, n_top_genes=cfg.n_top_genes, subset=True)
        else:
            idx = _select_hvg(to_dense(gen.X), cfg.n_top_genes)
            gen = gen[:, idx].copy()

    logger.info("loaded genomics %s and imaging %s from disk", gen.shape, img.shape)
    return gen, img, None


def make_synthetic_modalities(cfg: RecoverabilityConfig):
    """Synthetic genomics/imaging AnnData with known ground truth.

    Mirrors the notebook's synthetic fallback: a probabilistic-CCA draw wrapped
    in two ``AnnData`` objects, with the imaging patient order deliberately
    shuffled so the alignment step is exercised.

    Returns ``(genomics_adata, imaging_adata, truth)``.
    """
    import anndata as ad

    G, X, truth = make_synthetic_radiogenomics(
        n=cfg.synthetic_n,
        p=cfg.synthetic_p,
        d=cfg.synthetic_d,
        k=cfg.synthetic_k,
        random_state=cfg.seed,
        genomics_type=("gaussian" if cfg.genomics_data_type == "expression" else "bernoulli"),
    )
    patients = np.array([f"P{i:04d}" for i in range(G.shape[0])])
    genes = [f"GENE{i:04d}" for i in range(G.shape[1])]
    feats = [f"radiomic_{i:03d}" for i in range(X.shape[1])]

    gen = ad.AnnData(X=G, obs=pd.DataFrame(index=patients), var=pd.DataFrame(index=genes))
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(G.shape[0])
    img = ad.AnnData(
        X=X[order], obs=pd.DataFrame(index=patients[order]), var=pd.DataFrame(index=feats)
    )
    logger.info(
        "synthetic genomics %s and imaging %s (true shared rank k=%d)",
        gen.shape,
        img.shape,
        truth["k"],
    )
    return gen, img, truth


def load_modalities(cfg: RecoverabilityConfig):
    """Load the two modalities, or draw the synthetic dataset if files are absent.

    Returns ``(genomics_adata, imaging_adata, truth_or_None)``.
    """
    if cfg.is_synthetic():
        return make_synthetic_modalities(cfg)
    return _load_real(cfg)


# --------------------------------------------------------------------------
# preprocessing result
# --------------------------------------------------------------------------
@dataclass
class Preprocessed:
    """Patient-aligned, transformed, and dimension-reduced modalities.

    ``G_model``/``X_model`` are the matrices the recoverability model is fit on.
    The remaining fields carry everything the downstream analyses need:
    pre-transform marginals (for the assumption checks), the binary mutation
    matrix and per-gene latent-Gaussian summary (variant mode), the fitted PCA
    objects (to back-project directions to gene space), and the raw working
    matrices plus optional confounder design (for the deconfounding analyses).
    """

    shared: pd.Index
    n_samples: int

    G_model: np.ndarray
    X_model: np.ndarray
    n_components: int

    G_std: np.ndarray
    X_std: np.ndarray
    G_premarg: np.ndarray
    X_premarg: np.ndarray

    pca_g: object
    pca_x: object
    gen_space: str
    img_space: str

    gene_names: np.ndarray
    feat_names: np.ndarray

    # variant mode
    G_bin_global: Optional[np.ndarray] = None
    variant_pergene: Optional[dict] = None

    # deconfounding
    G_model_raw: np.ndarray = None
    X_model_raw: np.ndarray = None
    C_full: Optional[np.ndarray] = None
    conf_cols: list = field(default_factory=list)

    # accumulated stats fragments produced during preprocessing
    stats: dict = field(default_factory=dict)

    def to_gene_loadings(self, dirs: np.ndarray) -> np.ndarray:
        """Map directions in genomic working space back to standardized gene space."""
        return self.pca_g.components_.T @ dirs if self.pca_g is not None else dirs

    def to_feature_loadings(self, dirs: np.ndarray) -> np.ndarray:
        """Map directions in imaging working space back to standardized feature space."""
        return self.pca_x.components_.T @ dirs if self.pca_x is not None else dirs


def _bernoulli_copula_transform(G_raw: np.ndarray, n_samples: int):
    """Latent-Gaussian surrogate for a binary mutation matrix (manuscript copula).

    Replaces each binary call by the truncated-normal conditional mean of the
    latent coordinate, whose covariance is the first-order tetrachoric estimate
    of the latent ``Sigma_Z``. Returns ``(G_surrogate, G_bin, pergene)``.
    """
    G_bin = (G_raw > 0).astype(np.int8)
    p_hat = G_bin.mean(axis=0).astype(float)
    eps = 0.5 / max(n_samples, 2)
    p_hat = np.clip(p_hat, eps, 1.0 - eps)
    tau = _st.norm.ppf(1.0 - p_hat)
    phi_tau = _st.norm.pdf(tau)
    z_pos = phi_tau / p_hat
    z_neg = -phi_tau / (1.0 - p_hat)
    G_surrogate = np.where(G_bin == 1, z_pos[None, :], z_neg[None, :]).astype(float)
    h_z = 0.5 * np.log(2 * np.pi * np.e)
    h_g = -(p_hat * np.log(p_hat) + (1 - p_hat) * np.log(1 - p_hat))
    bin_gap = h_z - h_g
    pergene = {"p_hat": p_hat, "tau": tau, "bin_gap": bin_gap}
    return G_surrogate, G_bin, pergene


def residualize(M: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Residuals of the columns of ``M`` after least-squares regression on ``[1, z(C)]``."""
    D = np.column_stack([np.ones(len(C)), StandardScaler().fit_transform(C)])
    beta, *_ = np.linalg.lstsq(D, M, rcond=None)
    return M - D @ beta


def load_confounders(cfg: RecoverabilityConfig, genomics_adata, patient_ids):
    """ADNI-style confounder design matrix aligned to ``patient_ids``.

    Returns ``(DataFrame[confounder_cols] or None, used_cols)``. Returns
    ``(None, [])`` when no demographics CSV is configured or present.
    """
    if cfg.demographics_csv is None or not Path(cfg.demographics_csv).exists():
        return None, []
    demo = pd.read_csv(cfg.demographics_csv, low_memory=False).drop_duplicates("PTID").set_index("PTID")
    yob = pd.to_datetime(demo["PTDOBYY"], errors="coerce").dt.year
    yob = yob.fillna(pd.to_numeric(demo["PTDOBYY"], errors="coerce"))
    rows = {}
    for pid in patient_ids:
        if pid not in demo.index:
            continue
        try:
            ycoll = float(str(genomics_adata.obs.loc[pid, "YearofCollection"]))
        except Exception:
            ycoll = 2011.0
        rows[pid] = {
            "age": ycoll - float(yob.get(pid, np.nan)),
            "sex": float(demo.loc[pid, "PTGENDER"]),
            "education": pd.to_numeric(demo.loc[pid, "PTEDUCAT"], errors="coerce"),
        }
    C = pd.DataFrame.from_dict(rows, orient="index").replace({-4: np.nan})
    return C[cfg.confounder_cols], cfg.confounder_cols


def preprocess(genomics_adata, imaging_adata, cfg: RecoverabilityConfig) -> Preprocessed:
    """Align, copula-transform, standardize, and PCA-reduce both modalities.

    This is the package version of the notebook's alignment/preprocessing and
    confounder cells. The returned :class:`Preprocessed` is everything the
    analysis stage needs.
    """
    stats: dict = {}

    # 1 - align on shared patients
    shared = genomics_adata.obs_names.intersection(imaging_adata.obs_names)
    shared = pd.Index(sorted(shared))
    if cfg.downsample_n is not None and cfg.downsample_n < len(shared):
        ds_rng = np.random.default_rng(cfg.downsample_seed)
        shared = pd.Index(
            sorted(ds_rng.choice(np.asarray(shared), size=cfg.downsample_n, replace=False))
        )
        logger.info("downsample_n=%d: main analysis restricted to %d patients",
                    cfg.downsample_n, len(shared))
    genomics_adata = genomics_adata[shared].copy()
    imaging_adata = imaging_adata[shared].copy()
    assert (genomics_adata.obs_names == imaging_adata.obs_names).all()
    n_samples = len(shared)

    # 2 - copula transforms
    G_raw = to_dense(genomics_adata.X)
    X_raw = to_dense(imaging_adata.X)

    G_bin_global = None
    variant_pergene = None
    if cfg.genomics_data_type == "variant":
        G_raw, G_bin_global, variant_pergene = _bernoulli_copula_transform(G_raw, n_samples)
        p_hat = variant_pergene["p_hat"]
        bin_gap = variant_pergene["bin_gap"]
        stats["bernoulli_copula"] = {
            "n_genes": int(G_raw.shape[1]),
            "mean_mut_freq": float(p_hat.mean()),
            "median_mut_freq": float(np.median(p_hat)),
            "mean_bin_gap_nats": float(bin_gap.mean()),
        }

    G_premarg = G_raw.copy()
    X_premarg = X_raw.copy()

    if cfg.use_copula:
        X_raw = gaussian_rank_transform(X_raw)
        if cfg.genomics_data_type == "expression":
            G_raw = gaussian_rank_transform(G_raw)

    G_std = StandardScaler().fit_transform(G_raw)
    X_std = StandardScaler().fit_transform(X_raw)

    # 3 - adaptive PCA reduction to the working dimension
    from sklearn.decomposition import PCA

    def _reduce(M, name):
        p_native = M.shape[1]
        budget = max(cfg.min_working_dim, n_samples // cfg.samples_per_dim)
        target = min(p_native, n_samples - 1, budget)
        if target < p_native:
            pca = PCA(n_components=target, random_state=cfg.seed).fit(M)
            return pca.transform(M), pca, f"{target} {name} PCs"
        return M, None, f"standardized {name} features"

    G_model, pca_g, gen_space = _reduce(G_std, "genomic")
    X_model, pca_x, img_space = _reduce(X_std, "imaging")

    n_components = max(1, min(cfg.n_components_max, G_model.shape[1], X_model.shape[1], n_samples - 1))

    total_dim = G_model.shape[1] + X_model.shape[1]
    if total_dim >= n_samples:
        logger.warning(
            "dim(G)+dim(X) >= n -- in-sample CCA saturates by geometry; trust the "
            "cross-validated spectrum and permutation test, not in-sample numbers."
        )
    elif total_dim > n_samples // 2:
        logger.info("dim(G)+dim(X) > n/2 -- in-sample canonical correlations are biased upward.")

    stats.update({
        "n_samples": int(n_samples),
        "p_native_genes": int(G_raw.shape[1]),
        "d_native_feats": int(X_raw.shape[1]),
        "dim_g_model": int(G_model.shape[1]),
        "dim_x_model": int(X_model.shape[1]),
        "n_components": int(n_components),
    })

    # 4 - confounders (optional)
    C_full = None
    conf_cols: list = []
    C_df, conf_cols_loaded = load_confounders(cfg, genomics_adata, list(shared))
    if C_df is not None:
        C_full = C_df.reindex(shared)
        C_full = C_full.fillna(C_full.mean()).values.astype(float)
        conf_cols = conf_cols_loaded
        stats["confounders"] = {
            "cols": conf_cols,
            "n_with_confounders": int(C_df.dropna(how="any").shape[0]),
            "n_total": int(len(shared)),
        }

    G_model_raw, X_model_raw = G_model.copy(), X_model.copy()
    if cfg.deconfound and C_full is not None:
        G_model = residualize(G_model, C_full)
        X_model = residualize(X_model, C_full)
        logger.info("deconfound=True: main analysis uses features residualized on %s", conf_cols)

    return Preprocessed(
        shared=shared,
        n_samples=n_samples,
        G_model=G_model,
        X_model=X_model,
        n_components=n_components,
        G_std=G_std,
        X_std=X_std,
        G_premarg=G_premarg,
        X_premarg=X_premarg,
        pca_g=pca_g,
        pca_x=pca_x,
        gen_space=gen_space,
        img_space=img_space,
        gene_names=np.asarray(genomics_adata.var_names),
        feat_names=np.asarray(imaging_adata.var_names),
        G_bin_global=G_bin_global,
        variant_pergene=variant_pergene,
        G_model_raw=G_model_raw,
        X_model_raw=X_model_raw,
        C_full=C_full,
        conf_cols=conf_cols,
        stats=stats,
    )
