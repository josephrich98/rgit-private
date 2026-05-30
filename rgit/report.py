"""End-to-end recoverability analysis -- the notebook as a function.

:func:`run_recoverability_analysis` reproduces the full pipeline of
``notebooks/radiogenomic_recoverability.ipynb`` headlessly: load (or synthesize)
the two modalities, preprocess, fit the linear-Gaussian recoverability model,
run every diagnostic (assumption checks, imaging-variance decomposition,
held-out posterior recovery, gene loadings, per-gene AUC, cross-validation,
permutation test, MI upper bound + effective rank, optional sweeps, synthetic
ground-truth validation), and write ``stats.json`` plus figures to the output
directory.

The point of the module is reproducibility: ``pip install rgit`` then call this
once (or run the ``rgit-recoverability`` console script) and you get the same
numbers the manuscript quotes, without opening a notebook.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as _st

import rgit
from rgit.config import RecoverabilityConfig
from rgit.datasets import Preprocessed, load_modalities, preprocess, residualize
from rgit.model import RecoverabilityFit

logger = logging.getLogger("rgit")


# --------------------------------------------------------------------------
# figure helper (lazy, headless, optional)
# --------------------------------------------------------------------------
class _Figures:
    """Save matplotlib figures to the output dir, no-op when disabled/unavailable."""

    def __init__(self, cfg: RecoverabilityConfig):
        self.cfg = cfg
        self.enabled = bool(cfg.save_figures)
        self.plt = None
        if self.enabled:
            try:
                import matplotlib

                matplotlib.use("Agg")  # headless
                import matplotlib.pyplot as plt

                plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
                self.plt = plt
                cfg.figures_dir.mkdir(parents=True, exist_ok=True)
            except ImportError:
                logger.warning("matplotlib unavailable -- figures will not be written")
                self.enabled = False

    def subplots(self, *args, **kwargs):
        return self.plt.subplots(*args, **kwargs)

    def save(self, name: str, fig) -> None:
        if not self.enabled:
            return
        for ext in self.cfg.figure_formats:
            fig.savefig(self.cfg.figures_dir / f"{name}.{ext}", dpi=self.cfg.figure_dpi,
                        bbox_inches="tight")
        self.plt.close(fig)
        logger.info("saved figure: %s", self.cfg.figures_dir / name)


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
def mvn_diagnostic(M: np.ndarray, name: str = "", ax=None) -> Optional[dict]:
    """MVN check via Mahalanobis^2 vs chi^2_p, with standardized Mardia effect sizes."""
    n, p = M.shape
    if n <= p:
        logger.info("MVN diagnostic %s skipped (n=%d <= p=%d, singular)", name, n, p)
        return None
    Mc = M - M.mean(axis=0)
    S = (Mc.T @ Mc) / n
    S_inv = np.linalg.pinv(S)
    Gram = Mc @ S_inv @ Mc.T
    d2 = np.diag(Gram)
    b1p = np.sum(Gram ** 3) / n ** 2
    b2p = np.sum(d2 ** 2) / n
    skew_ratio = b1p / (p * (p + 2) * (p + 4))
    kurt_ratio = b2p / (p * (p + 2))
    q99 = _st.chi2.ppf(0.99, df=p)
    tail_frac = float(np.mean(d2 > q99))
    if ax is not None:
        probs = (np.arange(1, n + 1) - 0.5) / n
        q_theo = _st.chi2.ppf(probs, df=p)
        q_emp = np.sort(d2)
        ax.scatter(q_theo, q_emp, s=8, alpha=0.5, color="#4C72B0")
        lim = [0, max(q_theo.max(), q_emp.max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel(f"chi^2_{p} theoretical quantile")
        ax.set_ylabel("empirical squared Mahalanobis distance")
        ax.set_title(name, fontsize=9)
    return dict(skew_ratio=skew_ratio, kurt_ratio=kurt_ratio, tail_frac=tail_frac, d2=d2)


def _marg_summary(M: np.ndarray) -> dict:
    sk = _st.skew(M, axis=0)
    ku = _st.kurtosis(M, axis=0)
    frac_bad = float(np.mean((np.abs(sk) > 2) | (np.abs(ku) > 7)))
    return dict(
        frac_nonnormal=frac_bad,
        median_abs_skew=float(np.median(np.abs(sk))),
        median_abs_kurt=float(np.median(np.abs(ku))),
    )


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------
@dataclass
class RecoverabilityReport:
    """Programmatic result of :func:`run_recoverability_analysis`."""

    stats: dict
    fit: RecoverabilityFit
    pre: Preprocessed
    truth: Optional[dict]
    config: RecoverabilityConfig
    stats_path: Optional[Path]


def run_recoverability_analysis(cfg: Optional[RecoverabilityConfig] = None,
                                **overrides) -> RecoverabilityReport:
    """Run the full recoverability pipeline and return a :class:`RecoverabilityReport`.

    Parameters
    ----------
    cfg : RecoverabilityConfig, optional
        Full configuration. If omitted, a default config is built and any
        keyword ``overrides`` are applied (e.g. ``genomics_h5ad=...``).
    **overrides
        Field overrides applied on top of ``cfg`` (or the default config).

    Returns
    -------
    RecoverabilityReport
        Holds the accumulated ``stats`` dict (also written to ``stats.json``),
        the fitted model, the preprocessed modalities, and the synthetic
        ``truth`` (or ``None`` on real data).
    """
    if cfg is None:
        cfg = RecoverabilityConfig(**overrides)
    elif overrides:
        cfg = cfg.model_copy(update=overrides)

    np.random.default_rng(cfg.seed)
    figs = _Figures(cfg)

    # ---- load + preprocess ------------------------------------------------
    genomics_adata, imaging_adata, truth = load_modalities(cfg)
    pre = preprocess(genomics_adata, imaging_adata, cfg)

    stats: dict = {
        "genomics_h5ad": str(cfg.genomics_h5ad),
        "imaging_h5ad": str(cfg.imaging_h5ad),
        "genomics_data_type": cfg.genomics_data_type,
        "use_synthetic": cfg.is_synthetic(),
    }
    stats.update(pre.stats)

    G_model, X_model = pre.G_model, pre.X_model
    nc = pre.n_components

    # ---- assumption checks ------------------------------------------------
    _assumption_checks(cfg, pre, stats, figs)

    # ---- binarization gap (variant) --------------------------------------
    if pre.variant_pergene is not None and figs.enabled:
        _binarization_gap_figure(pre, figs)

    # ---- fit + spectrum ---------------------------------------------------
    fit = rgit.fit_recoverability(G_model, X_model, n_components=nc)
    mi_nats = fit.mutual_information()
    logger.info("I(G;X) = %.2f nats (%.2f bits)", mi_nats, mi_nats / np.log(2))
    stats.update({
        "in_sample_rho": [float(r) for r in fit.rho],
        "in_sample_recoverability": [float(r) for r in fit.recoverability],
        "in_sample_MI_nats": float(mi_nats),
        "in_sample_MI_bits": float(mi_nats / np.log(2)),
    })

    # ---- imaging variance decomposition ----------------------------------
    _imaging_variance(cfg, pre, fit, stats, figs)

    # ---- spectrum + MI figures -------------------------------------------
    if figs.enabled:
        _spectrum_figure(fit, truth, nc, figs)
        _mi_per_direction_figure(fit, figs)

    # ---- held-out posterior recovery -------------------------------------
    _heldout_recovery(cfg, pre, stats, figs)

    # ---- gene loadings ----------------------------------------------------
    _gene_loadings(pre, fit, stats, figs)

    # ---- per-gene AUC -----------------------------------------------------
    _per_gene_auc(cfg, pre, fit, stats, figs)

    # ---- cross-validated recoverability ----------------------------------
    cv = rgit.cross_validated_recoverability(
        G_model, X_model, n_components=nc, n_folds=cfg.n_folds, random_state=cfg.seed)
    cv_mean, cv_std = cv.mean(axis=0), cv.std(axis=0)
    stats.update({
        "cv_recoverability_mean": [float(x) for x in cv_mean],
        "cv_recoverability_std": [float(x) for x in cv_std],
    })
    if figs.enabled:
        _cv_figure(fit, cv_mean, cv_std, truth, nc, cfg, figs)

    # ---- permutation test -------------------------------------------------
    observed, null, pvalues = rgit.permutation_test(
        G_model, X_model, n_components=min(3, nc), n_perm=cfg.n_perm, random_state=cfg.seed)
    stats["permutation"] = {
        "n_perm": int(cfg.n_perm),
        "observed": [float(x) for x in observed],
        "p_values": [float(x) for x in pvalues],
        "null_rho1_mean": float(null[:, 0].mean()),
        "null_rho1_q95": float(np.quantile(null[:, 0], 0.95)),
        "null_rho1_q99": float(np.quantile(null[:, 0], 0.99)),
    }
    if figs.enabled:
        _permutation_figure(observed, null, cfg, figs)

    # ---- MI upper bound + effective rank ---------------------------------
    _mi_upper_bound(cfg, pre, observed, null, stats, figs)

    # ---- deconfound comparison -------------------------------------------
    if pre.C_full is not None and not cfg.is_synthetic():
        _deconfound_comparison(cfg, pre, stats, figs)

    # ---- sweeps -----------------------------------------------------------
    if cfg.run_sample_complexity_sweep:
        _sample_complexity_sweep(cfg, pre, stats, figs)
    if cfg.run_downsample_sweep and not cfg.is_synthetic():
        _downsample_sweep(cfg, pre, stats, figs)

    # ---- synthetic ground-truth validation -------------------------------
    if truth is not None:
        _synthetic_validation(pre, fit, cv_mean, truth, stats, figs)

    # ---- write outputs ----------------------------------------------------
    stats_path = None
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = cfg.stats_path
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=float)
    with open(cfg.config_path, "w") as f:
        f.write(cfg.model_dump_json(indent=2))
    logger.info("wrote %s (keys: %s)", stats_path, list(stats.keys()))

    return RecoverabilityReport(stats=stats, fit=fit, pre=pre, truth=truth,
                                config=cfg, stats_path=stats_path)


# --------------------------------------------------------------------------
# individual sections (kept as private helpers for readability/testability)
# --------------------------------------------------------------------------
def _assumption_checks(cfg, pre, stats, figs):
    GX_joint = np.column_stack([pre.G_model, pre.X_model])
    if figs.enabled:
        fig, ax = figs.subplots(figsize=(5, 4))
        mvn_joint = mvn_diagnostic(GX_joint, f"joint [G|X] ({GX_joint.shape[1]} dims)", ax=ax)
        fig.suptitle("Joint MVN check on concatenated working-space scores", y=1.02)
        figs.save("mvn_joint_diagnostic", fig)
    else:
        mvn_joint = mvn_diagnostic(GX_joint, "joint [G|X]")

    marg = {
        "genomics_raw": _marg_summary(pre.G_premarg),
        "genomics_post": _marg_summary(pre.G_std),
        "imaging_raw": _marg_summary(pre.X_premarg),
        "imaging_post": _marg_summary(pre.X_std),
    }
    dcor_obs, dcor_null, dcor_p = rgit.distance_correlation_test(
        pre.G_model, pre.X_model, n_perm=cfg.n_perm, random_state=cfg.seed)
    stats["assumption_checks"] = {
        "mvn_joint": ({k: float(v) for k, v in mvn_joint.items() if k != "d2"}
                      if mvn_joint else None),
        "per_feature_marginals": marg,
        "distance_correlation": {
            "observed": float(dcor_obs),
            "null_mean": float(dcor_null.mean()),
            "null_q95": float(np.quantile(dcor_null, 0.95)),
            "pvalue": float(dcor_p),
        },
    }


def _binarization_gap_figure(pre, figs):
    p_hat = pre.variant_pergene["p_hat"]
    bin_gap = pre.variant_pergene["bin_gap"]
    fig, ax = figs.subplots(figsize=(6, 4))
    ax.scatter(p_hat, bin_gap, s=8, alpha=0.5, color="#4C72B0")
    pp = np.linspace(0.001, 0.999, 400)
    hb = -(pp * np.log(pp) + (1 - pp) * np.log(1 - pp))
    ax.plot(pp, 0.5 * np.log(2 * np.pi * np.e) - hb, color="#C44E52", lw=1.2,
            label=r"$\frac{1}{2}\log(2\pi e) - h_2(p)$")
    ax.set_xlabel("mutation frequency $p_j$")
    ax.set_ylabel(r"$H(Z_j) - H(G_j)$  (nats)")
    ax.set_title("Per-gene binarization gap")
    ax.legend(loc="upper right", fontsize=9)
    figs.save("binarization_gap", fig)


def _imaging_variance(cfg, pre, fit, stats, figs):
    frac, _ = rgit.imaging_variance_explained(fit)
    cv_for_frac = rgit.cross_validated_recoverability(
        pre.G_model, pre.X_model, n_components=pre.n_components,
        n_folds=cfg.n_folds, random_state=cfg.seed).mean(axis=0)
    frac_cv = float(np.clip(cv_for_frac, 0, 1).sum()
                    / max(1, min(pre.G_model.shape[1], pre.X_model.shape[1])))
    stats.update({
        "imaging_var_share_genomics_insample": float(frac),
        "imaging_var_share_genomics_cv": float(frac_cv),
    })
    if figs.enabled:
        fig, axes = figs.subplots(2, 1, figsize=(5, 2.4))
        for ax, value, title in [(axes[0], frac, "in-sample (biased)"),
                                 (axes[1], frac_cv, "cross-validated")]:
            ax.barh([0], [value], color="#4C72B0")
            ax.barh([0], [1 - value], left=[value], color="#C7C7C7")
            ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_title(title, fontsize=9, loc="left")
            ax.grid(False)
        axes[1].set_xlabel("share of total imaging variance")
        figs.save("imaging_variance_decomposition", fig)


def _spectrum_figure(fit, truth, nc, figs):
    fig, ax1 = figs.subplots(figsize=(8, 4))
    comps = np.arange(1, fit.n_components + 1)
    ax1.bar(comps, fit.recoverability, color="#4C72B0",
            label="in-sample recoverability $R_i=\\rho_i^2$")
    if truth is not None:
        _, R_true = rgit.true_recoverability(truth)
        ax1.plot(np.arange(1, min(len(R_true), nc) + 1), R_true[:nc], "o--",
                 color="#C44E52", label="ground-truth $R_i$")
    ax1.set_xlabel("canonical component $i$"); ax1.set_ylabel("recoverability $R_i$")
    ax1.set_ylim(0, 1.02)
    ax2 = ax1.twinx()
    ax2.plot(comps, np.cumsum(-0.5 * np.log1p(-fit.rho ** 2)) / np.log(2), "s-",
             color="#55A868", label="cumulative MI (bits)")
    ax2.set_ylabel("cumulative mutual information (bits)"); ax2.grid(False)
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labs = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labs, loc="upper right", frameon=True)
    ax1.set_title("Recoverability spectrum")
    figs.save("recoverability_spectrum", fig)


def _mi_per_direction_figure(fit, figs):
    mi_per = -0.5 * np.log1p(-fit.rho ** 2)
    fig, ax = figs.subplots(figsize=(7, 3.5))
    ax.bar(np.arange(1, fit.n_components + 1), mi_per / np.log(2), color="#8172B2")
    ax.set_xlabel("canonical component $i$"); ax.set_ylabel("MI contribution (bits)")
    ax.set_title("Per-direction information contribution")
    figs.save("mi_per_direction", fig)


def _heldout_recovery(cfg, pre, stats, figs):
    from sklearn.model_selection import train_test_split

    idx = np.arange(pre.G_model.shape[0])
    tr, te = train_test_split(idx, test_size=0.3, random_state=cfg.seed)
    fit_tr = rgit.fit_recoverability(pre.G_model[tr], pre.X_model[tr], n_components=pre.n_components)
    B, Sgcx = rgit.posterior(fit_tr)
    G_te_c = pre.G_model[te] - fit_tr.g_mean
    X_te_c = pre.X_model[te] - fit_tr.x_mean
    G_te_hat = X_te_c @ B.T
    a1 = fit_tr.genomic_dirs[:, 0]
    score_true = G_te_c @ a1
    score_pred = G_te_hat @ a1
    ss_res = np.sum((score_true - score_pred) ** 2)
    ss_tot = np.sum((score_true - score_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    var_ratio = np.clip(np.diag(Sgcx) / np.diag(fit_tr.Sg), 0, 1)
    stats.update({
        "heldout_R2_leading_direction": float(r2),
        "median_posterior_var_retained": float(np.median(var_ratio)),
    })
    if figs.enabled:
        fig, axes = figs.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(score_true, score_pred, s=14, alpha=0.6, color="#4C72B0")
        lim = [score_true.min(), score_true.max()]
        axes[0].plot(lim, lim, "k--", lw=1)
        axes[0].set_xlabel("true leading genomic score")
        axes[0].set_ylabel("predicted from imaging")
        axes[0].set_title(f"Held-out recovery ($R^2$ = {r2:.2f})")
        axes[1].hist(var_ratio, bins=30, color="#937860")
        axes[1].axvline(np.median(var_ratio), color="k", ls="--", lw=1)
        axes[1].set_xlabel("posterior / prior variance per genomic feature")
        axes[1].set_ylabel("count")
        axes[1].set_title("Uncertainty retained after imaging")
        figs.save("heldout_recovery", fig)


def _gene_loadings(pre, fit, stats, figs):
    gene_dirs = pre.to_gene_loadings(fit.genomic_dirs)
    n_show = min(3, fit.n_components)
    top_k = 12
    rec = {}
    axes = None
    if figs.enabled:
        fig, axes = figs.subplots(1, n_show, figsize=(4.2 * n_show, 4), sharey=False)
        axes = np.atleast_1d(axes)
    for i in range(n_show):
        load = gene_dirs[:, i]
        order = np.argsort(-np.abs(load))[:top_k]
        order = order[np.argsort(load[order])]
        rec[f"a{i+1}"] = [
            {"gene": str(pre.gene_names[j]), "loading": float(load[j])} for j in order[::-1]
        ]
        if axes is not None:
            k = len(order)
            colors = ["#C44E52" if v < 0 else "#4C72B0" for v in load[order]]
            axes[i].barh(np.arange(k), load[order], color=colors)
            axes[i].set_yticks(np.arange(k))
            axes[i].set_yticklabels(pre.gene_names[order], fontsize=8)
            axes[i].set_title(f"direction $a_{{{i+1}}}$  ($R_{{{i+1}}}$ = {fit.recoverability[i]:.2f})")
            axes[i].set_xlabel("loading (standardized genes)")
    if axes is not None:
        fig.suptitle("Top genomic loadings of the most image-identifiable directions", y=1.03)
        figs.save("gene_loadings", fig)
    stats["top_gene_loadings"] = rec


def _per_gene_auc(cfg, pre, fit, stats, figs):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    _, Sgcx_full = rgit.posterior(fit)
    if pre.pca_g is not None:
        W = pre.pca_g.components_
        post_var_gene = np.einsum("ig,ij,jg->g", W, Sgcx_full, W)
        prior_var_gene = np.einsum("ig,ij,jg->g", W, fit.Sg, W)
    else:
        post_var_gene, prior_var_gene = np.diag(Sgcx_full), np.diag(fit.Sg)
    gene_recov = np.clip(1.0 - post_var_gene / np.maximum(prior_var_gene, 1e-12), 0, 1)

    if pre.G_bin_global is not None:
        G_bin = pre.G_bin_global.astype(int)
    else:
        G_bin = (pre.G_std > 0).astype(int)
    mut_freq = G_bin.mean(axis=0)
    MUT_LO, MUT_HI = 0.05, 0.95
    eligible = np.where((mut_freq >= MUT_LO) & (mut_freq <= MUT_HI))[0]
    show_ceiling = (cfg.genomics_data_type == "variant")

    if len(eligible) < 4:
        logger.info("only %d eligible genes -- skipping per-gene validation", len(eligible))
        return

    order = eligible[np.argsort(gene_recov[eligible])]
    n_pick = min(5, len(eligible) // 2)
    low_genes = order[:n_pick]
    high_genes = order[-n_pick:][::-1]

    def _cv_auc(y, x, n_folds=5):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.seed)
        a_lr, a_rf = [], []
        for tr, te in skf.split(x, y):
            if len(np.unique(y[te])) < 2:
                continue
            sc = StandardScaler().fit(x[tr])
            lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
            lr.fit(sc.transform(x[tr]), y[tr])
            a_lr.append(roc_auc_score(y[te], lr.predict_proba(sc.transform(x[te]))[:, 1]))
            rf = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=3,
                                        random_state=cfg.seed, class_weight="balanced", n_jobs=-1)
            rf.fit(x[tr], y[tr])
            a_rf.append(roc_auc_score(y[te], rf.predict_proba(x[te])[:, 1]))
        return (np.mean(a_lr) if a_lr else np.nan, np.mean(a_rf) if a_rf else np.nan)

    def _auc_ceiling(R):
        R = np.clip(R, 0.0, 1.0 - 1e-9)
        return float(_st.norm.cdf(np.sqrt(R / (1.0 - R)) / np.sqrt(2.0)))

    rows = []
    for j in np.concatenate([high_genes, low_genes]):
        auc_lr, auc_rf = _cv_auc(G_bin[:, j], pre.X_model)
        row = {
            "gene": pre.gene_names[j], "mut_freq": mut_freq[j], "R_linear": gene_recov[j],
            "group": "identifiable" if j in high_genes else "non-identifiable",
            "AUC_LR": auc_lr, "AUC_RF": auc_rf,
        }
        if show_ceiling:
            row["AUC_ceiling"] = _auc_ceiling(gene_recov[j])
        rows.append(row)
    gene_table = pd.DataFrame(rows)
    grp_cols = ["AUC_LR", "AUC_RF"] + (["AUC_ceiling"] if show_ceiling else [])
    grp = gene_table.groupby("group")[grp_cols].mean().round(3)
    stats["per_gene_auc"] = gene_table.assign(gene=lambda d: d.gene.astype(str)).to_dict(orient="records")
    stats["per_gene_auc_mean_by_group"] = grp.reset_index().to_dict(orient="records")

    if figs.enabled:
        fig, ax = figs.subplots(figsize=(9, 4))
        xp = np.arange(len(gene_table))
        if show_ceiling:
            w = 0.30
            ax.bar(xp - w, gene_table["AUC_LR"], w, label="logistic $L_2$", color="#4C72B0")
            ax.bar(xp, gene_table["AUC_RF"], w, label="random forest", color="#55A868")
            ax.bar(xp + w, gene_table["AUC_ceiling"], w,
                   label=r"Bayes ceiling $\Phi(\sqrt{R/(1-R)}/\sqrt{2})$", color="#C44E52", alpha=0.7)
        else:
            w = 0.38
            ax.bar(xp - w / 2, gene_table["AUC_LR"], w, label="logistic $L_2$", color="#4C72B0")
            ax.bar(xp + w / 2, gene_table["AUC_RF"], w, label="random forest", color="#55A868")
        ax.axhline(0.5, color="k", ls="--", lw=1, label="chance")
        ax.set_xticks(xp); ax.set_xticklabels(gene_table["gene"], rotation=45, ha="right")
        ax.set_ylabel("5-fold CV AUC"); ax.set_ylim(0, 1)
        nh = len(high_genes)
        ax.axvspan(-0.5, nh - 0.5, alpha=0.08, color="#4C72B0")
        ax.axvspan(nh - 0.5, len(gene_table) - 0.5, alpha=0.08, color="#C44E52")
        ax.set_title("Per-gene mutation predictability from imaging")
        ax.legend(loc="lower right", fontsize=8)
        figs.save("per_gene_auc", fig)


def _cv_figure(fit, cv_mean, cv_std, truth, nc, cfg, figs):
    fig, ax = figs.subplots(figsize=(8, 4))
    comps = np.arange(1, fit.n_components + 1)
    ax.bar(comps, fit.recoverability, color="#C7C7C7", label="in-sample (biased)")
    ax.errorbar(comps, cv_mean, yerr=cv_std, fmt="o-", color="#4C72B0", capsize=3,
                label=f"cross-validated ({cfg.n_folds}-fold)")
    if truth is not None:
        _, R_true = rgit.true_recoverability(truth)
        ax.plot(np.arange(1, min(len(R_true), nc) + 1), R_true[:nc], "x--",
                color="#C44E52", label="ground truth")
    ax.set_xlabel("canonical component $i$"); ax.set_ylabel("recoverability $R_i$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("In-sample vs. honest (cross-validated) recoverability")
    ax.legend()
    figs.save("cv_recoverability", fig)


def _permutation_figure(observed, null, cfg, figs):
    fig, ax = figs.subplots(figsize=(7, 3.8))
    ax.hist(null[:, 0], bins=30, color="#C7C7C7", label="null (permuted labels)")
    ax.axvline(observed[0], color="#C44E52", lw=2, label=f"observed $\\rho_1$ = {observed[0]:.3f}")
    ax.axvline(np.quantile(null[:, 0], 0.95), color="k", ls="--", lw=1,
               label=f"null 95% = {np.quantile(null[:, 0], 0.95):.3f}")
    ax.set_xlabel("leading canonical correlation $\\rho_1$"); ax.set_ylabel("count")
    ax.set_title(f"Permutation null  (n = {cfg.n_perm})")
    ax.legend()
    figs.save("permutation_null", fig)


def _mi_upper_bound(cfg, pre, observed, null, stats, figs):
    K = null.shape[1]
    mi_null_K = -0.5 * np.sum(np.log1p(-np.clip(null, 0, 1 - 1e-12) ** 2), axis=1)
    mi_obs_K = float(-0.5 * np.sum(np.log1p(-np.clip(observed, 0, 1 - 1e-12) ** 2)))
    mi_ucl_nats = float(np.quantile(mi_null_K, 0.95))
    mi_ucl_bits = mi_ucl_nats / np.log(2)

    cv_obs, cv_null, cv_pvals = rgit.cv_permutation_test(
        pre.G_model, pre.X_model, n_components=K, n_perm=cfg.n_perm,
        n_folds=cfg.n_folds, random_state=cfg.seed)
    cv_null_q95 = np.quantile(cv_null, 0.95, axis=0)
    eff_rank = int(np.sum((cv_obs > cv_null_q95) & (cv_pvals < 0.05)))
    null_R_q95 = np.quantile(null ** 2, 0.95, axis=0)

    stats["mi_upper_bound"] = {
        "K": int(K), "observed_nats": float(mi_obs_K), "observed_bits": float(mi_obs_K / np.log(2)),
        "ucl_95_nats": float(mi_ucl_nats), "ucl_95_bits": float(mi_ucl_bits),
        "null_mean_nats": float(mi_null_K.mean()),
    }
    stats["effective_identifiable_rank"] = {
        "K_tested": int(K), "rank": eff_rank, "method": "cv_permutation",
        "cv_R_observed": [float(x) for x in cv_obs],
        "cv_R_null_q95": [float(x) for x in cv_null_q95],
        "cv_R_perm_pvalues": [float(x) for x in cv_pvals],
        "legacy_inflated_null_R_q95": [float(x) for x in null_R_q95],
    }
    if figs.enabled:
        fig, ax = figs.subplots(figsize=(7.5, 4))
        ax.hist(mi_null_K / np.log(2), bins=30, color="#C7C7C7", label="null cumulative MI (bits)")
        ax.axvline(mi_obs_K / np.log(2), color="#C44E52", lw=2,
                   label=f"observed = {mi_obs_K/np.log(2):.2f} bits")
        ax.axvline(mi_ucl_bits, color="k", ls="--", lw=1.5,
                   label=f"95% UCL on $I(G;X)$ = {mi_ucl_bits:.2f} bits")
        ax.set_xlabel(f"cumulative MI over top {K} directions (bits)")
        ax.set_ylabel("permutation count")
        ax.set_title("Upper-bound on $I(G;X)$ from the permutation null")
        ax.legend(loc="best", fontsize=8)
        figs.save("mi_upper_bound", fig)


def _deconfound_comparison(cfg, pre, stats, figs):
    def _honest_spectrum(Gm, Xm):
        ncc = max(1, min(cfg.n_components_max, Gm.shape[1], Xm.shape[1]))
        cv = rgit.cross_validated_recoverability(
            Gm, Xm, n_components=ncc, n_folds=cfg.n_folds, random_state=cfg.seed).mean(0)
        obs, nullcv, pcv = rgit.cv_permutation_test(
            Gm, Xm, n_components=3, n_perm=cfg.n_perm, n_folds=cfg.n_folds, random_state=cfg.seed)
        rank = int(np.sum((obs > np.quantile(nullcv, 0.95, 0)) & (pcv < 0.05)))
        return cv, rank, [float(x) for x in pcv]

    Gd, Xd = residualize(pre.G_model_raw, pre.C_full), residualize(pre.X_model_raw, pre.C_full)
    cv_raw, rank_raw, p_raw = _honest_spectrum(pre.G_model_raw, pre.X_model_raw)
    cv_dec, rank_dec, p_dec = _honest_spectrum(Gd, Xd)
    stats["deconfound_comparison"] = {
        "confounders": pre.conf_cols,
        "raw": {"cv_R": [float(x) for x in cv_raw[:5]], "eff_rank": rank_raw, "cv_perm_p": p_raw},
        "deconfounded": {"cv_R": [float(x) for x in cv_dec[:5]], "eff_rank": rank_dec, "cv_perm_p": p_dec},
        "retained_frac_R1": float(cv_dec[0] / cv_raw[0]) if cv_raw[0] > 0 else None,
    }
    if figs.enabled:
        K = 5
        fig, ax = figs.subplots(figsize=(7, 4)); x = np.arange(1, K + 1); w = 0.38
        ax.bar(x - w / 2, cv_raw[:K], w, color="#4C72B0", label=f"raw (eff. rank {rank_raw})")
        ax.bar(x + w / 2, cv_dec[:K], w, color="#DD8452",
               label=f"deconfounded on {', '.join(pre.conf_cols)} (eff. rank {rank_dec})")
        ax.set_xlabel("canonical component $i$"); ax.set_ylabel(r"cross-validated $R_i$")
        ax.set_xticks(x); ax.set_title("Recoverability with vs. without demographic deconfounding")
        ax.legend(fontsize=8)
        figs.save("deconfound_comparison", fig)


def _sample_complexity_sweep(cfg, pre, stats, figs):
    from sklearn.decomposition import PCA

    def _wachter_edge(p_star, d_star, n):
        kg, kx = p_star / n, d_star / n
        if kg + kx >= 1:
            return 1.0
        return float((np.sqrt(kg * (1 - kx)) + np.sqrt(kx * (1 - kg))) ** 2)

    G_full_std, X_full_std = pre.G_std, pre.X_std
    n_full = G_full_std.shape[0]
    grid = sorted({int(round(f * n_full)) for f in (0.3, 0.5, 0.7, 0.85, 1.0)}
                  | ({50} if n_full >= 60 else {n_full}))
    grid = [n for n in grid if 30 <= n <= n_full]
    if not grid:
        return
    sweep_rng = np.random.default_rng(cfg.seed + 1)
    N_PERM_SWEEP = 100
    rows = []
    for n_sub in grid:
        sel = sweep_rng.choice(n_full, size=n_sub, replace=False)
        Gs, Xs = G_full_std[sel], X_full_std[sel]
        budget = max(cfg.min_working_dim, n_sub // cfg.samples_per_dim)
        p_star = min(Gs.shape[1], n_sub - 1, budget)
        d_star = min(Xs.shape[1], n_sub - 1, budget)
        Gm = PCA(n_components=p_star, random_state=cfg.seed).fit_transform(Gs)
        Xm = PCA(n_components=d_star, random_state=cfg.seed).fit_transform(Xs)
        ncomp = max(1, min(cfg.n_components_max, p_star, d_star, n_sub - 1))
        fit_s = rgit.fit_recoverability(Gm, Xm, n_components=ncomp)
        cv_s = rgit.cross_validated_recoverability(
            Gm, Xm, n_components=ncomp, n_folds=min(cfg.n_folds, n_sub // 10) or 2,
            random_state=cfg.seed).mean(axis=0)
        _, null_s, _ = rgit.permutation_test(
            Gm, Xm, n_components=1, n_perm=N_PERM_SWEEP, random_state=cfg.seed)
        rows.append({
            "n": int(n_sub), "p_star": int(p_star), "d_star": int(d_star),
            "rho1_insample": float(fit_s.rho[0]), "R1_cv": float(cv_s[0]),
            "MI_nats": float(fit_s.mutual_information()),
            "null_rho1_q95": float(np.quantile(null_s[:, 0], 0.95)),
            "wachter_edge": _wachter_edge(p_star, d_star, n_sub),
        })
    sweep = pd.DataFrame(rows)
    stats["sample_complexity_sweep"] = sweep.to_dict(orient="records")
    if figs.enabled:
        fig, ax = figs.subplots(figsize=(7.5, 4))
        ax.plot(sweep["n"], sweep["rho1_insample"] ** 2, "o-", color="#C7C7C7",
                label=r"in-sample $\rho_1^2$ (biased)")
        ax.plot(sweep["n"], sweep["R1_cv"], "s-", color="#4C72B0",
                label=r"cross-validated $R_1$ (honest)")
        ax.plot(sweep["n"], sweep["null_rho1_q95"] ** 2, "^--", color="#C44E52",
                label=r"permutation null 95% $\rho_1^2$")
        ax.plot(sweep["n"], sweep["wachter_edge"], "x:", color="#55A868",
                label=r"Wachter detection edge")
        ax.set_xlabel("subsampled cohort size $n$")
        ax.set_ylabel(r"leading $R_1$ / detection threshold")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("Sample-complexity sweep -- how recoverability scales with $n$")
        ax.legend(loc="best", fontsize=8)
        figs.save("sample_complexity_sweep", fig)


def _downsample_sweep(cfg, pre, stats, figs):
    from sklearn.decomposition import PCA

    DOWN_DIM = int(min(pre.G_model_raw.shape[1], pre.X_model_raw.shape[1], 30))
    down_rng = np.random.default_rng(cfg.seed + 7)
    n_full_d = pre.G_model_raw.shape[0]
    sizes = sorted({max(40, int(round(f * n_full_d))) for f in cfg.downsample_fracs})
    sizes = [n for n in sizes if n <= n_full_d]
    if not sizes:
        return

    def _fixed_cvr1(Gm, Xm, n_sub, sel):
        Gs = PCA(DOWN_DIM, random_state=cfg.seed).fit_transform(Gm[sel])
        Xs = PCA(DOWN_DIM, random_state=cfg.seed).fit_transform(Xm[sel])
        cv = rgit.cross_validated_recoverability(
            Gs, Xs, n_components=3, n_folds=min(cfg.n_folds, n_sub // 10) or 2,
            random_state=cfg.seed).mean(0)
        _, nulls, _ = rgit.permutation_test(Gs, Xs, n_components=1, n_perm=100, random_state=cfg.seed)
        return float(cv[0]), float(np.quantile(nulls[:, 0], 0.95) ** 2)

    have_dec = pre.C_full is not None
    Gdec = residualize(pre.G_model_raw, pre.C_full) if have_dec else None
    Xdec = residualize(pre.X_model_raw, pre.C_full) if have_dec else None
    rows = []
    for n_sub in sizes:
        sel = down_rng.choice(n_full_d, size=n_sub, replace=False)
        r_raw, thr = _fixed_cvr1(pre.G_model_raw, pre.X_model_raw, n_sub, sel)
        row = {"n": n_sub, "frac": round(n_sub / n_full_d, 3), "R1_cv_raw": r_raw, "null95": thr}
        if have_dec:
            row["R1_cv_deconfounded"], _ = _fixed_cvr1(Gdec, Xdec, n_sub, sel)
        rows.append(row)
    down = pd.DataFrame(rows)
    full_r1 = down["R1_cv_raw"].iloc[-1]
    reach = down[down["R1_cv_raw"] >= 0.9 * full_r1]
    n_needed = int(reach["n"].iloc[0]) if len(reach) else None
    stats["downsample_sweep"] = {"working_dim": DOWN_DIM, "rows": down.to_dict(orient="records"),
                                 "n_for_90pct_full_R1": n_needed}
    if figs.enabled:
        fig, ax = figs.subplots(figsize=(7.5, 4))
        ax.plot(down["n"], down["R1_cv_raw"], "s-", color="#4C72B0", label=r"CV $R_1$ (raw)")
        if have_dec:
            ax.plot(down["n"], down["R1_cv_deconfounded"], "o-", color="#DD8452",
                    label=r"CV $R_1$ (deconfounded)")
        ax.plot(down["n"], down["null95"], "^--", color="#C44E52", label=r"perm. null 95% $\rho_1^2$")
        if n_needed is not None:
            ax.axvline(n_needed, ls=":", color="gray")
        ax.set_xlabel(r"subsample size $n$  (fixed working dim $p^\star=d^\star=%d$)" % DOWN_DIM)
        ax.set_ylabel(r"leading cross-validated $R_1$")
        ax.set_title("How much data is needed? CV recoverability vs. subsample size")
        ax.legend(fontsize=8)
        figs.save("downsample_recoverability", fig)


def _synthetic_validation(pre, fit, cv_mean, truth, stats, figs):
    rho_true, R_true = rgit.true_recoverability(truth)
    k = truth["k"]
    est_dirs = pre.to_gene_loadings(fit.genomic_dirs[:, :k])
    cos_ang = rgit.subspace_alignment(est_dirs, truth["Wg"])
    stats["synthetic_validation"] = {
        "true_k": int(k),
        "true_R": [float(x) for x in R_true[:k + 2]],
        "cv_R": [float(x) for x in cv_mean[:k + 2]],
        "subspace_alignment_cosines": [float(x) for x in cos_ang],
        "mean_alignment": float(cos_ang.mean()),
    }
    if figs.enabled:
        fig, axes = figs.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(np.arange(1, k + 3), R_true[:k + 2], "o--", color="#C44E52", label="ground truth")
        axes[0].plot(np.arange(1, len(cv_mean[:k + 2]) + 1), cv_mean[:k + 2], "s-",
                     color="#4C72B0", label="cross-validated estimate")
        axes[0].set_xlabel("component"); axes[0].set_ylabel("recoverability $R_i$")
        axes[0].set_title("Estimated vs. true recoverability"); axes[0].legend()
        axes[1].bar(np.arange(1, k + 1), cos_ang, color="#55A868")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_xlabel("principal direction"); axes[1].set_ylabel("cosine of principal angle")
        axes[1].set_title("Estimated vs. true genomic subspace")
        figs.save("synthetic_truth_validation", fig)
