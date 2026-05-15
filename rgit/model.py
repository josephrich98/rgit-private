"""Linear--Gaussian radiogenomic recoverability model.

This module implements the estimators behind ``main.tex``: given a patient x
genomics matrix ``G`` and a patient x imaging matrix ``X``, it estimates the
recoverability spectrum (squared canonical correlations), the image-identifiable
genomic directions, the Bayes-optimal posterior ``E[g | x]`` and posterior
covariance ``Sigma_{G|X}``, and the mutual information ``I(G; X)``.

Notation mirrors the manuscript:

* ``Sg, Sx, Sgx``  -- genomic / imaging / cross covariance
* ``Sgcx``         -- posterior covariance of G given X (the recoverability floor)
* ``rho``, ``R``   -- canonical correlations and recoverability ``R = rho ** 2``
* ``genomic_dirs`` -- columns ``a_i``: image-identifiable genomic directions
* ``imaging_dirs`` -- columns ``b_i``: matched imaging directions

Everything reduces to regularized canonical correlation analysis between the two
modalities (Proposition 5 in the manuscript), so the transfer map ``A`` and the
noise level ``sigma^2`` never have to be estimated individually.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "RecoverabilityFit",
    "to_dense",
    "fit_recoverability",
    "posterior",
    "mutual_information",
    "direction_recoverability",
    "imaging_variance_explained",
    "cross_validated_recoverability",
    "permutation_test",
    "subspace_alignment",
    "make_synthetic_radiogenomics",
    "true_recoverability",
]


# --------------------------------------------------------------------------
# small linear-algebra / data helpers
# --------------------------------------------------------------------------
def to_dense(matrix) -> np.ndarray:
    """Return a dense float64 ``ndarray`` from an array, AnnData ``.X``, or sparse."""
    if hasattr(matrix, "toarray"):  # scipy sparse
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float64)


def _inv_sqrt(S: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Symmetric inverse square root of a symmetric positive (semi-)definite matrix."""
    w, V = np.linalg.eigh((S + S.T) / 2.0)
    w = np.clip(w, eps * np.max(w), None)
    return (V / np.sqrt(w)) @ V.T


def _covariance(M: np.ndarray, reg) -> np.ndarray:
    """Regularized covariance estimate.

    ``reg='lw'`` uses Ledoit--Wolf shrinkage; a float adds ``reg * mean_var * I``
    (ridge) to the sample covariance; ``0`` / ``None`` returns the plain estimate.
    """
    n, q = M.shape
    if reg == "lw":
        from sklearn.covariance import LedoitWolf

        return LedoitWolf(assume_centered=True).fit(M).covariance_
    S = (M.T @ M) / max(n - 1, 1)
    if reg:
        S = S + float(reg) * (np.trace(S) / q) * np.eye(q)
    return S


# --------------------------------------------------------------------------
# the fitted model
# --------------------------------------------------------------------------
@dataclass
class RecoverabilityFit:
    """Result of :func:`fit_recoverability`.

    Attributes
    ----------
    rho : (r,) array
        Canonical correlations, descending. ``r = n_components``.
    recoverability : (r,) array
        ``rho ** 2`` -- the per-direction recoverability ``R_i`` in ``[0, 1]``.
    genomic_dirs : (p, r) array
        Columns ``a_i``: image-identifiable genomic directions, ``Sg``-orthonormal.
    imaging_dirs : (d, r) array
        Columns ``b_i``: matched imaging directions, ``Sx``-orthonormal.
    Sg, Sx, Sgx : arrays
        Regularized covariances used for the fit.
    g_mean, x_mean : arrays
        Feature means subtracted before fitting (for transforming new data).
    """

    rho: np.ndarray
    recoverability: np.ndarray
    genomic_dirs: np.ndarray
    imaging_dirs: np.ndarray
    Sg: np.ndarray
    Sx: np.ndarray
    Sgx: np.ndarray
    g_mean: np.ndarray
    x_mean: np.ndarray

    @property
    def n_components(self) -> int:
        return len(self.rho)

    def mutual_information(self) -> float:
        """Mutual information ``I(G; X)`` in nats, from the canonical spectrum."""
        return mutual_information(self.rho)

    def genomic_scores(self, G) -> np.ndarray:
        """Project genomics onto the canonical genomic directions ``a_i``."""
        return (to_dense(G) - self.g_mean) @ self.genomic_dirs

    def imaging_scores(self, X) -> np.ndarray:
        """Project imaging onto the canonical imaging directions ``b_i``."""
        return (to_dense(X) - self.x_mean) @ self.imaging_dirs

    def posterior(self):
        """Bayes-optimal map ``B`` (``E[g|x] = B x``) and posterior covariance ``Sgcx``."""
        return posterior(self)


def fit_recoverability(
    G,
    X,
    reg_g="lw",
    reg_x="lw",
    n_components: int | None = None,
) -> RecoverabilityFit:
    """Fit the linear--Gaussian recoverability model via regularized CCA.

    Parameters
    ----------
    G : (n, p) array-like
        Patient x genomics matrix.
    X : (n, d) array-like
        Patient x imaging matrix. Rows must be patient-aligned with ``G``.
    reg_g, reg_x : 'lw' or float
        Covariance regularization for each modality (see :func:`_covariance`).
    n_components : int, optional
        Number of canonical directions to keep. Defaults to ``min(p, d)``.

    Returns
    -------
    RecoverabilityFit
    """
    G = to_dense(G)
    X = to_dense(X)
    if G.shape[0] != X.shape[0]:
        raise ValueError(
            f"G and X must have the same number of patients, got {G.shape[0]} and {X.shape[0]}"
        )

    g_mean = G.mean(axis=0)
    x_mean = X.mean(axis=0)
    Gc = G - g_mean
    Xc = X - x_mean
    n = G.shape[0]

    Sg = _covariance(Gc, reg_g)
    Sx = _covariance(Xc, reg_x)
    Sgx = (Gc.T @ Xc) / max(n - 1, 1)

    # whitened transfer operator  M = Sg^{-1/2} Sgx Sx^{-1/2}   (manuscript eq. 9)
    Sg_isqrt = _inv_sqrt(Sg)
    Sx_isqrt = _inv_sqrt(Sx)
    M = Sg_isqrt @ Sgx @ Sx_isqrt
    U, s, Vt = np.linalg.svd(M, full_matrices=False)

    rho = np.clip(s, 0.0, 1.0 - 1e-12)  # canonical correlations
    r = len(rho) if n_components is None else min(int(n_components), len(rho))

    genomic_dirs = Sg_isqrt @ U[:, :r]
    imaging_dirs = Sx_isqrt @ Vt[:r].T

    return RecoverabilityFit(
        rho=rho[:r],
        recoverability=rho[:r] ** 2,
        genomic_dirs=genomic_dirs,
        imaging_dirs=imaging_dirs,
        Sg=Sg,
        Sx=Sx,
        Sgx=Sgx,
        g_mean=g_mean,
        x_mean=x_mean,
    )


# --------------------------------------------------------------------------
# closed-form quantities from the manuscript
# --------------------------------------------------------------------------
def posterior(fit: RecoverabilityFit):
    """Bayes-optimal predictor and posterior covariance.

    Returns
    -------
    B : (p, d) array
        ``E[g | x] = B @ x``  with ``B = Sgx Sx^{-1}``  (manuscript eq. 4).
    Sgcx : (p, p) array
        Posterior covariance ``Sigma_{G|X} = Sg - Sgx Sx^{-1} Sxg`` -- the hard
        floor on residual genomic uncertainty (manuscript eq. 5).
    """
    Sx_inv = np.linalg.inv(fit.Sx)
    B = fit.Sgx @ Sx_inv
    Sgcx = fit.Sg - fit.Sgx @ Sx_inv @ fit.Sgx.T
    Sgcx = (Sgcx + Sgcx.T) / 2.0
    return B, Sgcx


def mutual_information(rho) -> float:
    """Gaussian mutual information ``I(G; X) = -0.5 * sum log(1 - rho_i^2)`` (nats)."""
    rho = np.clip(np.asarray(rho, dtype=np.float64), 0.0, 1.0 - 1e-12)
    return float(-0.5 * np.sum(np.log1p(-(rho ** 2))))


def direction_recoverability(v, Sg, Sgcx) -> float:
    """Recoverability ``R(v)`` of a named genomic direction (manuscript eq. 14).

    ``R(v) = 1 - (v' Sgcx v) / (v' Sg v)`` -- the population R^2 of the
    Bayes-optimal predictor of the score ``v' g`` from imaging.
    """
    v = np.asarray(v, dtype=np.float64)
    num = float(v @ Sgcx @ v)
    den = float(v @ Sg @ v)
    if den <= 0:
        raise ValueError("direction has non-positive prior variance under Sg")
    return 1.0 - num / den


def imaging_variance_explained(fit: RecoverabilityFit):
    """Decompose imaging covariance into genomics-driven and irreducible parts.

    Returns ``(fraction, signal_cov)`` where ``signal_cov = Sxg Sg^{-1} Sgx`` is
    the imaging covariance attributable to genomics and ``fraction`` is its share
    of total imaging variance, ``tr(signal_cov) / tr(Sx)``.
    """
    Sg_inv = np.linalg.inv(fit.Sg)
    signal_cov = fit.Sgx.T @ Sg_inv @ fit.Sgx
    fraction = float(np.trace(signal_cov) / np.trace(fit.Sx))
    return fraction, signal_cov


# --------------------------------------------------------------------------
# honest inference: cross-validation and permutation
# --------------------------------------------------------------------------
def cross_validated_recoverability(
    G,
    X,
    n_components: int,
    n_folds: int = 5,
    reg_g="lw",
    reg_x="lw",
    random_state: int = 0,
) -> np.ndarray:
    """Out-of-sample recoverability per canonical direction.

    Canonical directions are estimated on each training fold; the held-out
    squared correlation between the projected genomic and imaging scores is the
    honest recoverability. In-sample ``rho ** 2`` is biased upward; the gap to
    these values quantifies overfitting (manuscript sec. 10.4).

    Returns
    -------
    (n_folds, n_components) array of held-out recoverability values.
    """
    from sklearn.model_selection import KFold

    G = to_dense(G)
    X = to_dense(X)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    out = np.zeros((n_folds, n_components))

    for f, (tr, te) in enumerate(kf.split(G)):
        fit = fit_recoverability(G[tr], X[tr], reg_g, reg_x, n_components)
        sg = fit.genomic_scores(G[te])  # uses training-fold means + directions
        sx = fit.imaging_scores(X[te])
        for i in range(fit.n_components):
            c = np.corrcoef(sg[:, i], sx[:, i])[0, 1]
            out[f, i] = 0.0 if not np.isfinite(c) else c ** 2
    return out


def permutation_test(
    G,
    X,
    n_components: int = 1,
    n_perm: int = 200,
    reg_g="lw",
    reg_x="lw",
    random_state: int = 0,
):
    """Permutation test for image-identifiability of the leading genomic directions.

    Patient labels of ``G`` are repeatedly permuted to build a null distribution
    of the top canonical correlations.

    Returns
    -------
    observed : (n_components,) array of observed canonical correlations.
    null : (n_perm, n_components) array of permuted canonical correlations.
    pvalues : (n_components,) array of permutation p-values.
    """
    G = to_dense(G)
    X = to_dense(X)
    rng = np.random.default_rng(random_state)

    observed = fit_recoverability(G, X, reg_g, reg_x, n_components).rho.copy()
    null = np.zeros((n_perm, n_components))
    for b in range(n_perm):
        perm = rng.permutation(G.shape[0])
        null[b] = fit_recoverability(G[perm], X, reg_g, reg_x, n_components).rho

    pvalues = (1.0 + np.sum(null >= observed[None, :], axis=0)) / (1.0 + n_perm)
    return observed, null, pvalues


def subspace_alignment(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Cosines of the principal angles between ``span(U)`` and ``span(V)``.

    Values near 1 mean the two subspaces (e.g. estimated vs. true genomic
    directions) are well aligned.
    """
    Qu, _ = np.linalg.qr(np.asarray(U, dtype=np.float64))
    Qv, _ = np.linalg.qr(np.asarray(V, dtype=np.float64))
    s = np.linalg.svd(Qu.T @ Qv, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


# --------------------------------------------------------------------------
# synthetic data with known ground truth (probabilistic-CCA generative model)
# --------------------------------------------------------------------------
def make_synthetic_radiogenomics(
    n: int = 500,
    p: int = 120,
    d: int = 30,
    k: int = 5,
    shared_signal=None,
    noise_g: float = 1.0,
    noise_x: float = 1.0,
    random_state: int = 0,
):
    """Draw a synthetic radiogenomic dataset from the probabilistic-CCA model.

    A shared latent ``Z in R^k`` drives both modalities (manuscript eq. 19), so
    exactly ``k`` canonical correlations are nonzero and the ground-truth
    recoverability spectrum is computable in closed form via
    :func:`true_recoverability`.

    Parameters
    ----------
    n, p, d : int
        Number of patients, genomic features, imaging features.
    k : int
        Dimension of the shared (image-identifiable) genomic subspace.
    shared_signal : (k,) array-like, optional
        Per-component shared signal strength. Defaults to a decreasing ramp so
        the recoverability spectrum is non-trivial.
    noise_g, noise_x : float
        Scale of the modality-private noise.
    random_state : int

    Returns
    -------
    G : (n, p) array
    X : (n, d) array
    truth : dict
        Generative parameters plus the implied covariances ``Sg, Sx, Sgx``.
    """
    rng = np.random.default_rng(random_state)
    if shared_signal is None:
        shared_signal = np.linspace(4.0, 0.4, k)
    shared_signal = np.asarray(shared_signal, dtype=np.float64)

    Z = rng.standard_normal((n, k))
    Wg = rng.standard_normal((p, k))
    Wx = rng.standard_normal((d, k))
    # unit-norm loading columns, then inject the requested per-component signal
    Wg = Wg / np.linalg.norm(Wg, axis=0, keepdims=True) * np.sqrt(shared_signal)
    Wx = Wx / np.linalg.norm(Wx, axis=0, keepdims=True) * np.sqrt(shared_signal)

    Psi_g = noise_g * (0.5 + rng.random(p))  # anisotropic modality-private noise
    Psi_x = noise_x * (0.5 + rng.random(d))

    G = Z @ Wg.T + rng.standard_normal((n, p)) * np.sqrt(Psi_g)
    X = Z @ Wx.T + rng.standard_normal((n, d)) * np.sqrt(Psi_x)

    truth = {
        "k": k,
        "Z": Z,
        "Wg": Wg,
        "Wx": Wx,
        "Psi_g": Psi_g,
        "Psi_x": Psi_x,
        "shared_signal": shared_signal,
        "Sg": Wg @ Wg.T + np.diag(Psi_g),
        "Sx": Wx @ Wx.T + np.diag(Psi_x),
        "Sgx": Wg @ Wx.T,
    }
    return G, X, truth


def true_recoverability(truth: dict):
    """Closed-form ground-truth canonical correlations from synthetic parameters.

    Returns ``(rho, R)`` where ``R = rho ** 2``. Only the first ``truth['k']``
    entries are nonzero.
    """
    Sg, Sx, Sgx = truth["Sg"], truth["Sx"], truth["Sgx"]
    M = _inv_sqrt(Sg) @ Sgx @ _inv_sqrt(Sx)
    s = np.linalg.svd(M, compute_uv=False)
    rho = np.clip(s, 0.0, 1.0 - 1e-12)
    return rho, rho ** 2
