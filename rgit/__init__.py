"""rgit -- information-theoretic bounds on radiogenomic recoverability.

The core estimator is :func:`rgit.fit_recoverability`, which fits the
linear--Gaussian model of ``main.tex`` (regularized CCA between a patient x
genomics matrix and a patient x imaging matrix) and exposes the recoverability
spectrum, image-identifiable genomic directions, the Bayes-optimal posterior,
and the mutual information between modalities.
"""

from rgit.model import (
    RecoverabilityFit,
    cross_validated_recoverability,
    direction_recoverability,
    fit_recoverability,
    imaging_variance_explained,
    make_synthetic_radiogenomics,
    mutual_information,
    permutation_test,
    posterior,
    subspace_alignment,
    to_dense,
    true_recoverability,
)

__version__ = "0.1.0"

__all__ = [
    "RecoverabilityFit",
    "cross_validated_recoverability",
    "direction_recoverability",
    "fit_recoverability",
    "imaging_variance_explained",
    "make_synthetic_radiogenomics",
    "mutual_information",
    "permutation_test",
    "posterior",
    "subspace_alignment",
    "to_dense",
    "true_recoverability",
]
