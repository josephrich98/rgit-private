"""rgit -- information-theoretic bounds on radiogenomic recoverability.

The core estimator is :func:`rgit.fit_recoverability`, which fits the
linear--Gaussian model of ``main.tex`` (regularized CCA between a patient x
genomics matrix and a patient x imaging matrix) and exposes the recoverability
spectrum, image-identifiable genomic directions, the Bayes-optimal posterior,
and the mutual information between modalities.
"""

import logging
import sys

from rgit.radimagenet import (
    RadImageNetEmbedding,
    convert_image_to_radimagenet_format,
    get_radimagenet_embeddings,
)

from rgit.model import (
    RecoverabilityFit,
    cross_validated_recoverability,
    cv_permutation_test,
    direction_recoverability,
    distance_correlation,
    distance_correlation_test,
    fit_recoverability,
    gaussian_rank_transform,
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
    "RadImageNetEmbedding",
    "convert_image_to_radimagenet_format",
    "get_radimagenet_embeddings",
    "RecoverabilityFit",
    "cross_validated_recoverability",
    "cv_permutation_test",
    "direction_recoverability",
    "distance_correlation",
    "distance_correlation_test",
    "fit_recoverability",
    "gaussian_rank_transform",
    "imaging_variance_explained",
    "make_synthetic_radiogenomics",
    "mutual_information",
    "permutation_test",
    "posterior",
    "subspace_alignment",
    "to_dense",
    "true_recoverability",
]

logger = logging.getLogger("rgit")
# check if logger has been initialized
if not logger.hasHandlers() or len(logger.handlers) == 0:
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)