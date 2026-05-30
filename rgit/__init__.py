"""rgit -- information-theoretic bounds on radiogenomic recoverability.

The core estimator is :func:`rgit.fit_recoverability`, which fits the
linear--Gaussian model of ``main.tex`` (regularized CCA between a patient x
genomics matrix and a patient x imaging matrix) and exposes the recoverability
spectrum, image-identifiable genomic directions, the Bayes-optimal posterior,
and the mutual information between modalities.

For the full, reproducible analysis pipeline -- the headless equivalent of
``notebooks/radiogenomic_recoverability.ipynb`` -- use
:func:`rgit.run_recoverability_analysis` (or the ``rgit-recoverability`` console
script). With no data on disk it runs a synthetic ground-truth dataset, so the
package is exercisable straight after ``pip install``.
"""

import logging
import sys

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

from rgit.config import RecoverabilityConfig
from rgit.datasets import (
    Preprocessed,
    load_modalities,
    make_synthetic_modalities,
    preprocess,
)
from rgit.report import (
    RecoverabilityReport,
    mvn_diagnostic,
    run_recoverability_analysis,
)

# RadImageNet feature extraction depends on torch/torchvision, which are
# optional (the `processing` extra). Import lazily so the core analysis package
# is usable without them; the names resolve to None when torch is absent.
try:
    from rgit.radimagenet import (
        RadImageNetEmbedding,
        convert_image_to_radimagenet_format,
        get_radimagenet_embeddings,
    )

    _HAVE_RADIMAGENET = True
except ImportError:
    RadImageNetEmbedding = None
    convert_image_to_radimagenet_format = None
    get_radimagenet_embeddings = None
    _HAVE_RADIMAGENET = False

__version__ = "0.2.0"

__all__ = [
    # core estimators
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
    # reproducible analysis pipeline
    "RecoverabilityConfig",
    "RecoverabilityReport",
    "run_recoverability_analysis",
    "load_modalities",
    "make_synthetic_modalities",
    "preprocess",
    "Preprocessed",
    "mvn_diagnostic",
    # RadImageNet feature extraction (requires the `processing` extra)
    "RadImageNetEmbedding",
    "convert_image_to_radimagenet_format",
    "get_radimagenet_embeddings",
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