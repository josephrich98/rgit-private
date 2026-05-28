"""Configuration for the end-to-end recoverability analysis.

This is the package-level equivalent of the notebook's configuration cells.
Every knob the notebook exposes lives here as a validated field, so the full
analysis can be reproduced headlessly from a single object (or a JSON file)
without opening ``notebooks/radiogenomic_recoverability.ipynb``.

The model is intentionally *self-describing*: dumping it to JSON and writing it
alongside ``stats.json`` records exactly which preprocessing and analysis
settings produced a given set of numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

GenomicsDataType = Literal["variant", "expression"]
HVGMethod = Literal["auto", "scanpy", "variance"]


class RecoverabilityConfig(BaseModel):
    """All settings for :func:`rgit.run_recoverability_analysis`.

    Leave ``genomics_h5ad``/``imaging_h5ad`` as ``None`` (or point them at
    files that do not exist) to run on the synthetic probabilistic-CCA dataset
    with known ground truth -- this is what makes the package runnable straight
    after ``pip install`` with no data on disk.
    """

    model_config = ConfigDict(extra="forbid")

    # --- input data ---------------------------------------------------------
    genomics_h5ad: Optional[Path] = Field(
        default=None,
        description="Patient x genomics AnnData (.h5ad). None -> synthetic.",
    )
    imaging_h5ad: Optional[Path] = Field(
        default=None,
        description="Patient x imaging AnnData (.h5ad). None -> synthetic.",
    )
    genomics_data_type: GenomicsDataType = Field(
        default="expression",
        description="'variant' (binary mutations) or 'expression' (continuous).",
    )

    # --- gene-expression preprocessing -------------------------------------
    expression_layer: str = "unstranded"
    min_expressed_patients: int = 3
    do_cpm_normalization: bool = True
    do_log1p: bool = True
    do_hvg_selection: bool = True
    n_top_genes: int = 2000
    hvg_method: HVGMethod = Field(
        default="auto",
        description="'auto' uses scanpy if importable, else variance ranking.",
    )
    use_copula: bool = Field(
        default=True,
        description="Gaussian-copula (rank->inverse-normal) marginal transform.",
    )

    # --- working-dimension / analysis settings -----------------------------
    seed: int = 0
    n_components_max: int = 12
    samples_per_dim: int = Field(
        default=5, description="Target ratio n / working-dim after PCA reduction."
    )
    min_working_dim: int = 4
    n_folds: int = 5
    n_perm: int = 200

    # --- demographic deconfounding -----------------------------------------
    demographics_csv: Optional[Path] = None
    confounder_cols: List[str] = Field(default_factory=lambda: ["age", "sex", "education"])
    deconfound: bool = Field(
        default=False,
        description="Run the MAIN analysis on confounder-residualized features.",
    )

    # --- optional sweeps ----------------------------------------------------
    run_sample_complexity_sweep: bool = True
    run_downsample_sweep: bool = True
    downsample_n: Optional[int] = Field(
        default=None, description="Restrict the main analysis to this many patients."
    )
    downsample_seed: int = 0
    downsample_fracs: List[float] = Field(
        default_factory=lambda: [0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
    )

    # --- synthetic fallback parameters -------------------------------------
    synthetic_n: int = 800
    synthetic_p: int = 80
    synthetic_d: int = 40
    synthetic_k: int = 6

    # --- output -------------------------------------------------------------
    output_dir: Path = Field(
        default=Path("rgit_output"),
        description="Directory for stats.json, figures, and the resolved config.",
    )
    save_figures: bool = True
    figure_formats: List[str] = Field(default_factory=lambda: ["pdf", "png"])
    figure_dpi: int = 150

    @model_validator(mode="after")
    def _enforce_variant_preprocessing(self) -> "RecoverabilityConfig":
        # Binary mutation matrices are not RNA-seq counts: CPM, log1p, and HVG
        # selection are all inappropriate, so the notebook disables them in
        # variant mode regardless of the user setting. Mirror that here.
        if self.genomics_data_type == "variant":
            self.do_cpm_normalization = False
            self.do_log1p = False
            self.do_hvg_selection = False
        return self

    # --- derived helpers ----------------------------------------------------
    def is_synthetic(self) -> bool:
        """True when either modality path is missing on disk."""
        if self.genomics_h5ad is None or self.imaging_h5ad is None:
            return True
        return not (Path(self.genomics_h5ad).exists() and Path(self.imaging_h5ad).exists())

    @property
    def figures_dir(self) -> Path:
        return Path(self.output_dir) / "figures"

    @property
    def stats_path(self) -> Path:
        return Path(self.output_dir) / "stats.json"

    @property
    def config_path(self) -> Path:
        return Path(self.output_dir) / "config.json"
