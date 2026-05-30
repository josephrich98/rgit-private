"""Command-line entry point for the recoverability analysis.

Installed as the ``rgit-recoverability`` console script (see ``pyproject.toml``),
this reproduces the notebook headlessly. With no data paths it runs the
synthetic ground-truth dataset, so a fresh ``pip install rgit`` can be exercised
immediately::

    rgit-recoverability --output-dir out/synthetic
    rgit-recoverability --genomics data/g.h5ad --imaging data/x.h5ad \
        --genomics-data-type variant --output-dir out/kirc

A JSON config file (matching :class:`rgit.RecoverabilityConfig`) may be supplied
with ``--config``; any explicit flags override its fields.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rgit.config import RecoverabilityConfig
from rgit.report import run_recoverability_analysis


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rgit-recoverability",
        description="Reproduce the radiogenomic recoverability analysis (notebook as a CLI).",
    )
    p.add_argument("--config", type=Path, help="JSON file of RecoverabilityConfig fields.")
    p.add_argument("--genomics", type=Path, dest="genomics_h5ad",
                   help="Patient x genomics .h5ad (omit for synthetic).")
    p.add_argument("--imaging", type=Path, dest="imaging_h5ad",
                   help="Patient x imaging .h5ad (omit for synthetic).")
    p.add_argument("--genomics-data-type", choices=["variant", "expression"],
                   dest="genomics_data_type")
    p.add_argument("--output-dir", type=Path, dest="output_dir")
    p.add_argument("--seed", type=int)
    p.add_argument("--n-perm", type=int, dest="n_perm")
    p.add_argument("--n-folds", type=int, dest="n_folds")
    p.add_argument("--no-figures", action="store_true", help="Skip writing figures.")
    p.add_argument("--no-sweeps", action="store_true",
                   help="Skip the sample-complexity and downsample sweeps.")
    p.add_argument("--deconfound", action="store_true",
                   help="Run the main analysis on confounder-residualized features.")
    p.add_argument("--demographics-csv", type=Path, dest="demographics_csv")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _config_from_args(args: argparse.Namespace) -> RecoverabilityConfig:
    base: dict = {}
    if args.config:
        base = json.loads(Path(args.config).read_text())

    overrides = {}
    for field in ("genomics_h5ad", "imaging_h5ad", "genomics_data_type", "output_dir",
                  "seed", "n_perm", "n_folds", "demographics_csv"):
        val = getattr(args, field, None)
        if val is not None:
            overrides[field] = val
    if args.no_figures:
        overrides["save_figures"] = False
    if args.no_sweeps:
        overrides["run_sample_complexity_sweep"] = False
        overrides["run_downsample_sweep"] = False
    if args.deconfound:
        overrides["deconfound"] = True

    return RecoverabilityConfig(**{**base, **overrides})


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.getLogger("rgit").setLevel(logging.DEBUG if args.verbose else logging.INFO)

    cfg = _config_from_args(args)
    mode = "synthetic" if cfg.is_synthetic() else "real data"
    print(f"rgit-recoverability: running on {mode} -> {cfg.output_dir}", file=sys.stderr)

    report = run_recoverability_analysis(cfg)

    s = report.stats
    print(f"\nwrote {report.stats_path}")
    print(f"  n_samples           : {s.get('n_samples')}")
    print(f"  I(G;X) in-sample    : {s.get('in_sample_MI_bits', float('nan')):.2f} bits")
    er = s.get("effective_identifiable_rank", {})
    print(f"  effective rank      : {er.get('rank')} of {er.get('K_tested')}")
    mub = s.get("mi_upper_bound", {})
    print(f"  95% UCL on I(G;X)   : {mub.get('ucl_95_bits', float('nan')):.2f} bits")
    if "synthetic_validation" in s:
        sv = s["synthetic_validation"]
        print(f"  subspace alignment  : mean cos = {sv['mean_alignment']:.3f} (true k={sv['true_k']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
