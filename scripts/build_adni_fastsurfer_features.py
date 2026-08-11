"""Build the ADNI imaging feature matrix from FastSurfer outputs.

Two feature families per subject, concatenated into one patient x feature
AnnData written to ``data/adni/imaging/fastsurfer.h5ad``:

1. **FastSurfer morphometry** -- volume + intensity summary statistics for every
   region in ``stats/aseg+DKT.stats`` (the standard segmentation table).
2. **pyradiomics texture/shape** -- full pyradiomics feature set extracted from
   ``mri/orig_nu.mgz`` masked by each of the key Alzheimer's-disease regions in
   ``mri/aparc.DKTatlas+aseg.deep.mgz`` (hippocampus, entorhinal cortex,
   fusiform, inferior temporal, precuneus, posterior cingulate -- bilateral).

Subjects are keyed by ADNI subject ID (e.g. ``002_S_0413``) parsed from the
FastSurfer output directory name, so the matrix aligns with the genomics
AnnData once that is re-keyed by ``SubjectID``.
"""
from __future__ import annotations

import os
import re
import argparse
import logging
import warnings
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import anndata as ad
from scipy.sparse import csr_matrix

logging.getLogger("radiomics").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
DEFAULT_IMG_ROOT = REPO / "data/adni/imaging/nifti_fastsurfer/nifti_fastsurfer"
DEFAULT_OUT = REPO / "data/adni/imaging/fastsurfer.h5ad"

# Key AD regions -> (DKT/aseg label values to union into one bilateral mask).
AD_REGIONS = {
    "hippocampus":          [17, 53],
    "entorhinal":           [1006, 2006],
    "fusiform":             [1007, 2007],
    "inferiortemporal":     [1009, 2009],
    "precuneus":            [1025, 2025],
    "posteriorcingulate":   [1023, 2023],
}

SUBJ_RE = re.compile(r"ADNI_(\d{3}_S_\d{4})_")


def subject_id(dirname: str) -> str | None:
    m = SUBJ_RE.match(dirname)
    return m.group(1) if m else None


def parse_stats(stats_path: Path) -> dict[str, float]:
    """Volume + intensity summaries per region from an aseg+DKT.stats table."""
    feats: dict[str, float] = {}
    with open(stats_path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            # Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange
            if len(parts) < 10:
                continue
            struct = parts[4]
            try:
                feats[f"stats_{struct}_volume"]    = float(parts[3])
                feats[f"stats_{struct}_normMean"]  = float(parts[5])
                feats[f"stats_{struct}_normStd"]   = float(parts[6])
                feats[f"stats_{struct}_normRange"] = float(parts[9])
            except ValueError:
                continue
    return feats


def _to_sitk(arr: np.ndarray, is_mask: bool) -> sitk.Image:
    img = sitk.GetImageFromArray(arr.astype(np.uint8 if is_mask else np.float32))
    img.SetSpacing((1.0, 1.0, 1.0))  # FastSurfer conform space is 1 mm isotropic
    return img


def extract_subject(args) -> dict | None:
    sid, dirname, img_root = args
    d = Path(img_root) / dirname
    stats_file = d / "stats/aseg+DKT.stats"
    norm_file = d / "mri/orig_nu.mgz"
    seg_file = d / "mri/aparc.DKTatlas+aseg.deep.mgz"
    if not (stats_file.exists() and norm_file.exists() and seg_file.exists()):
        return None

    from radiomics import featureextractor

    feats: dict[str, float] = parse_stats(stats_file)

    norm = np.asarray(nib.load(norm_file).dataobj).astype(np.float32)
    seg = np.asarray(nib.load(seg_file).dataobj).astype(np.int32)
    img_sitk = _to_sitk(norm, is_mask=False)

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings["binWidth"] = 8  # intensities are 0-255 uint8-scale

    for region, labels in AD_REGIONS.items():
        mask = np.isin(seg, labels).astype(np.uint8)
        if mask.sum() < 30:  # too small / missing region
            continue
        mask_sitk = _to_sitk(mask, is_mask=True)
        mask_sitk.CopyInformation(img_sitk)
        try:
            res = extractor.execute(img_sitk, mask_sitk, label=1)
        except Exception:
            continue
        for k, v in res.items():
            if k.startswith("original_"):
                try:
                    feats[f"radiomics_{region}_{k}"] = float(np.asarray(v).item())
                except (ValueError, TypeError):
                    pass

    feats["patient_id"] = sid
    return feats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--img-root", type=Path, default=DEFAULT_IMG_ROOT,
        help="Root directory of FastSurfer per-subject output dirs "
             f"(default: {DEFAULT_IMG_ROOT})",
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=DEFAULT_OUT,
        help=f"Output .h5ad path (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()
    img_root, out = args.img_root, args.out

    out.parent.resolve().mkdir(parents=True, exist_ok=True)

    n_proc = int(os.environ.get("N_PROC", "16"))
    dirs = sorted(os.listdir(img_root))
    jobs = [(subject_id(d), d, img_root) for d in dirs if subject_id(d)]
    print(f"{len(jobs)} subjects to process with {n_proc} workers", flush=True)

    records = []
    with Pool(n_proc) as pool:
        for i, rec in enumerate(pool.imap_unordered(extract_subject, jobs, chunksize=4)):
            if rec is not None:
                records.append(rec)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(jobs)} done ({len(records)} ok)", flush=True)

    df = pd.DataFrame(records).set_index("patient_id")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, df.nunique(dropna=False) > 1]  # drop constant columns
    df = df.fillna(df.median(numeric_only=True))
    print(f"feature matrix: {df.shape[0]} subjects x {df.shape[1]} features", flush=True)

    adata = ad.AnnData(
        X=csr_matrix(df.values.astype(np.float32)),
        obs=pd.DataFrame(index=df.index.astype(str)),
        var=pd.DataFrame(index=df.columns),
    )
    adata.var_names_make_unique()
    adata.write_h5ad(out)
    print(f"saved {adata.shape} -> {out}", flush=True)


if __name__ == "__main__":
    main()
