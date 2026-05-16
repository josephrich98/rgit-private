"""
Build a patient × imaging-feature AnnData from a directory of series.

Metadata CSV/parquet must have columns:
  patient_id — obs index in the output AnnData

Embedders:
  pyradiomics — hand-crafted radiomic features (requires SimpleITK + pyradiomics)
  radimagenet  — deep CNN features from a RadImageNet-style model (requires torch + torchvision/timm)
"""

import argparse
import logging
import subprocess
import os
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.sparse import csr_matrix
import SimpleITK as sitk
from tcia_radiology_processing import utils
import torch

from rgit import logger
from rgit.radimagenet import (
    RadImageNetEmbedding,
    convert_image_to_radimagenet_format,
    get_radimagenet_embeddings,
)

# ---------------------------------------------------------------------------
# RadImageNet embedder
# ---------------------------------------------------------------------------

# Module-level cache so the model is loaded once per (type, path, device) triple.
_MODEL_CACHE: dict[tuple[str, str, str], RadImageNetEmbedding] = {}


def _load_radimagenet_model(model_type: str, model_path: str | None, device: str) -> RadImageNetEmbedding:
    key = (model_type, str(model_path), device)
    if key not in _MODEL_CACHE:
        model = RadImageNetEmbedding(model_type)
        if model_path is not None:
            state = torch.load(model_path, map_location=device)
            # Support bare state-dicts and common checkpoint wrappers
            if isinstance(state, dict):
                state = state.get("state_dict", state.get("model", state))
            model.load_state_dict(state, strict=False)
            logger.info(f"Loaded RadImageNet weights from {model_path}")
        else:
            logger.warning("No --model_path given; RadImageNet backbone is randomly initialised.")
        model.to(device).eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def _embed_radimagenet(
    image_file: Path,
    model_path: str | None,
    device: str = "cpu",
    model_type: str = "densenet121",
) -> dict[str, float] | None:
    """Embed a medical image with a RadImageNet backbone.

    Reads the image via SimpleITK, preprocesses it with
    :func:`convert_image_to_radimagenet_format`, and returns a flat dict of
    ``radimagenet_0 … radimagenet_{embed_dim-1}`` float values.  3-D volumes
    are embedded slice-by-slice and then mean-pooled across slices.
    """
    try:
        sitk_img = sitk.ReadImage(str(image_file))
        arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)  # (D,H,W) or (H,W)
    except Exception as exc:
        logger.warning(f"Could not read {image_file}: {exc}")
        return None

    t = convert_image_to_radimagenet_format(arr)  # (3,H,W) or (D,3,H,W)
    model = _load_radimagenet_model(model_type, model_path, device)

    if t.ndim == 3:
        # 2-D image: single frame
        emb = get_radimagenet_embeddings(t.unsqueeze(0).to(device), model)  # (1,E)
        vec = emb.squeeze(0).cpu().numpy()
    else:
        # 3-D volume: embed all slices, mean-pool
        emb = get_radimagenet_embeddings(t.to(device), model)  # (D,E)
        vec = emb.mean(dim=0).cpu().numpy()

    return {f"radimagenet_{i}": float(v) for i, v in enumerate(vec)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Build patient × imaging-feature AnnData")
    parser.add_argument("-m", "--metadata", required=True, help="CSV/parquet with columns 'patient_id' and <image_col>")
    parser.add_argument("-i", "--image_col", default="image", help="Column name in metadata containing image paths")
    parser.add_argument("-o", "--out", default="imaging.h5ad", help="Output .h5ad path")
    parser.add_argument("-e", "--embedder", choices=["pyradiomics", "radimagenet"], default="pyradiomics", help="Embedding method to use")
    parser.add_argument("--pyradiomics_params", default=None, help="[pyradiomics] Path to YAML params file for RadiomicsFeatureExtractor")
    parser.add_argument("--model_path", default=None, help="[radimagenet] Path to pretrained model checkpoint (.pth)")
    parser.add_argument("--model_type", default="densenet121", choices=["densenet121", "resnet50", "inceptionv3"], help="[radimagenet] Backbone architecture")
    parser.add_argument("--device", default="cpu", help="[radimagenet] Torch device (cpu, cuda, mps)")
    parser.add_argument("--extra_meta_cols", nargs="*", default=[], help="Additional metadata columns from the metadata file to store in adata.obs")
    parser.add_argument("--mask_col", default=None, help="Column name in metadata containing mask paths")
    parser.add_argument("--clip_min", type=float, default=None, help="Minimum intensity value to clip to (pyradiomics)")
    parser.add_argument("--clip_max", type=float, default=None, help="Maximum intensity value to clip to (pyradiomics)")
    parser.add_argument("--resample_spacing", type=str, default=None, help="Target spacing for resampling (mm), e.g. '1.0,1.0,1.0'")
    parser.add_argument("--apply_mask", action="store_true", help="Whether to apply the mask to the image before feature extraction")
    parser.add_argument("--crop_size", type=str, default=None, help="Target size for cropping/padding (voxels), e.g. '128,128,64'")
    parser.add_argument("--normalize", action="store_true", help="Whether to z-score normalize features across the dataset")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- load metadata ---
    meta_path = Path(args.metadata)
    if meta_path.suffix == ".parquet":
        meta = pd.read_parquet(meta_path)
    else:
        meta = pd.read_csv(meta_path)

    required = {"patient_id", args.image_col}
    if not required.issubset(meta.columns):
        raise ValueError(f"Metadata must have columns {required}; found {set(meta.columns)}")

    meta = meta.drop_duplicates(subset=["patient_id"])
    logger.info(f"Metadata: {len(meta)} series")
    
    if args.embedder == "pyradiomics":
        from radiomics import featureextractor
        args.clip_min, args.clip_max, args.resample_spacing, args.apply_mask, args.crop_size, args.normalize = None, None, None, None, None, None
        extractor = featureextractor.RadiomicsFeatureExtractor(args.pyradiomics_params)
        label = 1 if mask_file else None

        # utils.prepare_csv_for_pyradiomics(nifti_dir, output_csv_path=pyradiomics_input_csv_path, imaging_file_name=image_filename, mask_file_name=mask_filename)  # image_filename_nii, mask_filename_nii
        # pyradiomics_command = "pyradiomics <path/to/csv> -j 32"
        # if args.pyradiomics_params:
        #     pyradiomics_command += f" --param {args.pyradiomics_params}"
        # subprocess.run(pyradiomics_command, shell=True, check=True)

    # --- embed each series ---
    records = []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Embedding series"):
        patient_id = str(row["patient_id"])

        image_file = Path(row[args.image_col])
        mask_file = Path(row[args.mask_col]) if args.mask_col and args.mask_col in row else None
        if not image_file.exists():
            logger.warning(f"Image file for patient '{patient_id}' not found at {image_file}; skipping")
            continue

        if args.clip_min is not None or args.clip_max is not None:
            image_file = utils.clip_intensity_range(image_file, clip_min=args.clip_min, clip_max=args.clip_max, out=True)

        if args.resample_spacing is not None:
            target_spacing = tuple(map(float, args.resample_spacing.split(",")))
            image_file = utils.resample_image(image_file, target_spacing=target_spacing, is_label=False, out=True)
            mask_file = utils.resample_image(mask_file, target_spacing=target_spacing, is_label=True, out=True)
        
        if args.apply_mask and mask_file is not None:
            image_file, mask_file = utils.apply_mask(image_file, mask_file, label=1, min_value=args.clip_min, crop=True, pad_after_crop=5, out_image=True, out_mask=True)
        
        if args.crop_size is not None:
            xdim, ydim, zdim = map(int, args.crop_size.split(","))
            image_file = utils.crop_and_pad(image_file, xdim=xdim, ydim=ydim, zdim=zdim, min_value=args.clip_min, out=True)
            mask_file = utils.crop_and_pad(mask_file, xdim=xdim, ydim=ydim, zdim=zdim, min_value=0, out=True)

        if args.normalize:
            image_file = utils.normalize_intensity(image_file, normalization_method="volume", out=True)

        logger.debug(f"image_file: {image_file}, mask_file: {mask_file}")
        
        if args.embedder == "pyradiomics":
            features = extractor.execute(image_file, mask_file, label=label)
        elif args.embedder == "radimagenet":
            features = _embed_radimagenet(image_file, args.model_path, args.device, args.model_type)
        else:
            raise ValueError(f"Unsupported embedder: {args.embedder}")

        if features is None:
            logger.warning(f"No features extracted for patient {patient_id}; skipping")
            continue

        features["patient_id"] = patient_id
        for col in args.extra_meta_cols:
            if col in row.index:
                features[col] = row[col]
        records.append(features)

    if not records:
        raise RuntimeError("No features were extracted from any series.")

    feat_df = pd.DataFrame(records)
    feat_df = feat_df.set_index("patient_id")

    meta_cols = ["series_id"] + [c for c in args.extra_meta_cols if c in feat_df.columns]
    obs_df = feat_df[meta_cols].copy()
    X_df = feat_df.drop(columns=meta_cols, errors="ignore")

    # drop constant or all-NaN feature columns
    X_df = X_df.dropna(axis=1, how="all")
    X_df = X_df.loc[:, X_df.nunique() > 0]

    X = csr_matrix(X_df.values.astype(np.float32))
    adata = ad.AnnData(
        X=X,
        obs=obs_df,
        var=pd.DataFrame(index=X_df.columns),
    )
    adata.var_names_make_unique()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    logger.info(f"Saved imaging AnnData: {adata.shape} → {out_path}")


if __name__ == "__main__":
    main()
