"""
Build a patient × imaging-feature AnnData from a directory of series.

Metadata CSV/parquet must have columns:
  series_id  — name of the subdirectory containing that series' images
  patient_id — obs index in the output AnnData

Embedders:
  pyradiomics — hand-crafted radiomic features (requires SimpleITK + pyradiomics)
  radimagenet  — deep CNN features from a RadImageNet-style model (requires torch + torchvision/timm)
"""

import argparse
import logging
import os
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

_DICOM_EXTS = {".dcm", ".ima", ".dicom"}
_NIFTI_EXTS = {".nii", ".gz"}  # .nii.gz handled separately


def _find_series_dir(root: Path, series_id: str) -> Path | None:
    """Return the first directory under root whose name matches series_id."""
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d == series_id:
                return Path(dirpath) / d
    return None


def _collect_files(directory: Path, extensions: set[str]) -> list[Path]:
    files = []
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix.lower() in extensions:
            files.append(p)
        elif p.is_file() and p.name.lower().endswith(".nii.gz"):
            files.append(p)
    return files


# ---------------------------------------------------------------------------
# Pyradiomics embedder
# ---------------------------------------------------------------------------

def _embed_pyradiomics(series_dir: Path, mask_dir: Path | None, params: dict | None) -> dict[str, float] | None:
    try:
        import SimpleITK as sitk
        from radiomics import featureextractor
    except ImportError:
        raise ImportError("pyradiomics and SimpleITK are required: pip install pyradiomics SimpleITK")

    extractor = featureextractor.RadiomicsFeatureExtractor(**(params or {}))

    # --- load image ---
    niftis = [p for p in series_dir.rglob("*") if p.suffix in {".nii"} or str(p).endswith(".nii.gz")]
    dicoms = [p for p in series_dir.rglob("*") if p.suffix.lower() in _DICOM_EXTS]

    if niftis:
        image_path = str(niftis[0])
        image = sitk.ReadImage(image_path)
    elif dicoms:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(series_dir))
        if not series_ids:
            logger.warning(f"No DICOM series found in {series_dir}")
            return None
        dicom_files = reader.GetGDCMSeriesFileNames(str(series_dir), series_ids[0])
        reader.SetFileNames(dicom_files)
        image = reader.Execute()
        image_path = None
    else:
        logger.warning(f"No images found in {series_dir}")
        return None

    # --- load or create mask ---
    mask = None
    if mask_dir is not None:
        mask_candidates = list(Path(mask_dir).rglob(f"{series_dir.name}*"))
        if mask_candidates:
            mask = sitk.ReadImage(str(mask_candidates[0]))

    if mask is None:
        # whole-volume binary mask
        mask = sitk.Cast(sitk.BinaryFillhole(image > sitk.GetArrayFromImage(image).min()), sitk.sitkUInt8)
        mask = sitk.GetImageFromArray(
            np.ones(sitk.GetArrayFromImage(image).shape, dtype=np.uint8)
        )
        mask.CopyInformation(image)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = extractor.execute(image, mask)
        features = {k: float(v) for k, v in result.items() if not k.startswith("diagnostics_")}
        return features
    except Exception as e:
        logger.warning(f"pyradiomics failed for {series_dir}: {e}")
        return None


# ---------------------------------------------------------------------------
# RadImageNet embedder
# ---------------------------------------------------------------------------

def _embed_radimagenet(series_dir: Path, model_path: str | None, device: str = "cpu") -> dict[str, float] | None:
    try:
        import torch
        import torchvision.transforms as T
        from PIL import Image
    except ImportError:
        raise ImportError("torch and torchvision are required: pip install torch torchvision Pillow")

    try:
        import timm
        _has_timm = True
    except ImportError:
        _has_timm = False

    device = torch.device(device)

    # --- build model ---
    if model_path and os.path.exists(model_path):
        # Load arbitrary checkpoint: assume state_dict keys match a resnet50 backbone
        import torchvision.models as tvm
        backbone = tvm.resnet50(weights=None)
        backbone.fc = torch.nn.Identity()
        state = torch.load(model_path, map_location=device)
        # handle wrapped state dicts
        state = state.get("state_dict", state.get("model", state))
        state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        if missing:
            logger.debug(f"Missing keys when loading model: {missing[:5]}")
        model = backbone
    elif _has_timm:
        logger.info("No model_path provided; loading timm resnet50 with ImageNet weights as proxy.")
        model = timm.create_model("resnet50", pretrained=True, num_classes=0)
    else:
        import torchvision.models as tvm
        logger.info("No model_path provided; using torchvision ResNet50 (ImageNet) as proxy.")
        backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        backbone.fc = torch.nn.Identity()
        model = backbone

    model = model.to(device).eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- collect image files (DICOM or common formats) ---
    img_files = sorted(
        p for p in series_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".dcm", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        or str(p).endswith(".nii.gz")
        or p.suffix in {".nii"}
    )

    if not img_files:
        logger.warning(f"No image files found in {series_dir}")
        return None

    embeddings = []
    for img_path in img_files:
        try:
            if img_path.suffix.lower() == ".dcm":
                import SimpleITK as sitk
                arr = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path))).astype(np.float32)
                # normalize to 0-255 per slice
                for sl in range(arr.shape[0]) if arr.ndim == 3 else [0]:
                    slc = arr[sl] if arr.ndim == 3 else arr
                    slc = (slc - slc.min()) / (slc.max() - slc.min() + 1e-8) * 255
                    pil = Image.fromarray(slc.astype(np.uint8))
                    tensor = transform(pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = model(tensor).squeeze().cpu().numpy()
                    embeddings.append(emb)
            elif img_path.suffix in {".nii"} or str(img_path).endswith(".nii.gz"):
                import SimpleITK as sitk
                arr = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path))).astype(np.float32)
                slices = arr if arr.ndim == 3 else arr[None]
                for sl in slices:
                    sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8) * 255
                    pil = Image.fromarray(sl.astype(np.uint8))
                    tensor = transform(pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = model(tensor).squeeze().cpu().numpy()
                    embeddings.append(emb)
            else:
                pil = Image.open(img_path).convert("L")
                tensor = transform(pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(tensor).squeeze().cpu().numpy()
                embeddings.append(emb)
        except Exception as e:
            logger.warning(f"Failed to embed {img_path}: {e}")
            continue

    if not embeddings:
        return None

    mean_emb = np.mean(embeddings, axis=0)
    return {f"feat_{i}": float(v) for i, v in enumerate(mean_emb)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Build patient × imaging-feature AnnData")
    parser.add_argument("--dir", required=True, help="Root directory containing series subdirectories")
    parser.add_argument("--metadata", required=True, help="CSV/parquet with columns 'series_id' and 'patient_id'")
    parser.add_argument("--out", required=True, help="Output .h5ad path")
    parser.add_argument(
        "--embedder", choices=["pyradiomics", "radimagenet"], default="pyradiomics",
        help="Embedding method to use"
    )
    parser.add_argument("--mask_dir", default=None, help="[pyradiomics] Directory containing mask files (matched by series_id prefix)")
    parser.add_argument("--pyradiomics_params", default=None, help="[pyradiomics] Path to YAML params file for RadiomicsFeatureExtractor")
    parser.add_argument("--model_path", default=None, help="[radimagenet] Path to pretrained model checkpoint (.pth)")
    parser.add_argument("--device", default="cpu", help="[radimagenet] Torch device (cpu, cuda, mps)")
    parser.add_argument("--extra_meta_cols", nargs="*", default=[], help="Additional metadata columns from the metadata file to store in adata.obs")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.dir)
    if not root.is_dir():
        raise ValueError(f"--dir does not exist or is not a directory: {root}")

    # --- load metadata ---
    meta_path = Path(args.metadata)
    if meta_path.suffix == ".parquet":
        meta = pd.read_parquet(meta_path)
    else:
        meta = pd.read_csv(meta_path)

    required = {"series_id", "patient_id"}
    if not required.issubset(meta.columns):
        raise ValueError(f"Metadata must have columns {required}; found {set(meta.columns)}")

    meta = meta.drop_duplicates(subset=["series_id"])
    logger.info(f"Metadata: {len(meta)} series from {meta['patient_id'].nunique()} patients")

    # pyradiomics params
    pyrad_params = None
    if args.pyradiomics_params:
        pyrad_params = {"params": args.pyradiomics_params}

    # --- embed each series ---
    records = []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Embedding series"):
        series_id = str(row["series_id"])
        patient_id = str(row["patient_id"])

        series_dir = _find_series_dir(root, series_id)
        if series_dir is None:
            logger.warning(f"Series directory '{series_id}' not found under {root}; skipping")
            continue

        if args.embedder == "pyradiomics":
            mask_dir = Path(args.mask_dir) if args.mask_dir else None
            features = _embed_pyradiomics(series_dir, mask_dir, pyrad_params)
        else:
            features = _embed_radimagenet(series_dir, args.model_path, args.device)

        if features is None:
            logger.warning(f"No features extracted for series {series_id} ({patient_id}); skipping")
            continue

        features["patient_id"] = patient_id
        features["series_id"] = series_id
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
