"""
Standard brain MRI processing for ADNI.

Expects a directory of nested DICOM directories at <data_dir>/dicom (the
default ADNI layout is Subject/Visit/Sequence/SeriesUID/*.dcm).  Optionally
reads <data_dir>/metadata.csv (or .xlsx) and normalizes it to the same
schema produced by process_imaging_nsclc.py so that the resulting metadata
file can be fed directly into make_imaging_matrix.py.

Pipeline per series:
  1. DICOM -> NIfTI conversion + organization
  2. Set canonical (RAS) orientation
  3. N4 bias-field correction
  4. Resample to 1 mm isotropic
  5. Skull stripping via TotalSegmentator (task='total_mr', class='brain')
  6. Brain-region segmentation via TotalSegmentator (task='brain_structures')
"""

import os
import shutil
import argparse
import subprocess
import pandas as pd
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from tqdm import tqdm
from tcia_radiology_processing import utils

from rgit import logger


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", default="data/adni/imaging", help="Directory containing the ADNI imaging data.")
parser.add_argument("-p", "--patient_ids", default=None, help="Path to a text file with one patient ID per line to filter to.")
parser.add_argument("--resample_spacing", default="1.0,1.0,1.0", help="Target voxel spacing (mm) for resampling, e.g. '1.0,1.0,1.0'.")
parser.add_argument("--skip_bias_correction", action="store_true", help="Skip N4 bias-field correction (faster).")
parser.add_argument("--device", default=None, help="Device for TotalSegmentator (e.g. 'gpu', 'cpu').")
args = parser.parse_args()

data_dir = args.data_dir
metadata_name = "metadata.csv"
imaging_metadata_csv = os.path.join(data_dir, metadata_name)
dicom_dir = os.path.join(data_dir, "dicom")

# ---------------------------------------------------------------------------
# Build / normalize metadata CSV
# ---------------------------------------------------------------------------
if not os.path.exists(imaging_metadata_csv):
    additional_metadata_xlsx = os.path.join(data_dir, "metadata.xlsx")
    if os.path.exists(additional_metadata_xlsx):
        imaging_metadata_df = pd.read_excel(additional_metadata_xlsx)
    else:
        # Derive metadata by walking the dicom directory tree
        logger.info(f"No metadata file found; scanning {dicom_dir} to build one.")
        import pydicom
        rows = []
        for root, _, files in os.walk(dicom_dir):
            dcms = [f for f in files if f.lower().endswith(".dcm")]
            if not dcms:
                continue
            try:
                ds = pydicom.dcmread(os.path.join(root, dcms[0]), stop_before_pixels=True)
            except Exception as exc:
                logger.warning(f"Failed to read {os.path.join(root, dcms[0])}: {exc}")
                continue
            rows.append({
                "Series UID": getattr(ds, "SeriesInstanceUID", ""),
                "Study UID": getattr(ds, "StudyInstanceUID", ""),
                "Subject ID": getattr(ds, "PatientID", os.path.basename(root)),
                "Modality": getattr(ds, "Modality", "MR"),
                "Series Description": getattr(ds, "SeriesDescription", ""),
                "Number of Images": len(dcms),
            })
        imaging_metadata_df = pd.DataFrame(rows)

    imaging_metadata_df.insert(0, "series_id", [f"adni_series_{i:05d}" for i in range(len(imaging_metadata_df))])

    col_renames = {
        "Series Instance UID": "Series UID",
        "Study Instance UID": "study_id",
        "Study UID": "study_id",
        "Patient ID": "patient_id",
        "Subject ID": "patient_id",
        "Subject": "patient_id",
        "Image Count": "Number of Images Original",
        "Number of Images": "Number of Images Original",
    }
    imaging_metadata_df.rename(columns=col_renames, inplace=True)

    if "Modality" not in imaging_metadata_df.columns:
        imaging_metadata_df["Modality"] = "MRI"
    imaging_metadata_df["Modality"] = imaging_metadata_df["Modality"].replace({"MR": "MRI"})

    keep = ["series_id", "Series UID", "study_id", "patient_id", "Modality", "Number of Images Original"]
    keep = [c for c in keep if c in imaging_metadata_df.columns]
    imaging_metadata_df = imaging_metadata_df[keep]
    imaging_metadata_df.to_csv(imaging_metadata_csv, index=False)

metadata_df = pd.read_csv(imaging_metadata_csv)
logger.info(f"Number of series: {len(metadata_df)}, Number of patients: {metadata_df['patient_id'].nunique()}")

if args.patient_ids is not None:
    with open(args.patient_ids, "r") as f:
        patient_id_list = [line.strip() for line in f]
    metadata_df = metadata_df[metadata_df["patient_id"].isin(patient_id_list)].reset_index(drop=True)
    logger.info(f"After patient filter: {len(metadata_df)} series, {metadata_df['patient_id'].nunique()} patients")

# ---------------------------------------------------------------------------
# Viability filter (MR-appropriate thresholds)
# ---------------------------------------------------------------------------
if "is_viable" not in metadata_df.columns:
    metadata_df = utils.add_viable_info(
        dicom_dir, metadata_df,
        min_files=15, max_thickness_mm=6, include_kernel_keywords=False,
        out=imaging_metadata_csv, overwrite=True,
    )

metadata_df = metadata_df[metadata_df["is_viable"]]
metadata_df = metadata_df[metadata_df["Modality"] == "MRI"]
logger.info(f"After viability/MR filter: {len(metadata_df)} series, {metadata_df['patient_id'].nunique()} patients")

# ---------------------------------------------------------------------------
# DICOM -> NIfTI
# ---------------------------------------------------------------------------
image_filename = "imaging.nii.gz"
nifti_dir = os.path.join(data_dir, "nifti")

if not os.path.exists(nifti_dir) or len(os.listdir(nifti_dir)) == 0:
    utils.convert_dcm_to_nii_and_organize(dicom_dir, metadata_df, nifti_dir, segimage2itkimage_conda=False)
    logger.info(f"convert_dcm_to_nii_and_organize metrics: {utils.convert_dcm_to_nii_and_organize.last_metrics}")

    metadata_df = utils.check_and_delete_bad_niftis(
        metadata_df, nifti_dir, image_filename=image_filename,
        is_4d=True, min_z=25, max_in_plane_aniso=4, max_zoom_maximum=20,
        filter_if_max_zoom_not_in_si_position=False, out=imaging_metadata_csv,
    )
    metadata_df.to_csv(imaging_metadata_csv, index=False)

    for series_id in tqdm(sorted(os.listdir(nifti_dir)), desc="Pruning bad series"):
        if series_id not in metadata_df["series_id"].values:
            series_dir = os.path.join(nifti_dir, series_id)
            if os.path.exists(series_dir):
                shutil.rmtree(series_dir)

# de-duplicate patients: keep largest series per patient
metadata_df = metadata_df.sort_values("Number of Images Original", ascending=False).drop_duplicates("patient_id").reset_index(drop=True)
logger.info(f"After de-duplication: {len(metadata_df)} series, {metadata_df['patient_id'].nunique()} patients")

# ---------------------------------------------------------------------------
# Per-series preprocessing: orient -> N4 -> resample
# ---------------------------------------------------------------------------
target_spacing = tuple(map(float, args.resample_spacing.split(",")))

for series_id in tqdm(sorted(metadata_df["series_id"].tolist()), desc="Orient + bias + resample"):
    series_dir = os.path.join(nifti_dir, series_id)
    image_file = os.path.join(series_dir, image_filename)
    if not os.path.exists(image_file):
        continue

    image_file = utils.set_canonical_orientation(image_file, out=True)

    if not args.skip_bias_correction:
        img = sitk.ReadImage(image_file, sitk.sitkFloat32)
        mask = sitk.OtsuThreshold(img, 0, 1, 200)
        corrected = sitk.N4BiasFieldCorrectionImageFilter().Execute(img, mask)
        sitk.WriteImage(corrected, image_file)

    utils.resample_image(image_file, target_spacing=target_spacing, is_label=False, out=True)

# ---------------------------------------------------------------------------
# Skull stripping via TotalSegmentator (total_mr -> brain class)
# ---------------------------------------------------------------------------
brain_mask_filename = "brain_mask.nii.gz"
skullstripped_filename = "imaging_brain.nii.gz"

utils.run_totalsegmentator(
    nifti_dir,
    selected_segmentations=["brain"],
    metadata_csv=imaging_metadata_csv,
    metadata_csv_out=imaging_metadata_csv,
    remove_small_blobs=True, fill_holes=True, morphological_closing=True,
    image_filename=image_filename,
    tumor_mask_filename=None,
    combined_organ_mask_filename=brain_mask_filename,
    mask_filename_out="segmentation_brain_only.nii.gz",
    task="total_mr",
    visualize=False,
    device=args.device,
)

for series_id in tqdm(sorted(metadata_df["series_id"].tolist()), desc="Skull stripping"):
    series_dir = os.path.join(nifti_dir, series_id)
    image_file = os.path.join(series_dir, image_filename)
    mask_file = os.path.join(series_dir, brain_mask_filename)
    if not (os.path.exists(image_file) and os.path.exists(mask_file)):
        continue
    img = nib.load(image_file)
    msk = nib.load(mask_file)
    img_data = img.get_fdata()
    msk_data = (msk.get_fdata() > 0).astype(img_data.dtype)
    stripped = img_data * msk_data
    nib.save(nib.Nifti1Image(stripped, img.affine, img.header), os.path.join(series_dir, skullstripped_filename))

# ---------------------------------------------------------------------------
# Brain-region segmentation
# ---------------------------------------------------------------------------
segmentation_filename = "segmentation.nii.gz"

utils.run_totalsegmentator(
    nifti_dir,
    selected_segmentations=None,
    metadata_csv=imaging_metadata_csv,
    metadata_csv_out=imaging_metadata_csv,
    remove_small_blobs=False, fill_holes=False, morphological_closing=False,
    image_filename=skullstripped_filename,
    tumor_mask_filename=None,
    combined_organ_mask_filename=segmentation_filename,
    mask_filename_out=segmentation_filename,
    task="brain_structures",
    visualize=False,
    device=args.device,
)

# ---------------------------------------------------------------------------
# Finalize metadata CSV
# ---------------------------------------------------------------------------
metadata_df["image"] = metadata_df["series_id"].apply(lambda sid: os.path.join(nifti_dir, sid, skullstripped_filename))
metadata_df["organ_mask"] = metadata_df["series_id"].apply(lambda sid: os.path.join(nifti_dir, sid, brain_mask_filename))
metadata_df["segmentation"] = metadata_df["series_id"].apply(lambda sid: os.path.join(nifti_dir, sid, segmentation_filename))

logger.info(f"Final: {len(metadata_df)} series, {metadata_df['patient_id'].nunique()} patients")
metadata_df.to_csv(imaging_metadata_csv, index=False)
logger.info(f"Updated imaging metadata CSV saved to {imaging_metadata_csv}")
