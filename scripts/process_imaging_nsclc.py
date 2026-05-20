import os
import shutil
import argparse
import pandas as pd
from tqdm import tqdm
from tcia_radiology_processing import utils

from rgit import logger

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", default="data/nsclc/imaging", help="Directory containing the NSCLC imaging data.")
parser.add_argument("-p", "--patient_ids", default=None, help="Path to a text file with one patient ID per line to filter to.")
args = parser.parse_args()

data_dir = args.data_dir
patient_ids = args.patient_ids

metadata_name = "metadata.csv"
imaging_metadata_csv = os.path.join(data_dir, metadata_name)

if not os.path.exists(imaging_metadata_csv):
    additional_metadata_xlsx = os.path.join(data_dir, "metadata.xlsx")
    imaging_metadata_df = pd.read_excel(additional_metadata_xlsx)
    imaging_metadata_df.insert(0, "series_id", [f"nsclc_series_{i:05d}" for i in range(len(imaging_metadata_df))])

    col_renames = {
        "Series Instance UID": "Series UID",
        "Study Instance UID": "study_id",
        "Patient ID": "patient_id",
        "Image Count": "Number of Images Original",
    }
    imaging_metadata_df.rename(columns=col_renames, inplace=True)

    if "study_id" not in imaging_metadata_df.columns:
        imaging_metadata_df = imaging_metadata_df.rename(columns={"Study UID": "study_id"})
    if "patient_id" not in imaging_metadata_df.columns:
        imaging_metadata_df = imaging_metadata_df.rename(columns={"Subject ID": "patient_id"})
    if "Modality" not in imaging_metadata_df.columns:
        imaging_metadata_df["Modality"] = (
            imaging_metadata_df["Study Description"]
            .str.upper()
            .str.extract(r"(MR|MRI|CT|PT|XR|X-RAY|X RAY|US|ULTRASOUND|NM)", expand=False)
            .map({
                "MR": "MRI",
                "MRI": "MRI",
                "XR": "X-ray",
                "X-RAY": "X-ray",
                "X RAY": "X-ray",
                "US": "Ultrasound",
                "ULTRASOUND": "Ultrasound",
                "CT": "CT",
                "PT": "PET",
                "NM": "NM"
            })
            .fillna("CT")
        )

    imaging_metadata_df = imaging_metadata_df[["series_id", "Series UID", "study_id", "patient_id", "Modality", "Number of Images Original"]]
    imaging_metadata_df.to_csv(imaging_metadata_csv, index=False)

metadata_df = pd.read_csv(imaging_metadata_csv)
logger.info(f"Number of series: {len(metadata_df)}, Number of patients: {metadata_df['patient_id'].nunique()}")

if patient_ids is not None:
    with open(patient_ids, "r") as f:
        patient_id_list = [line.strip() for line in f]
    metadata_df = metadata_df[metadata_df["patient_id"].isin(patient_id_list)].reset_index(drop=True)
    logger.info(f"Length of patient_id_list: {len(patient_id_list)}")
    logger.info(f"Number of series: {len(metadata_df)}, Number of patients: {metadata_df['patient_id'].nunique()}")

dicom_dir = os.path.join(data_dir, "dicom")

if "is_viable" not in metadata_df.columns:
    metadata_df = utils.add_viable_info(dicom_dir, metadata_df, min_files=15, max_thickness_mm=10, include_kernel_keywords=True, out=imaging_metadata_csv, overwrite=True)

metadata_df = metadata_df[metadata_df["is_viable"]]
metadata_df = metadata_df[metadata_df["Modality"] == "CT"]
logger.info(f"Number of series: {len(metadata_df)}, Number of patients: {metadata_df['patient_id'].nunique()}")

image_filename = "imaging.nii.gz"
nifti_dir_name = "nifti"
nifti_dir = os.path.join(data_dir, nifti_dir_name)

if not os.path.exists(nifti_dir) or len(os.listdir(nifti_dir)) == 0:
    utils.convert_dcm_to_nii_and_organize(dicom_dir, metadata_df, nifti_dir, segimage2itkimage_conda=False)
    logger.info(f"convert_dcm_to_nii_and_organize metrics: {utils.convert_dcm_to_nii_and_organize.last_metrics}")

    # filter out 4D volumes and niis with big max zoom (sometimes some series will have an axial localizer but an otherwise coronal/sagittal series - we want to exclude these)
    metadata_df = utils.check_and_delete_bad_niftis(metadata_df, nifti_dir, image_filename=image_filename, is_4d=True, min_z=25, max_in_plane_aniso=4, max_zoom_maximum=20, filter_if_max_zoom_not_in_si_position=False, out=imaging_metadata_csv)
    metadata_df.to_csv(imaging_metadata_csv, index=False)

    for series_id in tqdm(sorted(os.listdir(nifti_dir)), desc="Processing images"):
        if series_id not in metadata_df["series_id"].values:
            series_dir = os.path.join(nifti_dir, series_id)
            if os.path.exists(series_dir):
                shutil.rmtree(series_dir)

# remove duplicate patients, keeping largest by Number of Images Original
metadata_df = metadata_df.sort_values("Number of Images Original", ascending=False).drop_duplicates("patient_id").reset_index(drop=True)
logger.info(f"Number of series: {len(metadata_df)}, Number of patients: {metadata_df['patient_id'].nunique()}")

for series_id in tqdm(sorted(os.listdir(nifti_dir)), desc="Processing images"):
    patient_dir = os.path.join(nifti_dir, series_id)
    image_file = os.path.join(patient_dir, image_filename)
    image_file = utils.set_canonical_orientation(image_file, out=True)

image_filename = image_file.split("/")[-1]

utils.run_totalsegmentator(nifti_dir, selected_segmentations=["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"], metadata_csv=imaging_metadata_csv, metadata_csv_out=imaging_metadata_csv, remove_small_blobs=True, fill_holes=True, morphological_closing=True, image_filename=image_filename, tumor_mask_filename=None, combined_organ_mask_filename="segmentation_organs.nii.gz", mask_filename_out="segmentation.nii.gz", visualize=False)

metadata_df["image"] = metadata_df["series_id"].apply(lambda sid: os.path.join(nifti_dir, sid, image_filename))
metadata_df["organ_mask"] = metadata_df["series_id"].apply(lambda sid: os.path.join(nifti_dir, sid, "segmentation_organs.nii.gz"))

logger.info(f"Number of series: {len(metadata_df)}, Number of patients: {metadata_df['patient_id'].nunique()}")

metadata_df.to_csv(imaging_metadata_csv, index=False)
logger.info(f"Updated imaging metadata CSV saved to {imaging_metadata_csv}")
