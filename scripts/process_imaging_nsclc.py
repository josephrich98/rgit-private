import os
import sys
import shutil
import subprocess
import argparse
import pandas as pd
from tqdm import tqdm
import yaml
import nibabel as nib
from tcia_radiology_processing import utils
from tcia_radiology_processing.constants import tcia_dataset_to_info

from rgit import logger

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", default="data/tcga_kirc/imaging", help="Directory to store the data.")
args = parser.parse_args()

#$ currently hard-coded for TCGA-KIRC where each patient is de-duplicated, totalsegmentator works returns kidney, and I assume I have tumor masks

data_dir = args.data_dir

metadata_name = "metadata.csv"
imaging_metadata_csv = os.path.join(data_dir, metadata_name)

if os.path.exists(imaging_metadata_csv):
    imaging_metadata_csv_columns = pd.read_csv(imaging_metadata_csv, nrows=0).columns
    expected_columns = {"patient_id", "tumor_side", "imaging", "tumor_mask", "organ_mask", "tumor_and_organ_mask"}
    if not expected_columns.issubset(imaging_metadata_csv_columns):
        logger.warning(f"Existing imaging metadata CSV at {imaging_metadata_csv} is missing expected columns. Expected at least: {expected_columns}. Found: {imaging_metadata_csv_columns}. The existing file will be backed up and a new one will be created.")

nifti_dir_name = "nifti_usc"
nifti_dir = os.path.join(data_dir, nifti_dir_name)

image_filename = "0502_VENOUS.nii"
tumor_mask_filename = "segmentation_tumor.nii.gz"
tumor_and_organ_mask_filename = "segmentation.nii.gz"  # tumor + organs
tumor_and_organ_mask_binary_filename = "segmentation_binary.nii.gz"  # tumor + organs, with all values ≥1 set to 1

if not os.path.exists(nifti_dir) or len(os.listdir(nifti_dir)) == 0:
    logger.info(f"Downloading and processing TCGA-KIRC imaging data into {nifti_dir}...")
    _ = utils.download_usc_tcga_kirc_data(data_dir, imaging_metadata_csv=imaging_metadata_csv, dst_dir_name=nifti_dir_name)

metadata_df = pd.read_csv(imaging_metadata_csv)

logger.info(f"Processing images in {nifti_dir} to ensure canonical orientation")
for patient_id in tqdm(sorted(os.listdir(nifti_dir)), desc="Processing images"):
    patient_dir = os.path.join(nifti_dir, patient_id)
    image_file = os.path.join(patient_dir, image_filename)
    mask_file = os.path.join(patient_dir, tumor_mask_filename)

    image_filename = "imaging_oriented.nii.gz"
    tumor_mask_filename = "segmentation_tumor_oriented.nii.gz"
    
    image_file = utils.set_canonical_orientation(image_file, out=os.path.join(patient_dir, image_filename))
    mask_file = utils.set_canonical_orientation(mask_file, out=os.path.join(patient_dir, tumor_mask_filename))

logger.info(f"Running TotalSegmentator to segment kidneys and create combined tumor+organ masks")
utils.run_totalsegmentator(nifti_dir, selected_segmentations=["kidney_left", "kidney_right"], metadata_csv=imaging_metadata_csv, metadata_csv_out=imaging_metadata_csv, remove_small_blobs=True, fill_holes=True, morphological_closing=True, image_filename=image_filename, tumor_mask_filename=tumor_mask_filename, combined_organ_mask_filename="segmentation_organs.nii.gz", mask_filename_out=tumor_and_organ_mask_filename, visualize=False)
metadata_df = pd.read_csv(imaging_metadata_csv)

for patient_id in tqdm(sorted(os.listdir(nifti_dir)), desc="Processing images"):
    patient_dir = os.path.join(nifti_dir, patient_id)
    mask_file = os.path.join(patient_dir, tumor_and_organ_mask_filename)
    # load in mask, set everything ≥1 to 1, and save back out
    mask = nib.load(mask_file)
    mask_data = mask.get_fdata()
    mask_data[mask_data >= 1] = 1
    new_mask = nib.Nifti1Image(mask_data, mask.affine, mask.header)
    nib.save(new_mask, os.path.join(patient_dir, tumor_and_organ_mask_binary_filename))

logger.info(f"Updating imaging metadata CSV at {imaging_metadata_csv} with paths to processed images and masks")
metadata_df["image"] = metadata_df["patient_id"].apply(lambda sid: os.path.join(nifti_dir, sid, image_filename))
metadata_df["tumor_mask"] = metadata_df["patient_id"].apply(lambda sid: os.path.join(nifti_dir, sid, tumor_mask_filename))
metadata_df["organ_mask"] = metadata_df["patient_id"].apply(lambda sid: os.path.join(nifti_dir, sid, tumor_and_organ_mask_binary_filename))
metadata_df["tumor_and_organ_mask"] = metadata_df["patient_id"].apply(lambda sid: os.path.join(nifti_dir, sid, tumor_and_organ_mask_filename))

# patient_ids_txt = os.path.join(data_dir, "patient_ids.txt")
# metadata_df["patient_id"].to_csv(patient_ids_txt, index=False, header=False)
# logger.info(f"Patient IDs saved to {patient_ids_txt}")

metadata_df.to_csv(imaging_metadata_csv, index=False)
logger.info(f"Updated imaging metadata CSV saved to {imaging_metadata_csv}")

# utils.prepare_csv_for_pyradiomics(nifti_dir, output_csv_path=pyradiomics_input_csv_path, imaging_file_name=image_filename, mask_file_name=mask_filename)  # image_filename_nii, mask_filename_nii
# utils.perform_radiomics_pipeline(pyradiomics_input_csv_path, output_csv_path=output_csv_path, label=[1,2], param=pyradiomics_param_file)
