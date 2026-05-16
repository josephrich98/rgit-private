# Radiogenomic Information-Theoretic Bounds

A statistical analysis framework for characterizing fundamental information-theoretic limits on the relationship between radiology imaging phenotypes and genomic data.

## Overview

This project investigates the theoretical upper and lower bounds on how much predictive information flows between:

- **Imaging phenotypes** — features derived from CT and MRI studies (radiomic descriptors, deep features, or structured reads)
- **Genomic alterations** — somatic mutations, copy number variants (CNV), and germline SNPs

Rather than optimizing a specific predictor, the goal is to quantify the *fundamental* limits imposed by information theory: how much mutual information exists between these modalities, what rate-distortion trade-offs apply when compressing one modality to predict the other, and where the predictive ceiling lies regardless of model choice.

## Research Questions

1. What is the mutual information between imaging-derived features and genomic alteration profiles?
2. What are the theoretical upper bounds on genomics → imaging (and imaging → genomics) prediction accuracy?
3. How do data quantity, feature dimensionality, and noise affect the achievable information transfer?

## Repository Structure

```
.
├── data/           # Raw and processed datasets (not committed)
├── notebooks/      # Exploratory analysis and figure generation
├── rgit/           # Core Python package
├── scripts/        # Standalone analysis scripts
├── main.tex        # Primary manuscript / technical report
└── pyproject.toml  # Python project configuration
```

## Methods

The analysis draws on:

- **Mutual information estimation** — non-parametric (KSG, MINE) and parametric estimators for continuous and mixed-type variables
- **Rate-distortion theory** — characterizing the minimum description length of one modality needed to predict the other at a given fidelity
- **Data processing inequality** — bounding information loss through feature extraction pipelines
- **Finite-sample corrections** — bias correction and bootstrap confidence intervals for MI estimates in high-dimensional settings

## Setup

```bash
pip install -e .[processing,notebooks,dev]
```

Python 3.10+ is recommended. Dependencies are declared in `pyproject.toml`.

## Usage

Analysis notebooks are in `notebooks/`. Reusable estimation utilities live in the `rgit/` package. Batch scripts for large-scale runs are in `scripts/`.

### Synthetic data
papermill notebooks/radiogenomic_recoverability.ipynb notebooks/out/radiogenomic_recoverability_output_synthetic.ipynb  # synthetic data

### Real data (TCGA-KIRC example)
#### Imaging data processing
python scripts/process_imaging.py -d data/tcga_kirc/imaging
python scripts/make_imaging_matrix.py -o data/tcga_kirc/imaging/whole_radiomics.h5ad -m data/tcga_kirc/imaging/metadata.csv --embedder pyradiomics
python scripts/make_imaging_matrix.py -o data/tcga_kirc/imaging/organ_radiomics.h5ad -m data/tcga_kirc/imaging/metadata.csv --mask_col organ_mask --embedder pyradiomics
python scripts/make_imaging_matrix.py -o data/tcga_kirc/imaging/tumor_radiomics.h5ad -m data/tcga_kirc/imaging/metadata.csv --mask_col tumor_mask --embedder pyradiomics --label 2
python scripts/make_imaging_matrix.py -o data/tcga_kirc/imaging/whole_radimagenet.h5ad -m data/tcga_kirc/imaging/metadata.csv --embedder radimagenet --clip_min -200 --clip_max 300 --resample_spacing 0.8,0.8,3.0 --apply_mask --crop_size 625,625,200
python scripts/make_imaging_matrix.py -o data/tcga_kirc/imaging/organ_radimagenet.h5ad -m data/tcga_kirc/imaging/metadata.csv --mask_col organ_mask --embedder radimagenet --clip_min -200 --clip_max 300 --resample_spacing 0.8,0.8,3.0 --apply_mask --crop_size 185,185,75
python scripts/make_imaging_matrix.py -o data/tcga_kirc/imaging/tumor_radimagenet.h5ad -m data/tcga_kirc/imaging/metadata.csv --mask_col tumor_mask --embedder radimagenet --clip_min -200 --clip_max 300 --resample_spacing 0.8,0.8,3.0 --apply_mask --crop_size 185,185,75 --label 2

#### Genomics data processing
wget -O data/tcga_kirc/genomics/mc3.v0.2.8.PUBLIC.maf.gz https://api.gdc.cancer.gov/data/1c8cfe5f-e52d-41ba-94da-f15ea1337efc
python scripts/make_genomics_matrix.py -o data/tcga_kirc/genomics/mutated_genes.h5ad data/tcga_kirc/genomics/mc3.v0.2.8.PUBLIC.maf.gz --feature gene_symbol --patient_ids data/tcga_kirc/imaging/metadata.csv
python scripts/make_genomics_matrix.py -o data/tcga_kirc/genomics/mutated_pathways.h5ad data/tcga_kirc/genomics/mc3.v0.2.8.PUBLIC.maf.gz --feature pathway --patient_ids data/tcga_kirc/imaging/metadata.csv

gdc-client download -m data/tcga_kirc/genomics/gene_expression_manifest.txt -d data/tcga_kirc/genomics
tar -xzvf data/tcga_kirc/genomics/gene_expression.tar.gz -C data/tcga_kirc/genomics/gene_expression
python scripts/make_genomics_matrix.py -o data/tcga_kirc/genomics/gene_expression.h5ad data/tcga_kirc/genomics/gene_expression --feature gene_expression --patient_ids data/tcga_kirc/imaging/metadata.csv --filename_to_patientid data/tcga_kirc/genomics/gene_expression/gene_expression_filename_to_patientid.csv --gene_expression_bins 2

#### Run notebooks
for genomics_h5ad in data/tcga_kirc/genomics/*.h5ad; do
    for imaging_h5ad in data/tcga_kirc/imaging/*_radimagenet.h5ad; do
        echo "Running recoverability analysis for genomics: $genomics_h5ad and imaging: $imaging_h5ad"
        output_notebook="notebooks/out/radiogenomic_recoverability_output_tcga_kirc_genomics_$(basename ${genomics_h5ad%.*})_imaging_$(basename ${imaging_h5ad%.*}).ipynb"
        papermill notebooks/radiogenomic_recoverability.ipynb "$output_notebook" -p GENOMICS_H5AD "$genomics_h5ad" -p IMAGING_H5AD "$imaging_h5ad"
    done
done

## Status

Early-stage research repository. Methods and structure are under active development.
