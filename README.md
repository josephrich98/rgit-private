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
pip install -e .
```

Python 3.10+ is recommended. Dependencies are declared in `pyproject.toml`.

## Usage

Analysis notebooks are in `notebooks/`. Reusable estimation utilities live in the `rgit/` package. Batch scripts for large-scale runs are in `scripts/`.

## Status

Early-stage research repository. Methods and structure are under active development.
