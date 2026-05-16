"""
Build a patient × genomic-feature AnnData from one or more MAF files or gene expression TSVs.

Feature modes (--feature):
  variant_id      — binary matrix: each column is a unique somatic variant
  gene_symbol     — binary matrix: each column is a mutated gene
  pathway         — binary matrix: each column is an altered pathway (KEGG_2016 by default)
  gene_expression — continuous matrix: each column is a gene (TPM by default)

Output obs index is patient_id. For MAF modes X is sparse binary; for gene_expression X is float32.
"""

import argparse
import logging
import os
import sys
import tarfile
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
# MAF loading
# ---------------------------------------------------------------------------

def load_maf(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MAF file not found: {path}")
    logger.info(f"Reading {path}...")
    if str(path).endswith(".gz"):
        df = pd.read_csv(path, sep="\t", comment="#", low_memory=False, compression="gzip")
    else:
        df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    logger.info(f"  → {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def annotate_maf(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2",
                "Tumor_Sample_Barcode", "Hugo_Symbol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"MAF is missing required columns: {missing}")

    df = df.copy()
    df["variant_id"] = (
        df["Chromosome"].astype(str)
        + ":"
        + df["Start_Position"].astype(str)
        + df["Reference_Allele"]
        + ">"
        + df["Tumor_Seq_Allele2"]
    )
    df["patient_id"] = df["Tumor_Sample_Barcode"].str.slice(0, 12)
    df["gene_symbol"] = df["Hugo_Symbol"]
    return df


def _get_pathway_library(name: str = "KEGG_2016") -> dict[str, set[str]]:
    import gseapy as gp
    lib = gp.get_library(name=name)
    return {k: set(map(str.upper, v)) for k, v in lib.items()}


def _load_gmt(path: str) -> dict[str, set[str]]:
    pathway_to_genes: dict[str, set[str]] = {}
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            pathway = parts[0]
            genes = set(g.upper() for g in parts[2:] if g)
            pathway_to_genes[pathway] = genes
    return pathway_to_genes


def build_pathway_matrix(
    maf_df: pd.DataFrame,
    pathway_library: str = "KEGG_2016",
    pathway_library_path: str | None = None,
    fraction_overlap_threshold: float = 0.1,
) -> pd.DataFrame:
    """Return binary patient × pathway DataFrame."""
    if pathway_library_path:
        pathway_to_genes = _load_gmt(pathway_library_path)
    else:
        pathway_to_genes = _get_pathway_library(pathway_library)

    # patient → set of mutated genes (uppercased)
    patient_to_genes: dict[str, set[str]] = (
        maf_df.groupby("patient_id")["gene_symbol"]
        .apply(lambda x: set(g.upper() for g in x if pd.notna(g)))
        .to_dict()
    )

    records = []
    for patient, genes in tqdm(patient_to_genes.items(), desc="Scoring pathways"):
        row: dict = {"patient_id": patient}
        for pathway, pathway_genes in pathway_to_genes.items():
            if not pathway_genes:
                continue
            frac = len(genes & pathway_genes) / len(pathway_genes)
            row[pathway] = int(frac >= fraction_overlap_threshold)
        records.append(row)

    pathway_df = pd.DataFrame(records).set_index("patient_id")
    # drop pathways never altered
    pathway_df = pathway_df.loc[:, pathway_df.sum() > 0]
    return pathway_df


# ---------------------------------------------------------------------------
# Binary pivot helpers
# ---------------------------------------------------------------------------

def build_binary_matrix(maf_df: pd.DataFrame, feature_col: str) -> pd.DataFrame:
    """Binary patient × feature pivot table."""
    sub = maf_df[["patient_id", feature_col]].dropna().drop_duplicates()
    sub["_present"] = 1
    mat = sub.pivot_table(index="patient_id", columns=feature_col, values="_present", fill_value=0)
    mat.columns.name = None
    return mat.astype(np.uint8)


# ---------------------------------------------------------------------------
# AnnData construction
# ---------------------------------------------------------------------------

def matrix_to_adata(
    mat: pd.DataFrame,
    obs_meta: pd.DataFrame | None = None,
    var_meta: pd.DataFrame | None = None,
) -> ad.AnnData:
    X = csr_matrix(mat.values.astype(np.float32))
    obs = obs_meta.loc[mat.index] if obs_meta is not None else pd.DataFrame(index=mat.index)
    var = var_meta.loc[mat.columns] if var_meta is not None else pd.DataFrame(index=mat.columns)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    return adata


# ---------------------------------------------------------------------------
# Gene expression loading
# ---------------------------------------------------------------------------

def load_filename_to_patientid(path: str) -> dict[str, str]:
    """Load CSV mapping file_name → patient_id (entity_submitter_id)."""
    df = pd.read_csv(path)
    if "file_name" not in df.columns:
        raise ValueError(f"filename_to_patientid CSV must have a 'file_name' column; found: {list(df.columns)}")
    id_col = "entity_submitter_id" if "entity_submitter_id" in df.columns else "patient_id"
    if id_col not in df.columns:
        raise ValueError(f"filename_to_patientid CSV must have 'entity_submitter_id' or 'patient_id' column; found: {list(df.columns)}")
    return dict(zip(df["file_name"], df[id_col]))


def _read_expression_tsv(fileobj, value_col: str) -> pd.Series:
    """Return a Series gene_id → expression value, skipping summary rows."""
    df = pd.read_csv(fileobj, sep="\t", comment="#")
    df = df[df["gene_name"].notna()]  # drop N_unmapped etc.
    df = df.set_index("gene_id")[value_col]
    return df


def build_gene_expression_matrix(
    inputs: list[str],
    filename_to_patientid: dict[str, str],
    value_col: str = "tpm_unstranded",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a patient × gene expression DataFrame from tar.gz or directory inputs.

    Returns (expr_df, var_meta_df) where expr_df has patient_id as index and
    gene_id as columns, and var_meta_df has gene_name indexed by gene_id.
    """
    series_list: list[pd.Series] = []
    patient_ids: list[str] = []
    var_meta: dict[str, str] = {}  # gene_id → gene_name

    for inp in inputs:
        p = Path(inp)
        if tarfile.is_tarfile(inp):
            with tarfile.open(inp, "r:gz") as tf:
                members = [m for m in tf.getmembers() if m.name.endswith(".tsv")]
                for member in tqdm(members, desc=f"Reading {p.name}"):
                    fname = Path(member.name).name
                    patient_id = filename_to_patientid.get(fname)
                    if patient_id is None:
                        logger.warning(f"No patient_id mapping for {fname}, skipping")
                        continue
                    fobj = tf.extractfile(member)
                    s = _read_expression_tsv(fobj, value_col)
                    series_list.append(s)
                    patient_ids.append(patient_id)
        elif p.is_dir():
            tsv_files = sorted(p.rglob("*.tsv"))
            for tsv_path in tqdm(tsv_files, desc=f"Reading {p.name}"):
                fname = tsv_path.name
                patient_id = filename_to_patientid.get(fname)
                if patient_id is None:
                    logger.warning(f"No patient_id mapping for {fname}, skipping")
                    continue
                with open(tsv_path) as fobj:
                    s = _read_expression_tsv(fobj, value_col)
                series_list.append(s)
                patient_ids.append(patient_id)
        else:
            raise ValueError(f"gene_expression input must be a .tar.gz or a directory: {inp}")

    if not series_list:
        raise RuntimeError("No gene expression files were loaded.")

    # build var metadata (gene_name) from first sample
    inp0 = inputs[0]
    if tarfile.is_tarfile(inp0):
        with tarfile.open(inp0, "r:gz") as tf:
            first = next(m for m in tf.getmembers() if m.name.endswith(".tsv"))
            fobj = tf.extractfile(first)
            tmp = pd.read_csv(fobj, sep="\t", comment="#")
            tmp = tmp[tmp["gene_name"].notna()].set_index("gene_id")
            for gid, row in tmp.iterrows():
                var_meta[gid] = row["gene_name"]
    else:
        first_tsv = next(Path(inp0).rglob("*.tsv"))
        tmp = pd.read_csv(first_tsv, sep="\t", comment="#")
        tmp = tmp[tmp["gene_name"].notna()].set_index("gene_id")
        for gid, row in tmp.iterrows():
            var_meta[gid] = row["gene_name"]

    expr_df = pd.DataFrame(series_list, index=patient_ids)
    expr_df.index.name = "patient_id"

    var_meta_df = pd.DataFrame({"gene_name": var_meta}, index=pd.Index(list(var_meta.keys()), name="gene_id"))
    var_meta_df = var_meta_df.reindex(expr_df.columns)

    logger.info(f"Gene expression matrix: {expr_df.shape[0]} patients × {expr_df.shape[1]} genes")
    return expr_df, var_meta_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Build patient × genomic-feature AnnData from MAF file(s)")
    parser.add_argument("inputs", nargs="+", help="One or more MAF files (.maf or .maf.gz) or directories with structure dir/filename/file.ext")
    parser.add_argument("-o", "--out", required=True, help="Output .h5ad path")
    parser.add_argument(
        "--feature", choices=["variant_id", "gene_symbol", "pathway", "gene_expression"], default="gene_symbol",
        help="Feature granularity for the var axis"
    )
    parser.add_argument(
        "--expression_value", default="tpm_unstranded",
        choices=["tpm_unstranded", "fpkm_unstranded", "fpkm_uq_unstranded", "unstranded"],
        help="[gene_expression mode] Which count/expression column to use (default: tpm_unstranded)"
    )
    parser.add_argument(
        "--gene_expression_bins", type=int, default=None, metavar="N",
        help="[gene_expression mode] Bin each gene's expression into N uniform quantile buckets (e.g. 2=halves, 3=thirds). Default: no binning."
    )
    parser.add_argument("--patient_ids", default=None, help="CSV/parquet with column 'patient_id' to restrict output")
    parser.add_argument("--filename_to_patientid", default=None, help="Path to CSV if input files are organized as dir/filename/file.ext and patient_id is derived from filename (one column 'filename' with relative path from input dir, one column 'patient_id')")
    parser.add_argument(
        "--pathway_library", default="KEGG_2016",
        help="[pathway mode] gseapy library name (default: KEGG_2016)"
    )
    parser.add_argument(
        "--pathway_library_path", default=None,
        help="[pathway mode] Path to a GMT file instead of downloading from gseapy"
    )
    parser.add_argument(
        "--pathway_threshold", type=float, default=0.1,
        help="[pathway mode] Fraction of pathway genes mutated to call pathway 'altered' (default: 0.1)"
    )
    parser.add_argument(
        "--extra_obs_cols", nargs="*", default=[],
        help="Extra MAF columns to pull into adata.obs (one value per patient, taken from first row)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Gene expression branch — entirely separate from MAF pipeline
    # -----------------------------------------------------------------------
    if args.feature == "gene_expression":
        if not args.filename_to_patientid:
            raise ValueError("--filename_to_patientid is required for --feature gene_expression")

        filename_to_patientid = load_filename_to_patientid(args.filename_to_patientid)
        logger.info(f"Loaded {len(filename_to_patientid)} filename→patient_id mappings")

        mat, var_meta = build_gene_expression_matrix(
            args.inputs,
            filename_to_patientid=filename_to_patientid,
            value_col=args.expression_value,
        )

        # optional patient filter
        if args.patient_ids:
            filter_path = Path(args.patient_ids)
            if filter_path.suffix == ".parquet":
                pid_df = pd.read_parquet(filter_path)
            else:
                pid_df = pd.read_csv(filter_path, sep="\t" if filter_path.suffix == ".tsv" else ",")
            keep = set(pid_df["patient_id"].astype(str))
            mat = mat[mat.index.isin(keep)]
            logger.info(f"After patient filter: {mat.shape[0]} patients")

        if len(mat) == 0:
            raise RuntimeError("No patients remain after filtering.")

        if args.gene_expression_bins is not None:
            n = args.gene_expression_bins
            if n < 2:
                raise ValueError("--gene_expression_bins must be >= 2")
            logger.info(f"Binning expression into {n} quantile buckets per gene (0-indexed)...")
            mat = mat.apply(
                lambda col: pd.qcut(col, q=n, labels=False, duplicates="drop"),
                axis=0,
            )

        obs_meta = pd.DataFrame(index=mat.index)
        adata = matrix_to_adata(mat, obs_meta=obs_meta, var_meta=var_meta)
        adata.uns["feature_type"] = "gene_expression"
        adata.uns["expression_value"] = args.expression_value
        if args.gene_expression_bins is not None:
            adata.uns["gene_expression_bins"] = args.gene_expression_bins
        adata.uns["inputs"] = args.inputs
        adata.write_h5ad(out_path)
        logger.info(f"Saved gene expression AnnData: {adata.shape} → {out_path}")
        return

    # -----------------------------------------------------------------------
    # MAF-based branches (variant_id, gene_symbol, pathway)
    # -----------------------------------------------------------------------

    # --- load & concatenate MAFs ---
    frames = []
    for path in args.inputs:
        df = load_maf(path)
        df = annotate_maf(df)
        frames.append(df)

    maf_df = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined MAF: {len(maf_df):,} rows, {maf_df['patient_id'].nunique()} patients")

    # --- optional patient filter ---
    if args.patient_ids:
        filter_path = Path(args.patient_ids)
        if filter_path.suffix == ".parquet":
            pid_df = pd.read_parquet(filter_path)
        elif filter_path.suffix in {".csv", ".tsv"}:
            pid_df = pd.read_csv(filter_path, sep="\t" if filter_path.suffix == ".tsv" else ",")
        elif filter_path.suffix == ".txt":
            pid_df = pd.read_csv(filter_path, header=None, names=["patient_id"])
        else:
            raise ValueError(f"Unsupported patient_ids file format: {filter_path.suffix}")

        if "patient_id" not in pid_df.columns:
            raise ValueError(f"patient_ids file must have a 'patient_id' column")
        keep = set(pid_df["patient_id"].astype(str))
        maf_df = maf_df[maf_df["patient_id"].isin(keep)]
        logger.info(f"After patient filter: {len(maf_df):,} rows, {maf_df['patient_id'].nunique()} patients")

    if len(maf_df) == 0:
        raise RuntimeError("No rows remain after filtering.")

    # --- per-patient metadata for obs ---
    obs_meta_cols = ["patient_id"] + [c for c in args.extra_obs_cols if c in maf_df.columns]
    obs_meta = (
        maf_df[obs_meta_cols]
        .drop_duplicates(subset=["patient_id"])
        .set_index("patient_id")
    )

    # --- build feature matrix ---
    if args.feature == "variant_id":
        logger.info("Building variant_id binary matrix...")
        mat = build_binary_matrix(maf_df, "variant_id")
        var_meta = pd.DataFrame(index=mat.columns)

    elif args.feature == "gene_symbol":
        logger.info("Building gene_symbol binary matrix...")
        mat = build_binary_matrix(maf_df, "gene_symbol")
        var_meta = pd.DataFrame(index=mat.columns)

    elif args.feature == "pathway":
        logger.info(f"Building pathway binary matrix (library={args.pathway_library}, threshold={args.pathway_threshold})...")
        mat = build_pathway_matrix(
            maf_df,
            pathway_library=args.pathway_library,
            pathway_library_path=args.pathway_library_path,
            fraction_overlap_threshold=args.pathway_threshold,
        )
        var_meta = pd.DataFrame(index=mat.columns)

    # align obs_meta to mat index
    obs_meta = obs_meta.reindex(mat.index)

    logger.info(f"Feature matrix: {mat.shape[0]} patients × {mat.shape[1]} {args.feature}s")

    adata = matrix_to_adata(mat, obs_meta=obs_meta, var_meta=var_meta)
    adata.uns["feature_type"] = args.feature
    adata.uns["maf_inputs"] = args.inputs

    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    logger.info(f"Saved genomics AnnData: {adata.shape} → {out_path}")


if __name__ == "__main__":
    main()
