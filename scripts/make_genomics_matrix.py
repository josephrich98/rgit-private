"""
Build a patient × genomic-feature AnnData from one or more MAF files,
gene expression TSVs, or an ADNI microarray CSV.

--dataset choices:
  tcga   — MAF files (.maf/.maf.gz) or gene-expression tar.gz / directories
  nsclc  — single gene-expression .txt / .txt.gz matrix (genes × samples)
  adni   — ADNI_Gene_Expression_Profile.csv (single input)

Feature modes for --dataset tcga (--feature):
  variant_id      — binary matrix: each column is a unique somatic variant
  gene_symbol     — binary matrix: each column is a mutated gene
  pathway         — binary matrix: each column is an altered pathway (KEGG_2016 by default)
  gene_expression — continuous matrix: each column is a gene (TPM by default)
"""

import argparse
import logging
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TCGA helpers
# ---------------------------------------------------------------------------

def load_maf(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MAF file not found: {path}")
    logger.info(f"Reading {path}...")
    compression = "gzip" if str(path).endswith(".gz") else None
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False, compression=compression)
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
    pathway_df = pathway_df.loc[:, pathway_df.sum() > 0]
    return pathway_df


def build_binary_matrix(maf_df: pd.DataFrame, feature_col: str) -> pd.DataFrame:
    """Binary patient × feature pivot table."""
    sub = maf_df[["patient_id", feature_col]].dropna().drop_duplicates().copy()
    sub["_present"] = 1
    mat = sub.pivot_table(index="patient_id", columns=feature_col, values="_present", fill_value=0)
    mat.columns.name = None
    return mat.astype(np.uint8)


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


def load_filename_to_patientid(path: str) -> dict[str, str]:
    """Load CSV mapping file_name → patient_id (entity_submitter_id)."""
    df = pd.read_csv(path)
    if "file_name" not in df.columns:
        raise ValueError(f"filename_to_patientid CSV must have a 'file_name' column; found: {list(df.columns)}")
    id_col = "entity_submitter_id" if "entity_submitter_id" in df.columns else "patient_id"
    if id_col not in df.columns:
        raise ValueError(f"filename_to_patientid CSV must have 'entity_submitter_id' or 'patient_id' column; found: {list(df.columns)}")
    return dict(zip(df["file_name"], df[id_col]))


_EXPRESSION_COLS = ["tpm_unstranded", "fpkm_unstranded", "fpkm_uq_unstranded", "unstranded"]


def _read_expression_tsv(fileobj) -> tuple[pd.DataFrame, pd.Series]:
    """Return (expr_df, gene_name_series) indexed by gene_id, skipping summary rows."""
    df = pd.read_csv(fileobj, sep="\t", comment="#")
    df = df[df["gene_name"].notna()]
    df = df.set_index("gene_id")
    return df[_EXPRESSION_COLS], df["gene_name"]


def build_gene_expression_matrix(
    inputs: list[str],
    filename_to_patientid: dict[str, str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Build patient × gene expression DataFrames from tar.gz or directory inputs.

    Returns (layers, var_meta_df) where layers is a dict keyed by expression
    column name and var_meta_df has gene_name indexed by gene_id.
    """
    frames: list[pd.DataFrame] = []
    patient_ids: list[str] = []
    gene_names: pd.Series | None = None

    for inp in inputs:
        p = Path(inp)
        if p.is_file() and p.name.endswith((".tar.gz", ".tgz", ".tar")):
            with tarfile.open(inp) as tf:
                members = [m for m in tf.getmembers() if m.name.endswith(".tsv")]
                for member in tqdm(members, desc=f"Reading {p.name}"):
                    fname = Path(member.name).name
                    patient_id = filename_to_patientid.get(fname)
                    if patient_id is None:
                        logger.warning(f"No patient_id mapping for {fname}, skipping")
                        continue
                    expr, gnames = _read_expression_tsv(tf.extractfile(member))
                    frames.append(expr)
                    patient_ids.append(patient_id)
                    if gene_names is None:
                        gene_names = gnames
        elif p.is_dir():
            for tsv_path in tqdm(sorted(p.rglob("*.tsv")), desc=f"Reading {p.name}"):
                fname = tsv_path.name
                patient_id = filename_to_patientid.get(fname)
                if patient_id is None:
                    logger.warning(f"No patient_id mapping for {fname}, skipping")
                    continue
                with open(tsv_path) as fobj:
                    expr, gnames = _read_expression_tsv(fobj)
                frames.append(expr)
                patient_ids.append(patient_id)
                if gene_names is None:
                    gene_names = gnames
        else:
            raise ValueError(f"gene_expression input must be a .tar.gz or a directory: {inp}")

    if not frames:
        raise RuntimeError("No gene expression files were loaded.")

    gene_ids = frames[0].index
    idx = pd.Index(patient_ids, name="patient_id")
    layers = {
        col: pd.DataFrame(
            [f[col].values for f in frames],
            index=idx,
            columns=gene_ids,
        )
        for col in _EXPRESSION_COLS
    }
    var_meta_df = pd.DataFrame({"gene_name": gene_names}, index=gene_ids)
    logger.info(f"Gene expression: {len(patient_ids)} patients × {len(gene_ids)} genes, {len(_EXPRESSION_COLS)} layers")
    return layers, var_meta_df


def _filter_patients(maf_or_layer, patient_ids_path: str):
    filter_path = Path(patient_ids_path)
    if filter_path.suffix == ".parquet":
        pid_df = pd.read_parquet(filter_path)
    elif filter_path.suffix in {".csv", ".tsv"}:
        pid_df = pd.read_csv(filter_path, sep="\t" if filter_path.suffix == ".tsv" else ",")
    elif filter_path.suffix == ".txt":
        pid_df = pd.read_csv(filter_path, header=None, names=["patient_id"])
    else:
        raise ValueError(f"Unsupported patient_ids file format: {filter_path.suffix}")
    if "patient_id" not in pid_df.columns:
        raise ValueError("patient_ids file must have a 'patient_id' column")
    return set(pid_df["patient_id"].astype(str))


def build_tcga(args) -> ad.AnnData:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # gene expression branch (tar.gz or directory inputs)
    if args.feature == "gene_expression":
        if not args.filename_to_patientid:
            raise ValueError("--filename_to_patientid is required for --feature gene_expression")
        filename_to_patientid = load_filename_to_patientid(args.filename_to_patientid)
        logger.info(f"Loaded {len(filename_to_patientid)} filename→patient_id mappings")

        layers, var_meta = build_gene_expression_matrix(args.inputs, filename_to_patientid)

        if args.patient_ids:
            keep = _filter_patients(None, args.patient_ids)
            layers = {k: v[v.index.isin(keep)] for k, v in layers.items()}
            logger.info(f"After patient filter: {next(iter(layers.values())).shape[0]} patients")

        tpm = layers["tpm_unstranded"]
        if len(tpm) == 0:
            raise RuntimeError("No patients remain after filtering.")

        adata = matrix_to_adata(tpm, obs_meta=pd.DataFrame(index=tpm.index), var_meta=var_meta)
        for col, mat in layers.items():
            adata.layers[col] = csr_matrix(mat.values.astype(np.float32))
        adata.uns["feature_type"] = "gene_expression"
        adata.uns["layers"] = _EXPRESSION_COLS
        adata.uns["inputs"] = args.inputs
        adata.write_h5ad(out_path)
        logger.info(f"Saved gene expression AnnData: {adata.shape} → {out_path}")
        return adata

    # MAF branch (variant_id, gene_symbol, pathway)
    frames = []
    for path in args.inputs:
        df = load_maf(path)
        df = annotate_maf(df)
        frames.append(df)

    maf_df = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined MAF: {len(maf_df):,} rows, {maf_df['patient_id'].nunique()} patients")

    if args.patient_ids:
        keep = _filter_patients(maf_df, args.patient_ids)
        maf_df = maf_df[maf_df["patient_id"].isin(keep)]
        logger.info(f"After patient filter: {len(maf_df):,} rows, {maf_df['patient_id'].nunique()} patients")

    if len(maf_df) == 0:
        raise RuntimeError("No rows remain after filtering.")

    obs_meta_cols = ["patient_id"] + [c for c in args.extra_obs_cols if c in maf_df.columns]
    obs_meta = (
        maf_df[obs_meta_cols]
        .drop_duplicates(subset=["patient_id"])
        .set_index("patient_id")
    )

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

    obs_meta = obs_meta.reindex(mat.index)
    logger.info(f"Feature matrix: {mat.shape[0]} patients × {mat.shape[1]} {args.feature}s")

    adata = matrix_to_adata(mat, obs_meta=obs_meta, var_meta=var_meta)
    adata.uns["feature_type"] = args.feature
    adata.uns["maf_inputs"] = args.inputs
    adata.write_h5ad(out_path)
    logger.info(f"Saved genomics AnnData: {adata.shape} → {out_path}")
    return adata


# ---------------------------------------------------------------------------
# NSCLC helpers
# ---------------------------------------------------------------------------

def build_nsclc(args) -> ad.AnnData:
    path = args.inputs[0]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    compression = "gzip" if path.endswith(".gz") else None
    df = pd.read_csv(path, sep="\t", index_col=0, compression=compression)

    adata = ad.AnnData(
        X=df.T.values,
        obs=pd.DataFrame(index=df.columns),
        var=pd.DataFrame(index=df.index),
    )
    adata.uns["feature_type"] = "gene_expression"
    adata.uns["inputs"] = args.inputs
    adata.write_h5ad(out_path)
    logger.info(f"Saved gene expression AnnData: {adata.shape} → {out_path}")
    return adata


# ---------------------------------------------------------------------------
# ADNI helpers
# ---------------------------------------------------------------------------

def build_adni(args, index_symbol: bool = True) -> ad.AnnData:
    path = args.inputs[0]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(path, header=None, low_memory=False)
    header_idx = raw.index[raw[0] == "ProbeSet"][0]

    # sample metadata lives in rows above the expression header
    metadata = raw.iloc[:header_idx].copy()
    metadata.index = metadata[0]

    sample_metadata_full = metadata.iloc[:, 3:].T.reset_index(drop=True)
    sample_metadata_full.columns = metadata[0]

    # track which positions survive the SubjectID filter
    valid_mask = sample_metadata_full["SubjectID"].notna()
    valid_positions = valid_mask[valid_mask].index.tolist()

    sample_metadata = sample_metadata_full[valid_mask].reset_index(drop=True)

    # expression matrix: probes × samples
    expr = pd.read_csv(path, skiprows=header_idx)

    non_sample_cols = [
        c for c in [
            "ProbeSet",
            "LocusLink",
            "Symbol",
            "GeneSymbol",
            "GeneTitle",
            "Chromosome",
            "Cytoband",
        ]
        if c in expr.columns
    ]

    sample_cols = [c for c in expr.columns if c not in non_sample_cols]
    sample_cols_valid = [sample_cols[i] for i in valid_positions]

    X = expr[sample_cols_valid].apply(pd.to_numeric, errors="coerce").T

    obs = sample_metadata.copy()
    obs.index = sample_cols_valid

    symbol_col = "Symbol" if "Symbol" in expr.columns else "GeneSymbol"

    var = pd.DataFrame(
        {
            "symbol": expr[symbol_col].astype(str).values,
        },
        index=expr["ProbeSet"].astype(str),
    )

    var.index.name = "gene_id"
    X.columns = var.index

    # -----------------------------------------
    # OPTIONAL: collapse probes -> gene symbols
    # -----------------------------------------
    if index_symbol:

        # remove bad symbols
        valid_symbol_mask = (
            var["symbol"].notna()
            & (var["symbol"] != "")
            & (var["symbol"] != "---")
            & (var["symbol"] != "nan")
        )

        var = var.loc[valid_symbol_mask]
        X = X.loc[:, valid_symbol_mask.values]

        # suffix extraction
        def get_suffix(probe):
            parts = probe.split("_", 1)
            return parts[1] if len(parts) > 1 else ""

        # priority map
        probe_priority = {
            "at": 0,
            "a_at": 1,
            "s_at": 2,
            "x_at": 3,
            "PM_at": 4,
            "PM_s_at": 5,
            "PM_x_at": 6,
            "3_at": 7,
            "5_at": 8,
            "M_at": 9,
            "MA_at": 10,
            "MB_at": 11,
            "alu_at": 12,
        }

        suffixes = [get_suffix(p) for p in var.index]

        priorities = [
            probe_priority.get(s, 999)
            for s in suffixes
        ]

        variances = X.var(axis=0).values

        probe_info = pd.DataFrame(
            {
                "probe": var.index,
                "symbol": var["symbol"].values,
                "suffix": suffixes,
                "priority": priorities,
                "variance": variances,
            }
        )

        # best = lowest priority, then highest variance
        probe_info = probe_info.sort_values(
            ["symbol", "priority", "variance"],
            ascending=[True, True, False],
        )

        best = probe_info.drop_duplicates(
            subset="symbol",
            keep="first",
        )

        keep_probes = best["probe"].tolist()

        X = X[keep_probes]
        var = var.loc[keep_probes]

        # reindex by gene symbol
        var.index = var["symbol"]
        var.index.name = "gene_symbol"

        X.columns = var.index

    else:
        X.columns = var.index

    adata = ad.AnnData(
        X=X.values,
        obs=obs,
        var=var,
    )

    adata.uns["feature_type"] = "gene_expression"
    adata.uns["inputs"] = args.inputs
    adata.uns["index_symbol"] = index_symbol

    adata.write_h5ad(out_path)

    logger.info(
        f"Saved ADNI AnnData: {adata.shape} → {out_path}"
    )

    return adata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Build patient × genomic-feature AnnData")
    parser.add_argument(
        "--dataset", required=True, choices=["tcga", "nsclc", "adni"],
        help="Dataset type, controls which loading pipeline is used",
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="Input path(s): MAF file(s) or tar.gz/directory (tcga), .txt/.txt.gz (nsclc), CSV (adni)",
    )
    parser.add_argument("-o", "--out", required=True, help="Output .h5ad path")
    parser.add_argument(
        "--feature", choices=["variant_id", "gene_symbol", "pathway", "gene_expression"],
        default="gene_symbol",
        help="[tcga] Feature granularity for the var axis",
    )
    parser.add_argument("--patient_ids", default=None, help="CSV/parquet with column 'patient_id' to restrict output")
    parser.add_argument(
        "--filename_to_patientid", default=None,
        help="[tcga gene_expression] CSV mapping file_name → patient_id",
    )
    parser.add_argument(
        "--pathway_library", default="KEGG_2016",
        help="[tcga pathway] gseapy library name (default: KEGG_2016)",
    )
    parser.add_argument(
        "--pathway_library_path", default=None,
        help="[tcga pathway] Path to a GMT file instead of downloading from gseapy",
    )
    parser.add_argument(
        "--pathway_threshold", type=float, default=0.1,
        help="[tcga pathway] Fraction of pathway genes mutated to call pathway 'altered' (default: 0.1)",
    )
    parser.add_argument(
        "--extra_obs_cols", nargs="*", default=[],
        help="[tcga maf] Extra MAF columns to pull into adata.obs",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset == "tcga":
        build_tcga(args)
    elif args.dataset == "nsclc":
        if len(args.inputs) != 1 or not Path(args.inputs[0]).is_file() or not args.inputs[0].endswith((".txt", ".txt.gz")):
            raise ValueError("--dataset nsclc expects a single .txt or .txt.gz input file")
        build_nsclc(args)
    elif args.dataset == "adni":
        if len(args.inputs) != 1:
            raise ValueError("--dataset adni expects a single input CSV path")
        build_adni(args)


if __name__ == "__main__":
    main()
