#!/usr/bin/env python
"""Build a patient x GO-pathway score matrix from ADNI microarray expression.

Input : data/adni/genomics/gene_expression.h5ad   (patients x genes, log2 microarray)
Output: data/adni/genomics/gene_expression_go.h5ad (patients x GO pathways)

Going from a patient x gene matrix to a patient x pathway matrix requires a
*per-sample* enrichment method, not a classical two-group differential-expression
+ GO-overrepresentation test (which collapses a contrast into a single ranked
gene list and yields no per-patient values). The canonical per-sample method is
single-sample GSEA (ssGSEA): for each patient it ranks all genes by expression
and scores how concentrated each GO gene set is toward the top/bottom of that
ranking. The resulting normalized enrichment score (NES) is the per-patient
"pathway activity" used downstream.

Run standalone from the repo root:

    python scripts/make_adni_go_matrix.py
    python scripts/make_adni_go_matrix.py --go-sets BP CC MF --min-size 10
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import anndata as ad
import gseapy as gp
import numpy as np
import pandas as pd

# GO Enrichr library names (Biological Process / Cellular Component / Molecular Function)
GO_LIBRARIES = {
    "BP": "GO_Biological_Process_2025",
    "CC": "GO_Cellular_Component_2025",
    "MF": "GO_Molecular_Function_2025",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_expression(h5ad_path: Path) -> ad.AnnData:
    a = ad.read_h5ad(h5ad_path)
    X = a.X if isinstance(a.X, np.ndarray) else a.X.toarray()
    if np.isnan(X).any():
        raise ValueError("Expression matrix contains NaNs; clean before scoring.")
    a.X = np.asarray(X, dtype=np.float64)
    return a


def expression_dataframe(a: ad.AnnData) -> pd.DataFrame:
    """Return a genes x samples DataFrame, collapsing duplicate symbols by mean."""
    df = pd.DataFrame(a.X.T, index=a.var_names.astype(str), columns=a.obs_names.astype(str))
    df.index.name = "gene"
    if df.index.has_duplicates:
        n_dup = int(df.index.duplicated().sum())
        print(f"[info] collapsing {n_dup} duplicate gene symbols by mean")
        df = df.groupby(level=0).mean()
    return df


def load_go_gene_sets(categories: list[str]) -> dict[str, list[str]]:
    """Fetch and merge requested GO gene-set libraries from Enrichr."""
    gene_sets: dict[str, list[str]] = {}
    for cat in categories:
        lib = GO_LIBRARIES[cat]
        print(f"[info] fetching {lib} ...")
        sets = gp.get_library(name=lib, organism="Human")
        # tag the source category so pathways from different GO branches stay distinct
        for term, genes in sets.items():
            gene_sets[f"{cat}::{term}"] = genes
        print(f"[info]   {len(sets)} terms")
    return gene_sets


def run_ssgsea(
    expr: pd.DataFrame,
    gene_sets: dict[str, list[str]],
    min_size: int,
    max_size: int,
    threads: int,
    seed: int,
) -> pd.DataFrame:
    """Run ssGSEA; return a samples x pathways DataFrame of normalized enrichment scores."""
    print(
        f"[info] ssGSEA: {expr.shape[0]} genes x {expr.shape[1]} samples "
        f"vs {len(gene_sets)} gene sets (threads={threads})"
    )
    res = gp.ssgsea(
        data=expr,
        gene_sets=gene_sets,
        outdir=None,
        sample_norm_method="rank",
        min_size=min_size,
        max_size=max_size,
        threads=threads,
        seed=seed,
        no_plot=True,
        verbose=True,
    )
    long = res.res2d.copy()
    long["NES"] = pd.to_numeric(long["NES"], errors="coerce")
    mat = long.pivot(index="Name", columns="Term", values="NES")
    # restore original sample order; drop any sample/pathway that failed entirely
    mat = mat.reindex(index=[s for s in expr.columns if s in mat.index])
    mat = mat.dropna(axis=1, how="all")
    print(f"[info] score matrix: {mat.shape[0]} samples x {mat.shape[1]} pathways")
    return mat


def build_anndata(scores: pd.DataFrame, source: ad.AnnData, gene_sets: dict) -> ad.AnnData:
    obs = source.obs.reindex(scores.index).copy()
    var = pd.DataFrame(index=scores.columns)
    var["category"] = [t.split("::", 1)[0] for t in var.index]
    var["go_term"] = [t.split("::", 1)[1] for t in var.index]
    var["n_genes"] = [len(gene_sets.get(t, [])) for t in var.index]
    out = ad.AnnData(X=scores.to_numpy(dtype=np.float64), obs=obs, var=var)
    out.uns["pathway_scoring"] = {
        "method": "ssGSEA (gseapy)",
        "score": "normalized enrichment score (NES)",
        "source_h5ad": "gene_expression.h5ad",
        "go_libraries": list({c: GO_LIBRARIES[c] for c in var["category"].unique()}.values()),
    }
    return out


def main() -> None:
    root = repo_root()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(root / "data/adni/genomics/gene_expression.h5ad"))
    p.add_argument("--output", default=str(root / "data/adni/genomics/gene_expression_go.h5ad"))
    p.add_argument("--go-sets", nargs="+", default=["BP"], choices=list(GO_LIBRARIES))
    p.add_argument("--min-size", type=int, default=15, help="min genes per GO set")
    p.add_argument("--max-size", type=int, default=500, help="max genes per GO set")
    p.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 4))
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    a = load_expression(Path(args.input))
    print(f"[info] loaded {a.shape[0]} patients x {a.shape[1]} genes from {args.input}")
    expr = expression_dataframe(a)
    gene_sets = load_go_gene_sets(args.go_sets)
    scores = run_ssgsea(expr, gene_sets, args.min_size, args.max_size, args.threads, args.seed)
    out = build_anndata(scores, a, gene_sets)
    out.write_h5ad(args.output)
    print(f"[done] wrote {out.shape[0]} patients x {out.shape[1]} pathways -> {args.output}")


if __name__ == "__main__":
    main()
