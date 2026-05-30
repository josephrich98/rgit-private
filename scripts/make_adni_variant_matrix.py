"""
Build a MAF and/or a patient × variant matrix from VCFs or PLINK genotypes.

--inputs accepts any mix of:
  - a single VCF, multiple VCFs, or a folder of VCFs (.vcf/.vcf.gz)
  - a PLINK binary fileset (.bed/.bim/.fam), or a folder of them

PLINK filesets are converted to a GATK-ready VCF with `plink --recode vcf`, then
every VCF is annotated with GATK Funcotator. The resulting MAFs are concatenated
and can be written out as:
  --out-maf      a single gzipped MAF (variant-level annotation)
  --out          a patient × variant AnnData (.h5ad) of additive genotype dosage
                 (0/1/2), with adata.var carrying the MAF annotation columns

Funcotator emits variant-level annotation only, so the per-patient dosage in
--out is read from the genotypes (PLINK via `--extract --recode A`, or the VCF
GT fields), keyed to the annotation by variant. --filter-impact keeps only
impactful variants (indels + medium/high impact); --groupby-gene sums dosage
over each gene's variants instead of keeping per-variant columns. --apoe merges
an ADNI APOE table (APOERES.csv) into --out, since APOE's defining SNPs are not
on the Omni2.5M array (adds APOE_e4/APOE_e2 dosage + obs['APOE_genotype']).

Examples
--------
  # PLINK genotypes → impactful, gene-level variant matrix
  python scripts/make_adni_variant_matrix.py \\
      --inputs /home/jrich/data/adni/plink \\
      --reference /home/jrich/data/reference/gatk_grch37/Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz \\
      --data-sources /home/jrich/data/reference/gatk_grch37/funcotator_dataSources.v1.8.hg37.20230908g \\
      --ref-version hg19 \\
      --out /home/jrich/data/adni/adni.variant_matrix.h5ad \\
      --filter-impact --groupby-gene

  # VCFs → concatenated gzipped MAF
  python scripts/make_adni_variant_matrix.py \\
      --inputs /home/jrich/data/adni/vcfs \\
      --reference .../Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz \\
      --out-maf /home/jrich/data/adni/adni.funcotated.maf.gz
"""

import argparse
import glob
import gzip
import logging
import os
from pathlib import Path
import subprocess

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Funcotator
# ---------------------------------------------------------------------------

def download_funcotator_data_sources(data_sources_path, somatic=True, ref_version="hg19"):
    os.makedirs(os.path.dirname(data_sources_path), exist_ok=True)
    somatic_symbol = "s" if somatic else "g"
    url = (
        f"ftp://gsapubftp-anonymous@ftp.broadinstitute.org/bundle/funcotator/funcotator_dataSources.v1.8.{ref_version}.20230908{somatic_symbol}.tar.gz"
    )
    subprocess.run(f"wget {url} -O {data_sources_path}.tar.gz", shell=True, check=True)
    subprocess.run(
        f"tar -xzf {data_sources_path}.tar.gz -C {Path(data_sources_path).parent}",
        shell=True, check=True,
    )


def run_funcotator(input_vcf, reference, data_sources=None, ref_version="hg19",
                   out_prefix=None, gzip_output=True, somatic=True):
    if not os.path.exists(reference):
        raise FileNotFoundError(f"Reference genome file not found: {reference}")
    if ref_version == "grch37":
        ref_version = "hg19"
    elif ref_version == "grch38":
        ref_version = "hg38"
    if ref_version not in ["hg19", "hg38"]:
        raise ValueError(
            f"Unsupported reference version: {ref_version}. Supported versions are hg19 and hg38."
        )

    somatic_symbol = "s" if somatic else "g"
    if data_sources is None:
        data_sources = f"funcotator_dataSources.v1.8.{ref_version}.20230908{somatic_symbol}"
    if not os.path.exists(data_sources):
        download_funcotator_data_sources(data_sources, somatic=somatic, ref_version=ref_version)

    if out_prefix is None:
        out_prefix = Path(input_vcf).with_suffix("").with_suffix("")
    out_maf = f"{out_prefix}.funcotated.maf"
    # Keep GATK's scratch off the (often small) system /tmp by co-locating it
    # with the output, and silence the per-variant dbSNP WARN flood that would
    # otherwise produce tens of GB of logs on a multi-million-variant VCF.
    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(out_maf)), "gatk_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    cmd = (
        f"gatk Funcotator "
        f"-R {reference} "
        f"--ref-version {ref_version} "
        f"-V {input_vcf} "
        f"-O {out_maf} "
        f"--output-file-format MAF "
        f"--data-sources-path {data_sources} "
        f"--tmp-dir {tmp_dir} "
        f"--verbosity ERROR "
        # PLINK-derived VCFs carry contig lengths that differ from the reference
        # dictionary; the genotype coordinates are correct, so skip the check.
        f"--disable-sequence-dictionary-validation"
    )

    if not os.path.exists(out_maf):
        if not os.path.exists(f"{reference}.fai"):
            subprocess.run(f"samtools faidx {reference}", shell=True, check=True)
        if not os.path.exists(reference.replace(".fasta", ".dict")):
            subprocess.run(f"gatk CreateSequenceDictionary -R {reference}", shell=True, check=True)
        if not any(os.path.exists(f"{input_vcf}{ext}") for ext in (".idx", ".tbi")):
            subprocess.run(f"gatk IndexFeatureFile -I {input_vcf}", shell=True, check=True)
        logger.info(f"Annotating {input_vcf} → {out_maf}")
        subprocess.run(cmd, shell=True, check=True)
    else:
        logger.info(f"Funcotated MAF {out_maf} already exists. Skipping Funcotator annotation.")

    if gzip_output:
        out_maf_gz = f"{out_maf}.gz"
        if not os.path.exists(out_maf_gz):
            subprocess.run(f"gzip -f {out_maf}", shell=True, check=True)
        out_maf = out_maf_gz

    return out_maf


# ---------------------------------------------------------------------------
# Input resolution: VCFs and PLINK genotypes → VCFs
# ---------------------------------------------------------------------------

def find_plink_prefixes(folder):
    """Return PLINK fileset prefixes (paths with a matching .bed/.bim/.fam) in folder."""
    prefixes = []
    for bed in sorted(glob.glob(os.path.join(folder, "*.bed"))):
        prefix = bed[:-len(".bed")]
        if os.path.exists(prefix + ".bim") and os.path.exists(prefix + ".fam"):
            prefixes.append(prefix)
    return prefixes


def _ensure_b37_contigs(vcf, reference):
    """Rewrite the VCF's ##contig lines with the reference's exact contig lengths.

    PLINK fills ##contig lengths from the max observed position, not the true
    chromosome length, so GATK cannot recognise the genome build. Funcotator
    only maps b37 contigs (1, 2, ...) onto its hg19 (chr1, chr2, ...) data
    sources when it recognises the build, so without this every variant is
    annotated as IGR. Idempotent: skips if the lengths already match. Rewriting
    the file invalidates any .idx, which is removed so it is rebuilt.
    """
    fai = reference + ".fai"
    if not os.path.exists(fai):
        subprocess.run(f"samtools faidx {reference}", shell=True, check=True)
    lengths, contig_lines = {}, []
    with open(fai) as fh:
        for line in fh:
            name, length = line.split("\t")[:2]
            lengths[name] = length
            contig_lines.append(f"##contig=<ID={name},length={length}>\n")

    opener = gzip.open if vcf.endswith(".gz") else open
    with opener(vcf, "rt") as fh:
        for line in fh:
            if line.startswith("##contig"):
                cur = dict(kv.split("=", 1) for kv in line[line.index("<") + 1:line.rindex(">")].split(","))
                if lengths.get(cur.get("ID")) == cur.get("length"):
                    return vcf  # already correct
                break
            if line.startswith("#CHROM"):
                break

    logger.info(f"Reheadering {vcf} with reference contig lengths")
    tmp = vcf + ".reheader.tmp"
    with opener(vcf, "rt") as fin, open(tmp, "w") as fout:
        wrote = False
        for line in fin:
            if line.startswith("##contig"):
                if not wrote:
                    fout.writelines(contig_lines)
                    wrote = True
                continue
            if line.startswith("#CHROM") and not wrote:
                fout.writelines(contig_lines)
                wrote = True
            fout.write(line)
    os.replace(tmp, vcf)
    for stale in (vcf + ".idx", vcf + ".tbi"):
        if os.path.exists(stale):
            os.remove(stale)
    return vcf


def plink_to_vcf(prefix, reference, plink_cmd="plink", output_chr="MT", drop_chr=("0", "25"),
                 mac=1, snps_only=True):
    """Convert a PLINK binary fileset to a GATK-ready VCF via `plink --recode vcf`.

    A raw `plink --recode vcf` is not GATK-compatible: unplaced markers land on
    contig 0 at POS 0 (invalid VCF), the sex/MT chromosomes are numeric
    (23/24/25/26) rather than X/Y/XY/MT, monomorphic sites get ALT='.', and
    array indels are coded with non-sequence allele codes (D/I) that GATK
    rejects. We therefore drop the unplaced (0) and pseudo-autosomal XY (25)
    contigs, rename 23/24/26 → X/Y/MT (`--output-chr`), keep only sites carrying
    a minor allele (`--mac`) so each record has a real ALT, and restrict to
    A/C/G/T SNPs (`--snps-only just-acgt`). Sample names use IID only
    (`vcf-iid`). Output is written to <prefix>.gatk.vcf.
    """
    out_prefix = prefix + ".gatk"
    out_vcf = out_prefix + ".vcf"
    if not os.path.exists(out_vcf):
        parts = [plink_cmd, f"--bfile {prefix}", f"--output-chr {output_chr}",
                 "--recode vcf-iid", f"--out {out_prefix}"]
        if drop_chr:
            parts.append("--not-chr " + " ".join(drop_chr))
        if mac:
            parts.append(f"--mac {mac}")
        if snps_only:
            parts.append("--snps-only just-acgt")
        cmd = " ".join(parts)
        logger.info(f"Converting PLINK → VCF: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
    else:
        logger.info(f"VCF {out_vcf} already exists. Reusing it.")
    _ensure_b37_contigs(out_vcf, reference)
    return out_vcf


def resolve_vcfs(inputs, reference, plink_cmd="plink", prefix=None):
    """Resolve inputs (VCFs and/or PLINK filesets) into a sorted list of VCF paths.

    A directory is treated as a PLINK fileset folder if it contains any
    .bed/.bim/.fam set, otherwise as a folder of VCFs. A .bed file (or its
    prefix) is converted to VCF; any other file is taken to be a VCF.
    """
    if isinstance(inputs, (str, os.PathLike)):
        inputs = [inputs]
    vcfs = []
    for item in inputs:
        item = str(item)
        if os.path.isdir(item):
            prefixes = find_plink_prefixes(item)
            if prefix is not None:
                prefixes = [p for p in prefixes if os.path.basename(p) == prefix or p == prefix]
            if prefixes:
                vcfs.extend(plink_to_vcf(p, reference, plink_cmd) for p in prefixes)
            else:
                vcfs.extend(glob.glob(os.path.join(item, "*.vcf")))
                vcfs.extend(glob.glob(os.path.join(item, "*.vcf.gz")))
        elif os.path.isfile(item):
            if item.endswith(".bed"):
                vcfs.append(plink_to_vcf(item[:-len(".bed")], reference, plink_cmd))
            else:
                vcfs.append(item)
        elif all(os.path.exists(item + ext) for ext in (".bed", ".bim", ".fam")):
            vcfs.append(plink_to_vcf(item, reference, plink_cmd))  # bare PLINK prefix
        else:
            raise FileNotFoundError(f"Input not found: {item}")
    vcfs = sorted(set(vcfs))
    if not vcfs:
        raise FileNotFoundError(f"No VCF or PLINK inputs found in: {inputs}")
    return vcfs


# MAF Variant_Classification values considered functionally impactful (non-silent).
HIGH_IMPACT_CLASSIFICATIONS = {
    "Missense_Mutation", "Nonsense_Mutation", "Nonstop_Mutation",
    "Frame_Shift_Del", "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
    "Splice_Site", "Splice_Region", "Translation_Start_Site",
}
INDEL_TYPES = {"INS", "DEL"}


def filter_impactful_variants(df):
    """Keep indels plus medium/high-impact variants (by Variant_Classification or IMPACT)."""
    import pandas as pd

    mask = pd.Series(False, index=df.index)
    if "Variant_Classification" in df.columns:
        mask |= df["Variant_Classification"].isin(HIGH_IMPACT_CLASSIFICATIONS)
    if "Variant_Type" in df.columns:
        mask |= df["Variant_Type"].astype(str).str.upper().isin(INDEL_TYPES)
    if "IMPACT" in df.columns:
        mask |= df["IMPACT"].astype(str).str.upper().isin({"MODERATE", "HIGH"})
    return df[mask]


def annotate_variants(df, groupby_gene=False):
    """Add variant_id / patient_id / gene_symbol helper columns used to build the matrix."""
    required = {"Chromosome", "Start_Position", "Reference_Allele",
                "Tumor_Seq_Allele2", "Tumor_Sample_Barcode"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"MAF is missing required columns for matrix building: {missing}")
    df = df.copy()
    df["variant_id"] = (
        df["Chromosome"].astype(str) + ":"
        + df["Start_Position"].astype(str)
        + df["Reference_Allele"].astype(str) + ">"
        + df["Tumor_Seq_Allele2"].astype(str)
    )
    df["patient_id"] = df["Tumor_Sample_Barcode"].astype(str)
    if "Hugo_Symbol" in df.columns:
        df["gene_symbol"] = df["Hugo_Symbol"].astype(str)
    elif groupby_gene:
        raise ValueError("--groupby-gene requires a 'Hugo_Symbol' column in the MAF.")
    return df


# ---------------------------------------------------------------------------
# Genotype dosage matrix
#
# Funcotator emits variant-level annotation only (one MAF row per variant, no
# per-sample genotypes), so the patient × variant matrix is built from the
# actual genotypes. For PLINK inputs we pull additive (0/1/2) dosage with
# `plink --extract <impactful sites> --recode A`; for plain VCFs we parse the
# GT fields. The MAF supplies the per-variant annotation (gene, classification)
# carried in adata.var and used for impact filtering / gene grouping.
# ---------------------------------------------------------------------------

def _read_vcf_site_map(vcf):
    """Return a DataFrame [variant_id, name, ref, alt] from a VCF's site columns."""
    import pandas as pd

    opener = gzip.open if vcf.endswith(".gz") else open
    n_skip = 0
    with opener(vcf, "rt") as fh:
        for line in fh:
            if line.startswith("#CHROM"):
                break
            n_skip += 1
    df = pd.read_csv(vcf, sep="\t", skiprows=n_skip, header=0, dtype=str,
                     usecols=["#CHROM", "POS", "ID", "REF", "ALT"],
                     compression="gzip" if vcf.endswith(".gz") else None)
    df["variant_id"] = df["#CHROM"] + ":" + df["POS"] + df["REF"] + ">" + df["ALT"]
    return df.rename(columns={"ID": "name", "REF": "ref", "ALT": "alt"})[
        ["variant_id", "name", "ref", "alt"]]


def _plink_extract_dosage(prefix, names, plink_cmd="plink"):
    """plink --extract <names> --recode A → DataFrame patient × '<name>_<allele>'."""
    import pandas as pd

    tmp_prefix = prefix + ".impactful"
    snplist = tmp_prefix + ".snplist"
    with open(snplist, "w") as fh:
        fh.write("\n".join(names) + "\n")
    cmd = f"{plink_cmd} --bfile {prefix} --extract {snplist} --recode A --out {tmp_prefix}"
    logger.info(f"Extracting impactful genotypes: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    raw = pd.read_csv(tmp_prefix + ".raw", sep=r"\s+").set_index("IID")
    return raw.drop(columns=[c for c in ("FID", "PAT", "MAT", "SEX", "PHENOTYPE")
                             if c in raw.columns])


def _vcf_parse_dosage(vcf, keep_ids, chunksize=20000):
    """Parse VCF GT fields → DataFrame patient × variant_id additive dosage (NaN=missing)."""
    import numpy as np
    import pandas as pd

    opener = gzip.open if vcf.endswith(".gz") else open
    n_skip = 0
    with opener(vcf, "rt") as fh:
        for line in fh:
            if line.startswith("#CHROM"):
                header = line.rstrip("\n").split("\t")
                break
            n_skip += 1
    samples = header[9:]
    cols = ["#CHROM", "POS", "REF", "ALT"] + samples
    reader = pd.read_csv(vcf, sep="\t", skiprows=n_skip, header=0, dtype=str,
                         usecols=cols, chunksize=chunksize,
                         compression="gzip" if vcf.endswith(".gz") else None)
    parts = []
    for chunk in reader:
        vid = chunk["#CHROM"] + ":" + chunk["POS"] + chunk["REF"] + ">" + chunk["ALT"]
        sub = chunk[vid.isin(keep_ids)] if keep_ids is not None else chunk
        if not len(sub):
            continue
        arr = sub[samples].to_numpy(dtype=str)
        dose = np.char.count(arr, "1").astype(np.float32)        # alt-allele count
        dose[np.char.count(arr, ".") > 0] = np.nan               # missing genotype
        parts.append(pd.DataFrame(dose, index=vid[sub.index].values, columns=samples))
    if not parts:
        return pd.DataFrame()
    geno = pd.concat(parts)
    geno = geno[~geno.index.duplicated(keep="first")]
    return geno.T  # patient × variant_id


def build_dosage_matrix(vcf, keep_ids, plink_cmd="plink"):
    """Return a patient × variant_id additive-dosage DataFrame for one VCF.

    Uses `plink --extract --recode A` when the VCF came from a PLINK fileset
    (i.e. <prefix>.gatk.vcf with the .bed alongside); otherwise parses the VCF.
    Dosage counts the ALT allele so it aligns with each variant_id.
    """
    import pandas as pd

    prefix = vcf[:-len(".gatk.vcf")] if vcf.endswith(".gatk.vcf") else None
    if prefix is None or not os.path.exists(prefix + ".bed"):
        return _vcf_parse_dosage(vcf, keep_ids)

    sites = _read_vcf_site_map(vcf)
    sites = sites[sites["variant_id"].isin(keep_ids)].drop_duplicates("name")
    if sites.empty:
        return pd.DataFrame()
    raw = _plink_extract_dosage(prefix, sites["name"].tolist(), plink_cmd)

    alt_of = dict(zip(sites["name"], sites["alt"]))
    vid_of = dict(zip(sites["name"], sites["variant_id"]))
    out = {}
    for col in raw.columns:
        name, _, allele = col.rpartition("_")       # plink names columns "<name>_<counted_allele>"
        if name not in vid_of:
            continue
        vals = raw[col].astype("float32")
        if allele != alt_of[name]:                  # counted REF, not ALT → flip to ALT dosage
            vals = 2 - vals
        out[vid_of[name]] = vals
    return pd.DataFrame(out)


def build_adata(geno, maf_df, groupby_gene=False):
    """Assemble the AnnData from a patient × variant_id dosage matrix + MAF annotation."""
    import anndata as ad
    import numpy as np
    import pandas as pd

    ann = annotate_variants(maf_df, groupby_gene=groupby_gene).drop_duplicates("variant_id")
    ann = ann.set_index("variant_id")
    common = [v for v in geno.columns if v in ann.index]
    if not common:
        raise ValueError("No overlap between genotyped variants and MAF annotation.")
    geno = geno[common]

    if groupby_gene:
        gene = ann.loc[common, "gene_symbol"]
        mat = geno.T.groupby(gene).sum(min_count=1).T          # sum dosage over a gene's variants
        var = pd.DataFrame(index=mat.columns)
        var.index.name = "gene_symbol"
        var["n_variants"] = gene.value_counts().reindex(mat.columns).astype("int64")
        for col in ("Hugo_Symbol", "Chromosome", "Gencode_34_geneType"):
            if col in ann.columns:
                var[col] = ann.groupby("gene_symbol")[col].first().reindex(mat.columns)
    else:
        mat = geno
        drop = {"patient_id", "Tumor_Sample_Barcode"}
        var = ann.loc[common].drop(columns=[c for c in drop if c in ann.columns])
        var.index.name = "variant_id"

    for col in var.columns:
        if var[col].dtype == object:
            var[col] = var[col].fillna("").astype(str)

    adata = ad.AnnData(
        X=np.asarray(mat.values, dtype=np.float32),            # dense: preserves NaN (missing genotype)
        obs=pd.DataFrame(index=mat.index.astype(str)),
        var=var,
    )
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    return adata


def merge_apoe(adata, apoe_csv):
    """Merge ADNI APOE genotypes (e.g. APOERES.csv) into the matrix.

    APOE's defining SNPs (rs429358 / rs7412) are absent from the Omni2.5M array,
    so APOE is added from the supplied table: the categorical ε-pair goes to
    adata.obs['APOE_genotype'], and the ε4/ε2 allele counts (0/1/2) are appended
    as additive-dosage feature columns 'APOE_e4' / 'APOE_e2' (Hugo_Symbol=APOE).
    Patients are matched on PTID == adata.obs_names (the ADNI SubjectID).
    """
    import anndata as ad
    import numpy as np
    import pandas as pd

    geno = (pd.read_csv(apoe_csv, usecols=["PTID", "GENOTYPE"]).dropna()
            .drop_duplicates("PTID").set_index("PTID")["GENOTYPE"].astype(str))
    geno = geno.reindex(adata.obs_names)
    n_missing = int(geno.isna().sum())
    if n_missing:
        logger.info(f"APOE: {n_missing}/{adata.n_obs} patients have no genotype (set NaN)")

    adata.obs["APOE_genotype"] = geno.fillna("").values
    e4 = geno.str.count("4").to_numpy(dtype=np.float32)
    e2 = geno.str.count("2").to_numpy(dtype=np.float32)

    new_var = adata.var.reindex(list(adata.var_names) + ["APOE_e4", "APOE_e2"])
    add = ["APOE_e4", "APOE_e2"]
    if "Hugo_Symbol" in new_var:
        new_var.loc[add, "Hugo_Symbol"] = "APOE"
    if "Chromosome" in new_var:
        new_var.loc[add, "Chromosome"] = "19"
    if "n_variants" in new_var:
        new_var["n_variants"] = new_var["n_variants"].fillna(1).astype("int64")
    for col in new_var.columns:
        if new_var[col].dtype == object:
            new_var[col] = new_var[col].fillna("").astype(str)

    X = np.hstack([np.asarray(adata.X, dtype=np.float32), np.column_stack([e4, e2])])
    merged = ad.AnnData(X=X, obs=adata.obs, var=new_var)
    merged.obs_names_make_unique()
    merged.var_names_make_unique()
    logger.info("APOE: added APOE_e4 / APOE_e2 dosage features and obs['APOE_genotype']")
    return merged


def run(args):
    import pandas as pd

    if not args.out_maf and not args.out:
        raise SystemExit("nothing to do: pass --out-maf and/or --out")

    vcfs = resolve_vcfs(args.inputs, args.reference, plink_cmd=args.plink_cmd, prefix=args.prefix)
    logger.info(f"Found {len(vcfs)} VCF(s) to annotate")

    frames = []
    for vcf in vcfs:
        maf = run_funcotator(
            vcf,
            reference=args.reference,
            data_sources=args.data_sources,
            ref_version=args.ref_version,
            out_prefix=None,
            gzip_output=False,
            somatic=False,
        )
        frames.append(pd.read_csv(maf, sep="\t", comment="#", low_memory=False))

    combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    logger.info(f"Combined MAF: {combined.shape[0]:,} variants × {combined.shape[1]} columns")

    if args.filter_impact:
        before = len(combined)
        combined = filter_impactful_variants(combined)
        logger.info(f"Impact filter: kept {len(combined):,}/{before:,} variants")

    if args.out_maf:
        out_maf = args.out_maf if args.out_maf.endswith(".gz") else args.out_maf + ".gz"
        os.makedirs(os.path.dirname(os.path.abspath(out_maf)), exist_ok=True)
        combined.to_csv(out_maf, sep="\t", index=False, compression="gzip")
        logger.info(f"Wrote MAF → {out_maf}")

    if args.out:
        keep_ids = set(annotate_variants(combined)["variant_id"])
        genos = [build_dosage_matrix(vcf, keep_ids, plink_cmd=args.plink_cmd) for vcf in vcfs]
        genos = [g for g in genos if not g.empty]
        if not genos:
            raise SystemExit("No genotypes recovered for the selected variants.")
        # combine across VCFs: align patients (rows), union variants (columns)
        geno = genos[0] if len(genos) == 1 else pd.concat(genos, axis=1)
        geno = geno.loc[:, ~geno.columns.duplicated(keep="first")]
        logger.info(f"Genotype dosage: {geno.shape[0]:,} patients × {geno.shape[1]:,} variants")

        adata = build_adata(geno, combined, groupby_gene=args.groupby_gene)
        if args.apoe:
            adata = merge_apoe(adata, args.apoe)
        unit = "genes" if args.groupby_gene else "variants"
        logger.info(f"Variant matrix: {adata.n_obs:,} patients × {adata.n_vars:,} {unit}")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        adata.write_h5ad(args.out)
        logger.info(f"Wrote variant matrix → {args.out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--inputs", required=True, nargs="+",
                        help="VCF(s)/folder of VCFs and/or PLINK fileset(s)/folder.")
    parser.add_argument("--reference", required=True,
                        help="Reference genome FASTA matching the VCF coordinates.")
    parser.add_argument("--data-sources", default=None,
                        help="Funcotator data-sources directory (downloaded if missing).")
    parser.add_argument("--ref-version", default="hg19",
                        choices=["hg19", "hg38", "grch37", "grch38"],
                        help="Reference build (default: hg19).")
    parser.add_argument("--prefix", default=None,
                        help="Restrict to this PLINK fileset prefix when an input "
                             "folder holds more than one.")
    parser.add_argument("--plink-cmd", default="plink",
                        help="PLINK 1.9 executable name/path (default: plink).")
    parser.add_argument("--out-maf", default=None,
                        help="Output MAF path (always gzipped). All inputs are "
                             "concatenated into this single MAF.")
    parser.add_argument("--out", default="adni_variant_matrix.h5ad",
                        help="Output patient × variant matrix (.h5ad); adata.var "
                             "holds the MAF annotation columns.")
    parser.add_argument("--filter-impact", action="store_true",
                        help="Keep only impactful variants (indels and medium/high-impact "
                             "Variant_Classification / IMPACT).")
    parser.add_argument("--groupby-gene", action="store_true",
                        help="Build the matrix per gene (Hugo_Symbol) instead of per variant.")
    parser.add_argument("--apoe", default=None,
                        help="ADNI APOE table (e.g. APOERES.csv) to merge into --out: adds "
                             "APOE_e4/APOE_e2 dosage features and obs['APOE_genotype'].")
    return parser


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
