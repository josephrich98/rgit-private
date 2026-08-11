"""Build TCGA-KIRC whole-slide-image (WSI) feature AnnData objects.

The digital-pathology analogue of ``make_imaging_matrix.py``: instead of CT
organ radiomics, we extract an interpretable handcrafted histopathology feature
bank (stain / first-order / texture / structure families; see
``interpretability_radiogenomics.wsi_features``) from each TCGA-KIRC H&E
whole-slide image, then fold the per-slide features into case-indexed
``AnnData`` matrices that drop straight into the recoverability harness in place
of ``organ_radiomics.h5ad``.

Slides map to TCGA Case IDs and tumour/normal status via the GDC
``sample_sheet.tsv``. Genomics is keyed by Case ID, so a case's tumour-slide (or
normal-slide) feature vector aligns to its expression / mutation profile.

Outputs (all in ``data/tcga_kirc/wsi/``):
  wsi_features_slide.h5ad  per-slide features (obs: slide; .obs has case_id, tissue)
  wsi_tumor.h5ad           per-case mean over that case's TUMOUR slides  (Case ID index)
  wsi_normal.h5ad          per-case mean over that case's NORMAL slides  (Case ID index)

Feature extraction is slow (~1 min/slide, single in-memory pyramid decode). The
580-slide bank produced by the companion interpretability repo is reused when
present (``--cache-pkl``); pass ``--force-extract`` to regenerate from the
``.svs`` files directly.

Run from the repo root:  python scripts/build_kirc_wsi_features.py
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

REPO = Path(__file__).parent.parent
WSI_DIR = REPO / "data/tcga_kirc/wsi"
SAMPLE_SHEET = WSI_DIR / "sample_sheet.tsv"
# Handcrafted-feature cache from the companion interpretability repo (same .svs,
# same extractor). Reused verbatim to avoid a multi-hour re-decode.
DEFAULT_CACHE = Path("/home/jrich/Desktop/interpretability_radiogenomics/"
                     "artifacts_kirc_wsi")


# ---------------------------------------------------------------------------
# Slide inventory + extraction (mirrors run_kirc_wsi.py)
# ---------------------------------------------------------------------------
def load_inventory():
    sheet = pd.read_csv(SAMPLE_SHEET, sep="\t")
    rows = []
    for _, r in sheet.iterrows():
        fid, fname = str(r["File ID"]), str(r["File Name"])
        path = WSI_DIR / "images" / fid / fname
        if not path.exists():
            continue
        rows.append({"slide": fname.split(".")[0], "path": str(path),
                     "case_id": str(r["Case ID"]),
                     "tissue": str(r["Tissue Type"]).strip().lower()})
    return pd.DataFrame(rows)


def _extract_one(args):
    import sys
    sys.path.insert(0, "/home/jrich/Desktop/interpretability_radiogenomics")
    from interpretability_radiogenomics import wsi_features as W
    slide, path, target_px = args
    feats, _ = W.slide_features(path, target_px=target_px, thumb=32)
    return slide, feats


def extract_features(inv, target_px, n_jobs):
    tasks = [(r.slide, r.path, target_px) for r in inv.itertuples()]
    feats = {}
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = {ex.submit(_extract_one, t): t[0] for t in tasks}
        done = 0
        for fu in as_completed(futs):
            slide, f = fu.result()
            done += 1
            if f is not None:
                feats[slide] = f
            if done % 25 == 0:
                print(f"  extracted {done}/{len(tasks)} ({len(feats)} ok)", flush=True)
    order = [s for s in inv.slide if s in feats]
    feat_df = pd.DataFrame([feats[s] for s in order], index=order)
    meta = inv.set_index("slide").loc[order, ["case_id", "tissue"]].reset_index()
    return feat_df, meta


def load_cache(cache_dir):
    feat_df = pd.read_pickle(cache_dir / "wsi_features.pkl")
    meta = pd.read_csv(cache_dir / "wsi_meta.csv")
    return feat_df, meta


# ---------------------------------------------------------------------------
# AnnData assembly
# ---------------------------------------------------------------------------
def case_aggregate(feat_df, meta, tissue):
    """Mean a case's per-slide features over slides of the given tissue type.

    Returns an (n_case, n_feat) AnnData keyed by Case ID, with the slide count
    folded into ``.obs['n_slides']``.
    """
    m = meta[meta["tissue"] == tissue]
    X = feat_df.loc[m["slide"].values]
    X = X.assign(case_id=m["case_id"].values)
    agg = X.groupby("case_id").mean()
    n = X.groupby("case_id").size().reindex(agg.index)
    A = ad.AnnData(agg.values.astype(np.float32),
                   obs=pd.DataFrame({"n_slides": n.values.astype(int)},
                                    index=agg.index.astype(str)),
                   var=pd.DataFrame(index=agg.columns.astype(str)))
    A.obs_names.name = "case_id"
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-pkl", default=str(DEFAULT_CACHE),
                    help="dir with wsi_features.pkl + wsi_meta.csv to reuse")
    ap.add_argument("--force-extract", action="store_true",
                    help="re-extract features from the .svs files (~1 min/slide)")
    ap.add_argument("--target-px", type=int, default=2048)
    ap.add_argument("--n-jobs", type=int, default=16)
    args = ap.parse_args()

    cache = Path(args.cache_pkl)
    if not args.force_extract and (cache / "wsi_features.pkl").exists():
        feat_df, meta = load_cache(cache)
        print(f"Reused cached handcrafted features from {cache}: {feat_df.shape}")
    else:
        inv = load_inventory()
        print(f"Slides on disk: {len(inv)} "
              f"(tumor={int((inv.tissue=='tumor').sum())}, "
              f"normal={int((inv.tissue=='normal').sum())})")
        feat_df, meta = extract_features(inv, args.target_px, args.n_jobs)
        print(f"Extracted features {feat_df.shape}")

    # clean: drop constant columns, impute residual NaNs by column median
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
    feat_df = feat_df.fillna(feat_df.median(numeric_only=True))
    feat_df = feat_df.loc[:, feat_df.std(0) > 1e-9]

    # per-slide AnnData (obs carries case_id + tissue)
    slide = ad.AnnData(feat_df.values.astype(np.float32),
                       obs=meta.set_index("slide")[["case_id", "tissue"]].loc[feat_df.index],
                       var=pd.DataFrame(index=feat_df.columns.astype(str)))
    slide.write_h5ad(WSI_DIR / "wsi_features_slide.h5ad")

    tumor = case_aggregate(feat_df, meta, "tumor")
    normal = case_aggregate(feat_df, meta, "normal")
    tumor.write_h5ad(WSI_DIR / "wsi_tumor.h5ad")
    normal.write_h5ad(WSI_DIR / "wsi_normal.h5ad")

    print(f"slides   : {slide.shape}  ({(meta.tissue=='tumor').sum()} tumor, "
          f"{(meta.tissue=='normal').sum()} normal)")
    print(f"wsi_tumor : {tumor.shape}  (cases with >=1 tumour slide)")
    print(f"wsi_normal: {normal.shape}  (cases with >=1 normal slide)")
    print(f"features  : {list(feat_df.columns)}")
    print(f"Wrote AnnData objects to {WSI_DIR}")


if __name__ == "__main__":
    main()
