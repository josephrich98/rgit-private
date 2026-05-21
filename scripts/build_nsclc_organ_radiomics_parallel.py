"""Parallel pyradiomics build for the NSCLC organ-radiomics matrix.

Same per-patient pipeline as make_imaging_matrix.py (resample -> apply organ
mask -> extract original_* features) but fanned out across processes, because
the BSpline resample of full-chest CT is the bottleneck. Resampled/masked
intermediates are cached on disk, so reruns are cheap.
"""
import os, logging, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix

logging.getLogger("radiomics").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

REPO = Path(__file__).parent.parent
META = REPO / "data/nsclc/imaging/metadata_subset.csv"
OUT = REPO / "data/nsclc/imaging/organ_radiomics_subset.h5ad"
SPACING = (2.0, 2.0, 2.0)


def one(row):
    from tcia_radiology_processing import utils
    from radiomics import featureextractor
    pid = str(row["patient_id"])
    img, mask = row["image"], row["organ_mask"]
    if not (os.path.exists(img) and os.path.exists(mask)):
        return pid, None
    try:
        img_r = utils.resample_image(img, target_spacing=SPACING, is_label=False, out=True)
        mask_r = utils.resample_image(mask, target_spacing=SPACING, is_label=True, out=True)
        oi = img_r.replace(".nii", "_organ_mask_masked.nii")
        om = mask_r.replace(".nii", "_organ_mask_masked.nii")
        img_m, mask_m = utils.apply_mask(img_r, mask_r, label=1, min_value=None,
                                         crop=True, pad_after_crop=5,
                                         out_image=oi, out_mask=om)
        ex = featureextractor.RadiomicsFeatureExtractor()
        f = ex.execute(img_m, mask_m, label=1)
        feats = {k: float(v) for k, v in f.items() if k.startswith("original_")}
        return pid, feats
    except Exception as e:
        return pid, f"ERR: {e}"


def main(workers=24):
    meta = pd.read_csv(META).drop_duplicates("patient_id")
    rows = [r for _, r in meta.iterrows()]
    records, errs = {}, {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, r): str(r["patient_id"]) for r in rows}
        done = 0
        for fut in as_completed(futs):
            pid, res = fut.result()
            done += 1
            if isinstance(res, dict):
                records[pid] = res
            else:
                errs[pid] = res
            if done % 10 == 0:
                print(f"  {done}/{len(rows)} done, {len(records)} ok", flush=True)
    print(f"extracted {len(records)} patients; {len(errs)} skipped/errored")
    if errs:
        print("examples:", list(errs.items())[:3])
    df = pd.DataFrame.from_dict(records, orient="index").sort_index()
    df = df.loc[:, df.nunique() > 1]                      # drop constant cols
    adata = ad.AnnData(X=csr_matrix(df.values.astype(np.float32)),
                       obs=pd.DataFrame(index=df.index),
                       var=pd.DataFrame(index=df.columns))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(OUT)
    print(f"Saved imaging AnnData: {adata.shape} -> {OUT}")


if __name__ == "__main__":
    main()
