#!/usr/bin/env python3
"""
eval_b1_b2_matched.py

5-fold × 10-repeat LGBM classification on the B1/B2 matched label sets
using CellProfiler, ilastik, and pretrained SupCon-B12 latent features.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from lightgbm import LGBMClassifier

FEAT    = "/net/projects/CLS/lding/data/fa_data_analysis/ae_results/features"
LAB     = "/net/projects/CLS/lding/data/fa_data_analysis/labelling"
LAT_RUN = (FEAT.replace("features", "contrastive_run/le_b12_ds1_lat12p8") +
           "/le_b12_ds1_lat12p8_fv0_nb750_r0/latents.csv")

N_SPLITS  = 5
N_REPEATS = 10

# ── Feature tables ───────────────────────────────────────────────────────────
cp_feat  = pd.read_csv(f"{FEAT}/cellprofiler/ds1.csv").set_index("filename")
il_feat  = pd.read_csv(f"{FEAT}/ilastik/ds1.csv").set_index("filename")
lat_df   = pd.read_csv(LAT_RUN)
lat_cols = [c for c in lat_df.columns if c.startswith("z_")]
lat_feat = lat_df[["filename"] + lat_cols].set_index("filename")

FEATURES = [("CP", cp_feat), ("ilastik", il_feat), ("SupCon-B12", lat_feat)]

def uid_to_fn(uid):
    return uid.replace("-f", "_f", 1)

def run_cv(X, y, n_splits, n_repeats):
    scores = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rep)
        for train_idx, test_idx in skf.split(X, y):
            clf = LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                 num_leaves=15, random_state=rep, verbose=-1,
                                 n_jobs=1)
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            scores.append(balanced_accuracy_score(y[test_idx], pred))
    return np.array(scores)

results = []
for ann_name, ann_path in [("B2", f"{LAB}/vinc_b2_matched_ds1.csv"),
                            ("B1", f"{LAB}/vinc_b1_matched_ds1.csv")]:
    ann = pd.read_csv(ann_path)
    ann["filename"] = ann["unique_ID"].apply(uid_to_fn)

    for feat_name, feat_df in FEATURES:
        matched = ann.join(feat_df, on="filename", how="inner")
        missing = len(ann) - len(matched)
        feat_cols = feat_df.columns.tolist()
        X = matched[feat_cols].values.astype(float)
        y = (matched["label"] == "adhesion").astype(int).values

        scores = run_cv(X, y, N_SPLITS, N_REPEATS)
        mean, std = scores.mean(), scores.std()
        print(f"{ann_name:>4}  {feat_name:>10}  n={len(matched)}  miss={missing}  "
              f"mean={mean:.3f}  std={std:.3f}  min={scores.min():.3f}  max={scores.max():.3f}",
              flush=True)
        results.append(dict(batch=ann_name, features=feat_name,
                            n=len(matched), missing=missing,
                            mean=round(mean, 4), std=round(std, 4)))

print("\n=== Summary ===")
print(f"{'Batch':>5}  {'Features':>10}  {'mean±std':>14}  {'n':>4}")
print("-" * 42)
for r in results:
    print(f"{r['batch']:>5}  {r['features']:>10}  "
          f"{r['mean']:.3f} ± {r['std']:.3f}  {r['n']:>4}")

out = pd.DataFrame(results)
out.to_csv(f"{LAB}/b1_b2_matched_cv_results.csv", index=False)
print(f"\nSaved → {LAB}/b1_b2_matched_cv_results.csv")
