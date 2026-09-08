#!/usr/bin/env python3
"""
eval_cnn_le_b1b2.py

Direct CNN patch classifier (EfficientNet-B0) on the B1/B2 LE benchmark.
Trains end-to-end on labeled patches only; no feature extraction step.

Reuses annotation CSVs from labelling/le_b1b2_le/ and fold splits from
labelling/le_b1b2_matched/.

Output: eval_results/cnn_b1b2_{batch}_ds1.csv
Columns: name, batch, fold, budget, repeat, n_train, n_test, bal_acc

Usage
-----
  python scripts/eval_cnn_le_b1b2.py --batch b1
  python scripts/eval_cnn_le_b1b2.py --batch b2
  python scripts/eval_cnn_le_b1b2.py --all
  python scripts/eval_cnn_le_b1b2.py --batch b1 --budget 50 --fold 0 --repeat 0
"""
from __future__ import annotations

import argparse
import math
import re
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import timm
from sklearn.metrics import balanced_accuracy_score
import tifffile

_DEFAULT_DATA = Path("/net/projects/CLS/lding/data/fa_data_analysis")

# All path constants are resolved in main() after --data-dir is parsed.
DATA: Path
EVAL_DIR: Path
ANN_DIR:  Path
FS_DIR:   Path
PATCH_DIRS: dict

BUDGETS   = [10, 20, 25, 50, 75, 100, 150, 200]
N_FOLDS   = 5
N_REPEATS = 5

# Training hyper-parameters
IMG_SIZE    = 64       # upsample 32→64
EPOCHS      = 100   # ~5–15 s per job on GPU A40; set lower for quick smoke tests
LR          = 5e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE  = 64
NUM_WORKERS = 2        # set 0 for local/debug


def _h2u(uid: str) -> str:
    """'control-f0001...' → 'control_f0001...' (filesystem name)"""
    return uid.replace("-f", "_f", 1)


def _uid_to_path(uid: str) -> Path:
    fn = _h2u(uid)                   # e.g. control_f0001x0592y0560ps32.tif
    cond = fn.split("_")[0]          # control / ycomp
    return PATCH_DIRS[cond] / fn


class PatchDataset(Dataset):
    def __init__(self, uids: list[str], labels: list[int], augment: bool = False):
        self.paths  = [_uid_to_path(u) for u in uids]
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = tifffile.imread(self.paths[idx]).astype(np.float32)
        # clip and normalize to [0, 1]
        img = np.clip(img, 0.0, None)
        vmax = img.max()
        if vmax > 0:
            img = img / vmax

        # upsample with numpy nearest-neighbor (fast, avoids torchvision dependency)
        if IMG_SIZE != img.shape[0]:
            scale = IMG_SIZE // img.shape[0]
            img = img.repeat(scale, axis=0).repeat(scale, axis=1)

        # channel dim
        img = torch.from_numpy(img).unsqueeze(0)   # (1, H, W)

        if self.augment:
            if torch.rand(1) > 0.5:
                img = torch.flip(img, [2])          # horizontal flip
            if torch.rand(1) > 0.5:
                img = torch.flip(img, [1])          # vertical flip
            k = torch.randint(4, (1,)).item()
            if k:
                img = torch.rot90(img, k, [1, 2])

        return img, self.labels[idx]


def _build_model(device: torch.device) -> nn.Module:
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=2,
    )
    # Adapt pretrained 3-channel stem to single channel
    w = model.conv_stem.weight.data        # (32, 3, 3, 3)
    model.conv_stem.weight = nn.Parameter(w.mean(dim=1, keepdim=True))
    model.conv_stem.in_channels = 1
    return model.to(device)


def _label_enc(labels: list[str]) -> tuple[list[int], dict]:
    classes = sorted(set(labels))
    enc = {c: i for i, c in enumerate(classes)}
    return [enc[l] for l in labels], enc


def _train_eval(
    train_uids: list[str], train_labels: list[int],
    test_uids:  list[str], test_labels:  list[int],
    class_weights: torch.Tensor,
    device: torch.device,
) -> float:
    model = _build_model(device)

    train_ds = PatchDataset(train_uids, train_labels, augment=True)
    test_ds  = PatchDataset(test_uids,  test_labels,  augment=False)
    train_dl = DataLoader(train_ds, batch_size=min(BATCH_SIZE, len(train_ds)),
                          shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=128,
                          shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    model.train()
    for _ in range(EPOCHS):
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for imgs, lbls in test_dl:
            preds = model(imgs.to(device)).argmax(dim=1).cpu().numpy()
            all_pred.extend(preds)
            all_true.extend(lbls.numpy())

    return balanced_accuracy_score(all_true, all_pred)


def _run_one(batch: str, fold: int, budget: str, repeat: int, fold_splits: pd.DataFrame, device: torch.device) -> dict | None:
    ann_name = f"le_b1b2_{batch}_fv{fold}_nb{budget}_r{repeat}"
    ann_path = ANN_DIR / f"{ann_name}.csv"
    if not ann_path.exists():
        return None

    train_ann = pd.read_csv(ann_path)
    test_fs   = fold_splits[fold_splits["fold"] == fold].copy()
    train_set = set(train_ann["unique_ID"])
    test_fs   = test_fs[~test_fs["unique_ID"].isin(train_set)]

    # filter to existing patches
    train_ok = [_uid_to_path(u).exists() for u in train_ann["unique_ID"]]
    test_ok  = [_uid_to_path(u).exists() for u in test_fs["unique_ID"]]
    train_ann = train_ann[train_ok]
    test_fs   = test_fs[test_ok]

    if len(train_ann) == 0 or len(test_fs) == 0:
        return None
    if len(train_ann["label"].unique()) < 2:
        return None

    train_labels_str = train_ann["label"].tolist()
    test_labels_str  = test_fs["label"].tolist()
    all_labels_str   = train_labels_str + test_labels_str
    _, enc = _label_enc(all_labels_str)
    train_labels_int = [enc[l] for l in train_labels_str]
    test_labels_int  = [enc[l] for l in test_labels_str]

    # class weights for loss (inverse frequency)
    counts = np.bincount(train_labels_int, minlength=2).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    cw = torch.tensor(1.0 / counts / (1.0 / counts).sum() * len(counts), dtype=torch.float32)

    bal_acc = _train_eval(
        train_ann["unique_ID"].tolist(), train_labels_int,
        test_fs["unique_ID"].tolist(),   test_labels_int,
        cw, device,
    )

    name = f"le_b1b2_{batch}_fv{fold}_nb{budget}_r{repeat}"
    return dict(
        name=name, batch=batch, fold=fold, budget=str(budget), repeat=repeat,
        n_train=len(train_labels_int), n_test=len(test_labels_int),
        bal_acc=round(bal_acc, 6),
    )


def _run_batch(batch: str, device: torch.device,
               single_fold: int | None   = None,
               single_budget: str | None = None,
               single_repeat: int | None = None) -> pd.DataFrame:
    fs = pd.read_csv(FS_DIR / f"fold_splits_{batch}.csv")

    folds   = [single_fold]   if single_fold   is not None else list(range(N_FOLDS))
    budgets = [single_budget] if single_budget is not None else [str(b) for b in BUDGETS] + ["all"]
    repeats = [single_repeat] if single_repeat is not None else list(range(N_REPEATS))

    # "all" budget has only repeat=0
    results, n_skip = [], 0
    total = sum(1 if b == "all" else len(repeats) for b in budgets) * len(folds)
    done  = 0
    print(f"  {batch}: {total} jobs", flush=True)

    for fold in folds:
        for budget in budgets:
            reps = [0] if budget == "all" else repeats
            for repeat in reps:
                if done % 20 == 0:
                    print(f"    {done}/{total} ...", flush=True)
                row = _run_one(batch, fold, budget, repeat, fs, device)
                if row:
                    results.append(row)
                else:
                    n_skip += 1
                done += 1

    print(f"    Done: {len(results)} rows, {n_skip} skipped", flush=True)
    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--batch", choices=["b1", "b2"])
    grp.add_argument("--all",   action="store_true")
    ap.add_argument("--fold",     type=int,  default=None)
    ap.add_argument("--budget",   type=str,  default=None)
    ap.add_argument("--repeat",   type=int,  default=None)
    ap.add_argument("--device",   default="auto")
    ap.add_argument("--data-dir", default=None,
                    help="Root data dir (default: cluster path). Override for local runs.")
    args = ap.parse_args()

    # Resolve path constants (allows portable local runs)
    global DATA, EVAL_DIR, ANN_DIR, FS_DIR, PATCH_DIRS
    DATA = Path(args.data_dir) if args.data_dir else _DEFAULT_DATA
    EVAL_DIR   = DATA / "ae_results" / "features" / "eval_results"
    ANN_DIR    = DATA / "labelling" / "le_b1b2_le"
    FS_DIR     = DATA / "labelling" / "le_b1b2_matched"
    PATCH_DIRS = {
        "control": DATA / "ae_results" / "patches" / "cio" / "vinc" / "control" / "tiff_patches32_mr10",
        "ycomp":   DATA / "ae_results" / "patches" / "cio" / "vinc" / "ycomp"   / "tiff_patches32_mr10",
    }

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    batches = ["b1", "b2"] if args.all else [args.batch]
    for batch in batches:
        print(f"\n=== CNN EfficientNet-B0 | {batch} ===", flush=True)
        df = _run_batch(
            batch, device,
            single_fold   = args.fold,
            single_budget = args.budget,
            single_repeat = args.repeat,
        )
        if df.empty:
            print("  No results.")
            continue

        out = EVAL_DIR / f"cnn_b1b2_{batch}_ds1.csv"
        # append if partial run
        if out.exists() and (args.fold is not None or args.budget is not None):
            existing = pd.read_csv(out)
            df = pd.concat([existing, df], ignore_index=True).drop_duplicates(
                subset=["batch", "fold", "budget", "repeat"], keep="last"
            )
        df.to_csv(out, index=False)
        print(f"  Saved → {out} ({len(df)} rows)", flush=True)

        num = df[df["budget"] != "all"].copy()
        if not num.empty:
            num["b"] = num["budget"].astype(int)
            s = num.groupby("b")["bal_acc"].mean()
            print(f"  Mean bal_acc by budget:\n{s.round(3).to_string()}", flush=True)


if __name__ == "__main__":
    main()
