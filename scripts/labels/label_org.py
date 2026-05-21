# /home/liyading/miniconda3/envs/subcellae-cuda/bin/python3 - <<'EOF'
import re, pandas as pd
from pathlib import Path
from datetime import date

lbl_dir  = Path("/net/projects/CLS/lding/data/fa_data_analysis/labelling")
today    = date.today().strftime("%Y%m%d")   # 20260520

OUT_COLS = ["dataset", "unique_ID", "condition", "crop_img_filename",
            "czi_filename", "classification", "Position", "annotator"]

COORD_RE = re.compile(r'(f\d+x\d+y\d+ps\d+\.tiff?)$', re.IGNORECASE)

def bare_coord(fn):
    m = COORD_RE.search(Path(fn).name)
    return m.group(1) if m else None

def make_uid(condition, crop_img_filename):
    return f"{condition}-{crop_img_filename}"

# ── 1. VINC ─────────────────────────────────────────────────────────────────
pax = pd.read_csv(lbl_dir / "paxdata_paxpatch_batch1and2_combined_labels.csv")
# pax has: unique_ID, crop_img_filename, group (=condition), group_ID, Position, classification
# get czi_filename from Margaret source CSVs via crop_img_filename join
ctrl_src  = pd.read_csv(lbl_dir / "Margaret_Control_V2_project-19-at-2026-02-09-21-46-5552392a.csv",
                        usecols=["crop_img_filename", "czi_filename"])
ycomp_src = pd.read_csv(lbl_dir / "Margaret_Ycomp_V2_project-20-at-2026-02-09-21-40-9d1e4d7c.csv",
                        usecols=["crop_img_filename", "czi_filename"])
# batch1 source CSVs have czi_filename but it was dropped in LABEL_COLS when combined;
# add them here to fill the 376 rows that Margaret CSVs don't cover
b1_ctrl1 = pd.read_csv(lbl_dir / "project-13-at-2025-12-18-15-32-df671bd2.csv",
                        usecols=["crop_img_filename", "czi_filename"])
b1_ctrl2 = pd.read_csv(lbl_dir / "project-13-at-2025-12-19-15-41-effff775.csv",
                        usecols=["crop_img_filename", "czi_filename"])
b1_ycomp = pd.read_csv(lbl_dir / "project-15-at-2025-12-22-18-44-b7e23381.csv",
                        usecols=["crop_img_filename", "czi_filename"])
czi_map   = pd.concat([ctrl_src, ycomp_src, b1_ctrl1, b1_ctrl2, b1_ycomp]).drop_duplicates("crop_img_filename")

vinc = pax.rename(columns={"group": "condition"}).merge(
    czi_map, on="crop_img_filename", how="left"
)
vinc["dataset"]   = "vinc"
vinc["annotator"] = "Margaret"
vinc_out = vinc[OUT_COLS].copy()
print(f"vinc: {len(vinc_out)} rows, czi_filename coverage: {vinc_out['czi_filename'].notna().sum()}")

# ── 2. PPAX ─────────────────────────────────────────────────────────────────
ppax_src = pd.read_csv(lbl_dir / "project-1-at-2025-12-01-16-46-54a904cb.csv")
# crop_img_filename = ctrl_ch1_f0000x0176y0336ps32.tif  → bare coord
ppax_src["crop_img_filename"] = ppax_src["crop_img_filename"].apply(bare_coord)
# condition from czi_filename
ppax_src["condition"] = (
    ppax_src["czi_filename"].str.lower()
    .str.extract(r'(control|ycomp)', expand=False)
)
ppax_src["unique_ID"] = ppax_src.apply(
    lambda r: make_uid(r["condition"], r["crop_img_filename"]), axis=1
)
ppax_src["dataset"] = "ppax"
ppax_out = ppax_src.reindex(columns=OUT_COLS).copy()
print(f"ppax: {len(ppax_out)} rows, czi_filename coverage: {ppax_out['czi_filename'].notna().sum()}")
print(f"  condition breakdown: {ppax_src['condition'].value_counts().to_dict()}")

# ── 3. PFAK ─────────────────────────────────────────────────────────────────
pfak_src = pd.read_csv(lbl_dir / "pfak_labels_Annabel_20260427_1035.csv")
# filename = control_f0000x0400y0400ps32.tif
pfak_src["condition"]        = pfak_src["filename"].str.extract(r'^(control|ycomp)', expand=False)
pfak_src["crop_img_filename"] = pfak_src["filename"].apply(bare_coord)
pfak_src["unique_ID"]        = pfak_src.apply(
    lambda r: make_uid(r["condition"], r["crop_img_filename"]), axis=1
)
pfak_src["dataset"]          = "pfak"
pfak_src["classification"]   = pfak_src["label"]
pfak_src["czi_filename"]     = None
pfak_src["Position"]         = None
pfak_out = pfak_src.reindex(columns=OUT_COLS).copy()
print(f"pfak: {len(pfak_out)} rows, czi_filename coverage: {pfak_out['czi_filename'].notna().sum()}")

# ── SAVE ────────────────────────────────────────────────────────────────────
for name, df in [("vinc", vinc_out), ("ppax", ppax_out), ("pfak", pfak_out)]:
    out_path = lbl_dir / f"labels_{name}_{today}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path.name}")
    print(df["classification"].value_counts().to_dict())
