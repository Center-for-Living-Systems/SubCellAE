# FA Patch Labeling Guide

Manual annotation tool for classifying focal adhesion (FA) patches from fluorescence microscopy images.  
Browser-based — no programming required to use once the tool is running.

---

## What you need

An `.h5` data file for your dataset, located on the NAS at:

```
/mnt/p/Liya/FA_patch_group_label/{dataset}/{dataset}_{condition}_label.h5
```

Example: `/mnt/p/Liya/FA_patch_group_label/vinc/vinc_control_label.h5`

The person running the server (Liya) handles generating these files. Just open the URL they send you.

---

## Opening the tool

### If someone is already running the server

Open the URL in any browser — nothing to install.

```
http://<server-ip>:5007/vinc_control_label
```

### Running it yourself (local or server)

```bash
# Single dataset — opens at http://localhost:5007
python scripts/label_patches.py /mnt/p/Liya/FA_patch_group_label/vinc/vinc_control_label.h5

# Multiple datasets at once
python scripts/label_patches.py \
    /mnt/p/Liya/FA_patch_group_label/vinc/vinc_control_label.h5 \
    /mnt/p/Liya/FA_patch_group_label/vinc/vinc_ycomp_label.h5

# Shared lab server mode (others can connect over the network)
python scripts/label_patches.py /mnt/p/Liya/FA_patch_group_label/vinc/vinc_control_label.h5 \
    --serve --port 5007 --nas-mount /mnt/p/ --nas-name "GardelNas Expansion"
```

# for all
python scripts/label_patches.py   /mnt/p/Liya/FA_patch_group_label/vinc/vinc_control_label.h5   /mnt/p/Liya/FA_patch_group_label/vinc/vinc_ycomp_label.h5  /mnt/p/Liya/FA_patch_group_label/ppax/ppax_control_label.h5   /mnt/p/Liya/FA_patch_group_label/ppax/ppax_ycomp_label.h5 /mnt/p/Liya/FA_patch_group_label/pfak/pfak_control_label.h5   /mnt/p/Liya/FA_patch_group_label/pfak/pfak_ycomp_label.h5 /mnt/p/Liya/FA_patch_group_label/nih3t3/nih3t3_control_label.h5   /mnt/p/Liya/FA_patch_group_label/nih3t3/nih3t3_ycomp_label.h5 --port 5007 --serve





---

## Interface overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Annotator: [your name]     [frame selector ▼]                                  │
│  Resume CSV: [path/to/previous.csv]  [Load CSV]                                 │
│  Active label: [Nascent Adhesion] [focal complex] [focal adhesion] ...          │
│                                               Labeled: 12   [Finish & Save]     │
├────────────────────────────────────────┬────────────────────────────────────────┤
│                                        │  [marker]  [zyxin]  [actin]            │
│   Paxillin — main canvas (720×720)     │  (side canvases, linked pan/zoom)      │
│   patches shown as coloured boxes      │                                        │
│                                        ├────────────────────────────────────────┤
│                                        │  [paxillin] [marker] [zyxin] [actin]  │
│                                        │  (patch thumbnails, on click)          │
└────────────────────────────────────────┴────────────────────────────────────────┘
```

**Main canvas** — full paxillin frame; all patches in the current field of view are outlined as boxes. Unlabeled patches are faint; labeled patches fill with the label colour.

**Side canvases** — the other three channels (marker, zyxin, actin) at the same zoom level, linked to the main canvas. Pan and zoom on any canvas and all four move together.

**Patch thumbnails** — appear when you click a patch; show a close-up crop from each channel.

---

## How to label

1. **Enter your name** in the *Annotator* field (used in the output filename).
2. **Select a label** from the row of buttons.
3. **Choose a frame** from the dropdown (format: `condition | frame`).
4. **Click a patch box** on the canvas to assign the active label.
   - The box fills with colour immediately.
   - Thumbnails for all four channels appear on the right.
5. **Double-click a patch** to remove its label.
6. Switch frames freely — labels accumulate across the whole session.
7. Click **Finish & Save** when done.

> Clicking outside any patch box does nothing. You must click inside the coloured outline.

---

## Label categories

| Label | Meaning |
|-------|---------|
| **Nascent Adhesion** | Small, round, newly formed adhesion near the cell edge |
| **focal complex** | Slightly larger than nascent; still near the periphery |
| **focal adhesion** | Mature, elongated adhesion; strong paxillin signal |
| **fibrillar adhesion** | Long, thin, centrally located; often associated with fibronectin fibrils |
| **No adhesion** | Patch contains no real FA structure (background, debris, etc.) |

---

## Resuming a previous session

The tool auto-fills the **Resume CSV** field with the most recent saved file for this dataset. Click **Load CSV** to reload those labels before continuing.

You can also paste any CSV path manually — it must have `filename` and `label` columns.

---

## Saving

Click **Finish & Save**. The file is written next to the `.h5` on the NAS:

```
{dataset}_{condition}_label_{your_name}_{YYYYMMDD_HHMM}.csv
```

Example: `vinc_control_label_annabel_20260705_1432.csv`

Each row has three columns: `filename`, `label`, `annotator`.  
You can save multiple times in one session — each save creates a new timestamped file.

---

## Tips

- **Pan/zoom**: use the toolbar above the main canvas (scroll to zoom, drag to pan). All canvases stay in sync.
- **Reset view**: click the reset (house) icon in the toolbar.
- **Multiple annotators**: each person opens the tool in their own browser tab. Sessions are independent; labels do not interfere with each other.
- If the page loads but nothing responds, the server may have restarted — refresh the browser tab.
