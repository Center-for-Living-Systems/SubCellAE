# FA Patch Labeling Guide

Manual annotation tool for classifying focal adhesion (FA) patches from fluorescence microscopy images.  
Browser-based — no programming required to use once the tool is running.

# Datasets
When on UChicago network, you can access these four datasets (both control and ycomp)'s FA type annotation based on paxillin and other facilitating compositions. Here are the eight links.

Vinc-Pax-Zyx-Act-031125:<br>
Control: http://128.135.108.226:5007/vinc_control_label<br>
Ycomp: http://128.135.108.226:5007/vinc_ycomp_label


pPax-Pax-Zyx-Act-072025:<br>
Control: http://128.135.108.226:5007/ppax_control_label<br>
Ycomp: http://128.135.108.226:5007/ppax_ycomp_label


pFAK-Pax-Zyx-Act-072125:<br>
Control: http://128.135.108.226:5007/pfak_control_label <br>
Ycomp: http://128.135.108.226:5007/pfak_ycomp_label


Vinc-Pax-Zyx-Act-022726:<br>
Control: http://128.135.108.226:5007/nih3t3_control_label<br>
Ycomp: http://128.135.108.226:5007/nih3t3_ycomp_label  

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
│                                        │  [paxillin] [marker] [zyxin] [actin]   │
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



