"""
generate_patchprep_ch02.py
==========================
Generates patchprep configs and a Slurm submission script for
ch0 (vinc/pfak/ppax protein) and ch2 (zyxin) patches across all 4 datasets.

These directories are needed by the protein sweep (generate_protein_sweep.py):
  {ds}_ch0/{condition}/tiff_patches32_mr10  — ch0 (FA marker)
  {ds}_ch2/{condition}/tiff_patches32_mr10  — ch2 (zyxin)

ch1 (pax → {ds}/) and ch3 (act → {ds}_ch3/) already exist.

Run:
    python scripts/generate_patchprep_ch02.py [--dry_run]
    # then: bash scripts/submit_patchprep_ch02.sh
"""

from __future__ import annotations
import argparse
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
ROOT      = "/net/projects/CLS/lding/data/fa_data_analysis"
PATCH_BASE = f"{ROOT}/ae_results/patches/cio_rb"
CONFIG_OUT = Path(__file__).resolve().parent.parent / "config" / "patchprep_ch02_config"
REPO       = "/net/projects/CLS/lding/gitcode/SubCellAE"
PYTHON     = "/home/liyading/miniconda3/bin/python3"
PYTHONPATH = f"{REPO}:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

# ---------------------------------------------------------------------------
# Per-dataset info
DATASETS = {
    "vinc": dict(
        ch0_name="vinc", ch0_scale=5.0, ch0_ch_idx=0,
        ch2_name="zyx",  ch2_scale=4.0, ch2_ch_idx=2,
        conditions={
            "control": "fa_data/other_paxillin/20250311_eGFPZyxin488_Phalloidin405_Vinculin(rb)647_paxillin(m)568/Control",
            "ycomp":   "fa_data/other_paxillin/20250311_eGFPZyxin488_Phalloidin405_Vinculin(rb)647_paxillin(m)568/Ycomp",
        },
    ),
    "pfak": dict(
        ch0_name="pfak", ch0_scale=5.0, ch0_ch_idx=0,
        ch2_name="zyx",  ch2_scale=4.0, ch2_ch_idx=2,
        conditions={
            "control": "fa_data/other_paxillin/20250720_eGFP-Zyxin 488, Phalloidin 405, pFAK (rb) 647, paxillin(m)568/072025/Control",
            "ycomp":   "fa_data/other_paxillin/20250720_eGFP-Zyxin 488, Phalloidin 405, pFAK (rb) 647, paxillin(m)568/072025/Ycomp",
        },
    ),
    "ppax": dict(
        ch0_name="ppax", ch0_scale=5.0, ch0_ch_idx=0,
        ch2_name="zyx",  ch2_scale=4.0, ch2_ch_idx=2,
        conditions={
            "control": "fa_data/other_paxillin/20250721_eGFP-Zyxin 488_Phalloidin405_pPaxy118(rb) 647_Pax(m)568/Control",
            "ycomp":   "fa_data/other_paxillin/20250721_eGFP-Zyxin 488_Phalloidin405_pPaxy118(rb) 647_Pax(m)568/Y-comp",
        },
    ),
    "nih3t3": dict(
        ch0_name="vinc", ch0_scale=5.0, ch0_ch_idx=0,
        ch2_name="zyx",  ch2_scale=4.0, ch2_ch_idx=2,
        conditions={
            "control": "fa_data/other_paxillin/20260227_NIH3T3_ZyxinGFP,Phalloidin405,Vinc_rb647,Pax_m555_reduced_size_AH/Control",
            "ycomp":   "fa_data/other_paxillin/20260227_NIH3T3_ZyxinGFP,Phalloidin405,Vinc_rb647,Pax_m555_reduced_size_AH/YCompound",
        },
    ),
}

# Segmentation params (consistent with existing configs; seg always on paxillin ch1)
SEG = dict(
    seg_ch=1,
    seg_threshold=0.1,
    seg_close_size=11,
    seg_min_size_initial=3,
    seg_min_size_post_close=10,
    seg_min_size_final=30000,
)


def make_patchprep_cfg(ds: str, cond: str, ch_idx: int, ch_name: str, scale: float,
                       img_rel: str, out_suffix: str) -> dict:
    return {
        "paths": {
            "root_folder":     ROOT,
            "image_folder":    f"{ROOT}/{img_rel}",
            "cell_mask_folder": None,
            "patch_output_dir": f"{PATCH_BASE}/{ds}{out_suffix}/{cond}/tiff_patches32_mr10",
            "plot_output_dir":  f"{PATCH_BASE}/{ds}{out_suffix}/{cond}/plot_patches32_mr10",
        },
        "experiment": {
            "condition": cond,
            "major_ch":  ch_idx,
        },
        "input": {
            "file_type": "czi",
            "start_ind": 0,
            "end_ind":   999,
        },
        "patch": {
            "patch_size":   32,
            "mask_ratio":   0.1,
            "pad_size":     64,
            "patch_prefix": cond,
        },
        "preprocessing": {
            "rolling_ball_radius": 20,
        },
        "normalization": {
            "norm_mode": "cell_insideoutside",
        },
        "segmentation": SEG,
        "augmentation": {
            "rand_trans":    False,
            "max_shift_px":  0,
            "rand_rota":     False,
            "max_angle_deg": 0.0,
        },
        "misc": {
            "dpi":           256,
            "debug":         False,
            "log_level":     "INFO",
            "use_timestamp": False,
        },
    }


def generate_all(dry_run: bool = False):
    CONFIG_OUT.mkdir(parents=True, exist_ok=True)

    all_jobs: list[tuple[str, str]] = []  # (config_path, job_name)

    for ds, dinfo in DATASETS.items():
        for cond, img_rel in dinfo["conditions"].items():
            for ch_label, ch_idx, ch_name, scale, suffix in [
                ("ch0", dinfo["ch0_ch_idx"], dinfo["ch0_name"], dinfo["ch0_scale"], "_ch0"),
                ("ch2", dinfo["ch2_ch_idx"], dinfo["ch2_name"], dinfo["ch2_scale"], "_ch2"),
            ]:
                cfg_name = f"{ds}_{cond}_cio_rb_{ch_label}.yaml"
                cfg_path = CONFIG_OUT / cfg_name
                job_name = f"patchprep_{ds}_{cond}_{ch_label}"

                cfg = make_patchprep_cfg(ds, cond, ch_idx, ch_name, scale,
                                         img_rel, suffix)

                hdr = (
                    f"# Dataset   : {ds} ({cond}) — {ch_label} ({ch_name})\n"
                    f"# Condition : {cond}\n"
                    f"# major_ch={ch_idx} ({ch_name}); seg_ch=1 (paxillin) for FA position detection\n"
                )

                if not dry_run:
                    with open(cfg_path, "w") as f:
                        f.write(hdr)
                        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

                all_jobs.append((str(cfg_path), job_name))
                print(f"  {'(dry) ' if dry_run else ''}Config: {cfg_path}")

    print(f"\nTotal patchprep jobs: {len(all_jobs)}")

    submit_path = Path(REPO) / "scripts" / "submit_patchprep_ch02.sh"
    if not dry_run:
        _write_submit(submit_path, all_jobs)
        print(f"Submit script: {submit_path}")
    else:
        print("[dry_run] No files written.")


def _write_submit(path: Path, jobs: list[tuple[str, str]]):
    lines = [
        "#!/usr/bin/env bash",
        "# submit_patchprep_ch02.sh — auto-generated by generate_patchprep_ch02.py",
        "# Extracts ch0 and ch2 patches for all datasets (needed by protein sweep)",
        "#",
        f"# Total jobs: {len(jobs)}",
        "set -eo pipefail",
        "mkdir -p logs/slurm",
        "",
        f"PYTHON={PYTHON}",
        f'export PYTHONPATH="{PYTHONPATH}"',
        f"RUNNER={REPO}/scripts/run_patchprep_from_config.py",
        "",
        f'echo "Submitting {len(jobs)} patchprep jobs (ch0 + ch2)..."',
        "",
    ]
    for cfg_path, job_name in jobs:
        lines += [
            f"sbatch --job-name={job_name} \\",
            f'  --output=logs/slurm/{job_name}_%j.out \\',
            f"  --ntasks=1 --cpus-per-task=4 --mem=16G --time=02:00:00 \\",
            f'  --wrap="$PYTHON $RUNNER {cfg_path}"',
            "",
        ]
    lines.append('echo "All jobs submitted."')
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    generate_all(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
