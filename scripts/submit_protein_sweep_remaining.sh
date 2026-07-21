#!/usr/bin/env bash
# submit_protein_sweep_remaining.sh — zyx, vinc/pfak/ppax ch0, 4ch, 3ch jobs
# Total: 120 jobs
set -eo pipefail
mkdir -p logs/slurm

PYTHON=/home/liyading/miniconda3/bin/python3
RUNNER=/net/projects/CLS/lding/gitcode/SubCellAE/scripts/run_ae_from_config.py

echo "Submitting 120 remaining protein-sweep jobs..."

JOB=$(sbatch --parsable \
    --job-name="conae_pfak_f_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pfak_f_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pfak_f_l1.yaml
echo End: $(date)")
echo "  conae_pfak_f_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pfak_f_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pfak_f_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pfak_f_mse.yaml
echo End: $(date)")
echo "  conae_pfak_f_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pfak_f_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pfak_f_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pfak_f_nl1.yaml
echo End: $(date)")
echo "  conae_pfak_f_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_ppax_p_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_ppax_p_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_ppax_p_l1.yaml
echo End: $(date)")
echo "  conae_ppax_p_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_ppax_p_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_ppax_p_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_ppax_p_mse.yaml
echo End: $(date)")
echo "  conae_ppax_p_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_ppax_p_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_ppax_p_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_ppax_p_nl1.yaml
echo End: $(date)")
echo "  conae_ppax_p_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_vinc_n_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_vinc_n_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_vinc_n_l1.yaml
echo End: $(date)")
echo "  conae_vinc_n_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_vinc_n_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_vinc_n_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_vinc_n_mse.yaml
echo End: $(date)")
echo "  conae_vinc_n_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_vinc_n_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_vinc_n_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_vinc_n_nl1.yaml
echo End: $(date)")
echo "  conae_vinc_n_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_vinc_nv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_vinc_nv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_vinc_nv_l1.yaml
echo End: $(date)")
echo "  conae_vinc_nv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_vinc_nv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_vinc_nv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_vinc_nv_mse.yaml
echo End: $(date)")
echo "  conae_vinc_nv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_vinc_nv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_vinc_nv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_vinc_nv_nl1.yaml
echo End: $(date)")
echo "  conae_vinc_nv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_vinc_v_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_vinc_v_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_vinc_v_l1.yaml
echo End: $(date)")
echo "  conae_vinc_v_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_vinc_v_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_vinc_v_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_vinc_v_mse.yaml
echo End: $(date)")
echo "  conae_vinc_v_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_vinc_v_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_vinc_v_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_vinc_v_nl1.yaml
echo End: $(date)")
echo "  conae_vinc_v_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_f_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_f_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_f_l1.yaml
echo End: $(date)")
echo "  conae_zyx_f_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_f_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_f_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_f_mse.yaml
echo End: $(date)")
echo "  conae_zyx_f_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_f_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_f_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_f_nl1.yaml
echo End: $(date)")
echo "  conae_zyx_f_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_fn_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_fn_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_fn_l1.yaml
echo End: $(date)")
echo "  conae_zyx_fn_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_fn_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_fn_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_fn_mse.yaml
echo End: $(date)")
echo "  conae_zyx_fn_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_fn_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_fn_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_fn_nl1.yaml
echo End: $(date)")
echo "  conae_zyx_fn_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_fnp_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_fnp_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_fnp_l1.yaml
echo End: $(date)")
echo "  conae_zyx_fnp_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_fnp_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_fnp_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_fnp_mse.yaml
echo End: $(date)")
echo "  conae_zyx_fnp_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_fnp_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_fnp_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_fnp_nl1.yaml
echo End: $(date)")
echo "  conae_zyx_fnp_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_fnpv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_fnpv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_fnpv_l1.yaml
echo End: $(date)")
echo "  conae_zyx_fnpv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_fnpv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_fnpv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_fnpv_mse.yaml
echo End: $(date)")
echo "  conae_zyx_fnpv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_fnpv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_fnpv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_fnpv_nl1.yaml
echo End: $(date)")
echo "  conae_zyx_fnpv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_fnv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_fnv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_fnv_l1.yaml
echo End: $(date)")
echo "  conae_zyx_fnv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_fnv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_fnv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_fnv_mse.yaml
echo End: $(date)")
echo "  conae_zyx_fnv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_fnv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_fnv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_fnv_nl1.yaml
echo End: $(date)")
echo "  conae_zyx_fnv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_fp_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_fp_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_fp_l1.yaml
echo End: $(date)")
echo "  conae_zyx_fp_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_fp_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_fp_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_fp_mse.yaml
echo End: $(date)")
echo "  conae_zyx_fp_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_fp_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_fp_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_fp_nl1.yaml
echo End: $(date)")
echo "  conae_zyx_fp_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_fpv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_fpv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_fpv_l1.yaml
echo End: $(date)")
echo "  conae_zyx_fpv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_fpv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_fpv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_fpv_mse.yaml
echo End: $(date)")
echo "  conae_zyx_fpv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_fpv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_fpv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_fpv_nl1.yaml
echo End: $(date)")
echo "  conae_zyx_fpv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_fv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_fv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_fv_l1.yaml
echo End: $(date)")
echo "  conae_zyx_fv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_fv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_fv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_fv_mse.yaml
echo End: $(date)")
echo "  conae_zyx_fv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_fv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_fv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_fv_nl1.yaml
echo End: $(date)")
echo "  conae_zyx_fv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_n_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_n_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_n_l1.yaml
echo End: $(date)")
echo "  conae_zyx_n_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_n_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_n_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_n_mse.yaml
echo End: $(date)")
echo "  conae_zyx_n_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_n_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_n_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_n_nl1.yaml
echo End: $(date)")
echo "  conae_zyx_n_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_np_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_np_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_np_l1.yaml
echo End: $(date)")
echo "  conae_zyx_np_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_np_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_np_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_np_mse.yaml
echo End: $(date)")
echo "  conae_zyx_np_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_np_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_np_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_np_nl1.yaml
echo End: $(date)")
echo "  conae_zyx_np_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_npv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_npv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_npv_l1.yaml
echo End: $(date)")
echo "  conae_zyx_npv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_npv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_npv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_npv_mse.yaml
echo End: $(date)")
echo "  conae_zyx_npv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_npv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_npv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_npv_nl1.yaml
echo End: $(date)")
echo "  conae_zyx_npv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_nv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_nv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_nv_l1.yaml
echo End: $(date)")
echo "  conae_zyx_nv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_nv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_nv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_nv_mse.yaml
echo End: $(date)")
echo "  conae_zyx_nv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_nv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_nv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_nv_nl1.yaml
echo End: $(date)")
echo "  conae_zyx_nv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_p_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_p_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_p_l1.yaml
echo End: $(date)")
echo "  conae_zyx_p_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_p_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_p_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_p_mse.yaml
echo End: $(date)")
echo "  conae_zyx_p_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_p_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_p_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_p_nl1.yaml
echo End: $(date)")
echo "  conae_zyx_p_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_pv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_pv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_pv_l1.yaml
echo End: $(date)")
echo "  conae_zyx_pv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_pv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_pv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_pv_mse.yaml
echo End: $(date)")
echo "  conae_zyx_pv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_pv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_pv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_pv_nl1.yaml
echo End: $(date)")
echo "  conae_zyx_pv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_v_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_v_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_v_l1.yaml
echo End: $(date)")
echo "  conae_zyx_v_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_v_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_v_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_v_mse.yaml
echo End: $(date)")
echo "  conae_zyx_v_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_zyx_v_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_zyx_v_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_zyx_v_nl1.yaml
echo End: $(date)")
echo "  conae_zyx_v_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_4ch_pfak_f_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_4ch_pfak_f_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set2_4ch/conae_4ch_pfak_f_l1.yaml
echo End: $(date)")
echo "  conae_4ch_pfak_f_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_4ch_pfak_f_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_4ch_pfak_f_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set2_4ch/conae_4ch_pfak_f_mse.yaml
echo End: $(date)")
echo "  conae_4ch_pfak_f_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_4ch_pfak_f_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_4ch_pfak_f_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set2_4ch/conae_4ch_pfak_f_nl1.yaml
echo End: $(date)")
echo "  conae_4ch_pfak_f_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_4ch_ppax_p_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_4ch_ppax_p_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set2_4ch/conae_4ch_ppax_p_l1.yaml
echo End: $(date)")
echo "  conae_4ch_ppax_p_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_4ch_ppax_p_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_4ch_ppax_p_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set2_4ch/conae_4ch_ppax_p_mse.yaml
echo End: $(date)")
echo "  conae_4ch_ppax_p_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_4ch_ppax_p_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_4ch_ppax_p_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set2_4ch/conae_4ch_ppax_p_nl1.yaml
echo End: $(date)")
echo "  conae_4ch_ppax_p_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_4ch_vinc_n_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_4ch_vinc_n_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set2_4ch/conae_4ch_vinc_n_l1.yaml
echo End: $(date)")
echo "  conae_4ch_vinc_n_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_4ch_vinc_n_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_4ch_vinc_n_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set2_4ch/conae_4ch_vinc_n_mse.yaml
echo End: $(date)")
echo "  conae_4ch_vinc_n_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_4ch_vinc_n_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_4ch_vinc_n_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set2_4ch/conae_4ch_vinc_n_nl1.yaml
echo End: $(date)")
echo "  conae_4ch_vinc_n_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_4ch_vinc_nv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_4ch_vinc_nv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set2_4ch/conae_4ch_vinc_nv_l1.yaml
echo End: $(date)")
echo "  conae_4ch_vinc_nv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_4ch_vinc_nv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_4ch_vinc_nv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set2_4ch/conae_4ch_vinc_nv_mse.yaml
echo End: $(date)")
echo "  conae_4ch_vinc_nv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_4ch_vinc_nv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_4ch_vinc_nv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set2_4ch/conae_4ch_vinc_nv_nl1.yaml
echo End: $(date)")
echo "  conae_4ch_vinc_nv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_4ch_vinc_v_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_4ch_vinc_v_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set2_4ch/conae_4ch_vinc_v_l1.yaml
echo End: $(date)")
echo "  conae_4ch_vinc_v_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_4ch_vinc_v_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_4ch_vinc_v_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set2_4ch/conae_4ch_vinc_v_mse.yaml
echo End: $(date)")
echo "  conae_4ch_vinc_v_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_4ch_vinc_v_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_4ch_vinc_v_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set2_4ch/conae_4ch_vinc_v_nl1.yaml
echo End: $(date)")
echo "  conae_4ch_vinc_v_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_f_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_f_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_f_l1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_f_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_f_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_f_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_f_mse.yaml
echo End: $(date)")
echo "  conae_3ch_pza_f_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_f_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_f_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_f_nl1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_f_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_fn_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_fn_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_fn_l1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_fn_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_fn_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_fn_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_fn_mse.yaml
echo End: $(date)")
echo "  conae_3ch_pza_fn_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_fn_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_fn_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_fn_nl1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_fn_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_fnp_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_fnp_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_fnp_l1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_fnp_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_fnp_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_fnp_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_fnp_mse.yaml
echo End: $(date)")
echo "  conae_3ch_pza_fnp_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_fnp_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_fnp_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_fnp_nl1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_fnp_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_fnpv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_fnpv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_fnpv_l1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_fnpv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_fnpv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_fnpv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_fnpv_mse.yaml
echo End: $(date)")
echo "  conae_3ch_pza_fnpv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_fnpv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_fnpv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_fnpv_nl1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_fnpv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_fnv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_fnv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_fnv_l1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_fnv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_fnv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_fnv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_fnv_mse.yaml
echo End: $(date)")
echo "  conae_3ch_pza_fnv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_fnv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_fnv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_fnv_nl1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_fnv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_fp_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_fp_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_fp_l1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_fp_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_fp_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_fp_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_fp_mse.yaml
echo End: $(date)")
echo "  conae_3ch_pza_fp_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_fp_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_fp_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_fp_nl1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_fp_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_fpv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_fpv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_fpv_l1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_fpv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_fpv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_fpv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_fpv_mse.yaml
echo End: $(date)")
echo "  conae_3ch_pza_fpv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_fpv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_fpv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_fpv_nl1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_fpv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_fv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_fv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_fv_l1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_fv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_fv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_fv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_fv_mse.yaml
echo End: $(date)")
echo "  conae_3ch_pza_fv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_fv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_fv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_fv_nl1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_fv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_n_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_n_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_n_l1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_n_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_n_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_n_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_n_mse.yaml
echo End: $(date)")
echo "  conae_3ch_pza_n_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_n_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_n_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_n_nl1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_n_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_np_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_np_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_np_l1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_np_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_np_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_np_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_np_mse.yaml
echo End: $(date)")
echo "  conae_3ch_pza_np_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_np_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_np_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_np_nl1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_np_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_npv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_npv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_npv_l1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_npv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_npv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_npv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_npv_mse.yaml
echo End: $(date)")
echo "  conae_3ch_pza_npv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_npv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_npv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_npv_nl1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_npv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_nv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_nv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_nv_l1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_nv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_nv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_nv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_nv_mse.yaml
echo End: $(date)")
echo "  conae_3ch_pza_nv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_nv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_nv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_nv_nl1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_nv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_p_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_p_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_p_l1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_p_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_p_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_p_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_p_mse.yaml
echo End: $(date)")
echo "  conae_3ch_pza_p_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_p_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_p_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_p_nl1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_p_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_pv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_pv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_pv_l1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_pv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_pv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_pv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_pv_mse.yaml
echo End: $(date)")
echo "  conae_3ch_pza_pv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_pv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_pv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_pv_nl1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_pv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_v_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_v_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_v_l1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_v_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_v_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_v_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_v_mse.yaml
echo End: $(date)")
echo "  conae_3ch_pza_v_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_3ch_pza_v_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_3ch_pza_v_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set3_3ch/conae_3ch_pza_v_nl1.yaml
echo End: $(date)")
echo "  conae_3ch_pza_v_nl1 -> job $JOB"

echo "All remaining jobs submitted."
