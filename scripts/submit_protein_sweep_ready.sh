#!/usr/bin/env bash
# submit_protein_sweep_ready.sh — pax + act jobs (patches already exist)
# Total: 90 jobs
set -eo pipefail
mkdir -p logs/slurm

PYTHON=/home/liyading/miniconda3/bin/python3
RUNNER=/net/projects/CLS/lding/gitcode/SubCellAE/scripts/run_ae_from_config.py

echo "Submitting 90 ready protein-sweep jobs (pax + act)..."

JOB=$(sbatch --parsable \
    --job-name="conae_act_f_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_f_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_f_l1.yaml
echo End: $(date)")
echo "  conae_act_f_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_f_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_f_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_f_mse.yaml
echo End: $(date)")
echo "  conae_act_f_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_f_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_f_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_f_nl1.yaml
echo End: $(date)")
echo "  conae_act_f_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_fn_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_fn_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_fn_l1.yaml
echo End: $(date)")
echo "  conae_act_fn_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_fn_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_fn_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_fn_mse.yaml
echo End: $(date)")
echo "  conae_act_fn_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_fn_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_fn_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_fn_nl1.yaml
echo End: $(date)")
echo "  conae_act_fn_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_fnp_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_fnp_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_fnp_l1.yaml
echo End: $(date)")
echo "  conae_act_fnp_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_fnp_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_fnp_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_fnp_mse.yaml
echo End: $(date)")
echo "  conae_act_fnp_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_fnp_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_fnp_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_fnp_nl1.yaml
echo End: $(date)")
echo "  conae_act_fnp_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_fnpv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_fnpv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_fnpv_l1.yaml
echo End: $(date)")
echo "  conae_act_fnpv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_fnpv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_fnpv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_fnpv_mse.yaml
echo End: $(date)")
echo "  conae_act_fnpv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_fnpv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_fnpv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_fnpv_nl1.yaml
echo End: $(date)")
echo "  conae_act_fnpv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_fnv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_fnv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_fnv_l1.yaml
echo End: $(date)")
echo "  conae_act_fnv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_fnv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_fnv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_fnv_mse.yaml
echo End: $(date)")
echo "  conae_act_fnv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_fnv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_fnv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_fnv_nl1.yaml
echo End: $(date)")
echo "  conae_act_fnv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_fp_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_fp_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_fp_l1.yaml
echo End: $(date)")
echo "  conae_act_fp_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_fp_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_fp_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_fp_mse.yaml
echo End: $(date)")
echo "  conae_act_fp_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_fp_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_fp_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_fp_nl1.yaml
echo End: $(date)")
echo "  conae_act_fp_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_fpv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_fpv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_fpv_l1.yaml
echo End: $(date)")
echo "  conae_act_fpv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_fpv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_fpv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_fpv_mse.yaml
echo End: $(date)")
echo "  conae_act_fpv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_fpv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_fpv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_fpv_nl1.yaml
echo End: $(date)")
echo "  conae_act_fpv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_fv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_fv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_fv_l1.yaml
echo End: $(date)")
echo "  conae_act_fv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_fv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_fv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_fv_mse.yaml
echo End: $(date)")
echo "  conae_act_fv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_fv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_fv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_fv_nl1.yaml
echo End: $(date)")
echo "  conae_act_fv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_n_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_n_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_n_l1.yaml
echo End: $(date)")
echo "  conae_act_n_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_n_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_n_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_n_mse.yaml
echo End: $(date)")
echo "  conae_act_n_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_n_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_n_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_n_nl1.yaml
echo End: $(date)")
echo "  conae_act_n_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_np_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_np_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_np_l1.yaml
echo End: $(date)")
echo "  conae_act_np_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_np_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_np_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_np_mse.yaml
echo End: $(date)")
echo "  conae_act_np_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_np_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_np_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_np_nl1.yaml
echo End: $(date)")
echo "  conae_act_np_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_npv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_npv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_npv_l1.yaml
echo End: $(date)")
echo "  conae_act_npv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_npv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_npv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_npv_mse.yaml
echo End: $(date)")
echo "  conae_act_npv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_npv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_npv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_npv_nl1.yaml
echo End: $(date)")
echo "  conae_act_npv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_nv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_nv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_nv_l1.yaml
echo End: $(date)")
echo "  conae_act_nv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_nv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_nv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_nv_mse.yaml
echo End: $(date)")
echo "  conae_act_nv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_nv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_nv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_nv_nl1.yaml
echo End: $(date)")
echo "  conae_act_nv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_p_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_p_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_p_l1.yaml
echo End: $(date)")
echo "  conae_act_p_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_p_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_p_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_p_mse.yaml
echo End: $(date)")
echo "  conae_act_p_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_p_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_p_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_p_nl1.yaml
echo End: $(date)")
echo "  conae_act_p_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_pv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_pv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_pv_l1.yaml
echo End: $(date)")
echo "  conae_act_pv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_pv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_pv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_pv_mse.yaml
echo End: $(date)")
echo "  conae_act_pv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_pv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_pv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_pv_nl1.yaml
echo End: $(date)")
echo "  conae_act_pv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_v_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_v_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_v_l1.yaml
echo End: $(date)")
echo "  conae_act_v_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_v_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_v_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_v_mse.yaml
echo End: $(date)")
echo "  conae_act_v_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_act_v_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_act_v_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_act_v_nl1.yaml
echo End: $(date)")
echo "  conae_act_v_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_f_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_f_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_f_l1.yaml
echo End: $(date)")
echo "  conae_pax_f_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_f_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_f_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_f_mse.yaml
echo End: $(date)")
echo "  conae_pax_f_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_f_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_f_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_f_nl1.yaml
echo End: $(date)")
echo "  conae_pax_f_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_fn_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_fn_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_fn_l1.yaml
echo End: $(date)")
echo "  conae_pax_fn_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_fn_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_fn_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_fn_mse.yaml
echo End: $(date)")
echo "  conae_pax_fn_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_fn_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_fn_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_fn_nl1.yaml
echo End: $(date)")
echo "  conae_pax_fn_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_fnp_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_fnp_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_fnp_l1.yaml
echo End: $(date)")
echo "  conae_pax_fnp_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_fnp_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_fnp_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_fnp_mse.yaml
echo End: $(date)")
echo "  conae_pax_fnp_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_fnp_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_fnp_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_fnp_nl1.yaml
echo End: $(date)")
echo "  conae_pax_fnp_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_fnpv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_fnpv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_fnpv_l1.yaml
echo End: $(date)")
echo "  conae_pax_fnpv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_fnpv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_fnpv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_fnpv_mse.yaml
echo End: $(date)")
echo "  conae_pax_fnpv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_fnpv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_fnpv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_fnpv_nl1.yaml
echo End: $(date)")
echo "  conae_pax_fnpv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_fnv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_fnv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_fnv_l1.yaml
echo End: $(date)")
echo "  conae_pax_fnv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_fnv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_fnv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_fnv_mse.yaml
echo End: $(date)")
echo "  conae_pax_fnv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_fnv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_fnv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_fnv_nl1.yaml
echo End: $(date)")
echo "  conae_pax_fnv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_fp_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_fp_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_fp_l1.yaml
echo End: $(date)")
echo "  conae_pax_fp_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_fp_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_fp_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_fp_mse.yaml
echo End: $(date)")
echo "  conae_pax_fp_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_fp_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_fp_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_fp_nl1.yaml
echo End: $(date)")
echo "  conae_pax_fp_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_fpv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_fpv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_fpv_l1.yaml
echo End: $(date)")
echo "  conae_pax_fpv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_fpv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_fpv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_fpv_mse.yaml
echo End: $(date)")
echo "  conae_pax_fpv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_fpv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_fpv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_fpv_nl1.yaml
echo End: $(date)")
echo "  conae_pax_fpv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_fv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_fv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_fv_l1.yaml
echo End: $(date)")
echo "  conae_pax_fv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_fv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_fv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_fv_mse.yaml
echo End: $(date)")
echo "  conae_pax_fv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_fv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_fv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_fv_nl1.yaml
echo End: $(date)")
echo "  conae_pax_fv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_n_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_n_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_n_l1.yaml
echo End: $(date)")
echo "  conae_pax_n_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_n_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_n_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_n_mse.yaml
echo End: $(date)")
echo "  conae_pax_n_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_n_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_n_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_n_nl1.yaml
echo End: $(date)")
echo "  conae_pax_n_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_np_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_np_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_np_l1.yaml
echo End: $(date)")
echo "  conae_pax_np_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_np_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_np_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_np_mse.yaml
echo End: $(date)")
echo "  conae_pax_np_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_np_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_np_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_np_nl1.yaml
echo End: $(date)")
echo "  conae_pax_np_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_npv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_npv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_npv_l1.yaml
echo End: $(date)")
echo "  conae_pax_npv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_npv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_npv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_npv_mse.yaml
echo End: $(date)")
echo "  conae_pax_npv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_npv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_npv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_npv_nl1.yaml
echo End: $(date)")
echo "  conae_pax_npv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_nv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_nv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_nv_l1.yaml
echo End: $(date)")
echo "  conae_pax_nv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_nv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_nv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_nv_mse.yaml
echo End: $(date)")
echo "  conae_pax_nv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_nv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_nv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_nv_nl1.yaml
echo End: $(date)")
echo "  conae_pax_nv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_p_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_p_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_p_l1.yaml
echo End: $(date)")
echo "  conae_pax_p_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_p_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_p_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_p_mse.yaml
echo End: $(date)")
echo "  conae_pax_p_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_p_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_p_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_p_nl1.yaml
echo End: $(date)")
echo "  conae_pax_p_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_pv_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_pv_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_pv_l1.yaml
echo End: $(date)")
echo "  conae_pax_pv_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_pv_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_pv_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_pv_mse.yaml
echo End: $(date)")
echo "  conae_pax_pv_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_pv_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_pv_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_pv_nl1.yaml
echo End: $(date)")
echo "  conae_pax_pv_nl1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_v_l1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_v_l1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_v_l1.yaml
echo End: $(date)")
echo "  conae_pax_v_l1 -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_v_mse" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_v_mse_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_v_mse.yaml
echo End: $(date)")
echo "  conae_pax_v_mse -> job $JOB"

JOB=$(sbatch --parsable \
    --job-name="conae_pax_v_nl1" \
    --partition=general \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=08:00:00 \
    --output="logs/slurm/conae_pax_v_nl1_%j.out" \
    --wrap="exec 2>&1
export PYTHONPATH='/net/projects/CLS/lding/gitcode/SubCellAE:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages'
echo Node: $(hostname)
echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo Start: $(date)
$PYTHON $RUNNER config/protein_sweep/set1_single_ch/conae_pax_v_nl1.yaml
echo End: $(date)")
echo "  conae_pax_v_nl1 -> job $JOB"

echo "All ready jobs submitted."
