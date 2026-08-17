#!/bin/bash
#SBATCH --job-name=pulse_camus
#SBATCH --output=/SLURM/home/slurm_g202518690/student251/CVPR/logs/camus_%j.log
#SBATCH --error=/SLURM/home/slurm_g202518690/student251/CVPR/logs/camus_%j.err
#SBATCH --partition=A100
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=40G --time=04:00:00

PY=/SLURM/home/slurm_g202518690/student251/conda_envs/dinov3/bin/python3
export TRANSFORMERS_OFFLINE=1
export CAMUS_ROOT=/SLURM/home/slurm_g202518690/student251/CAMUS/database_nifti
cd /SLURM/home/slurm_g202518690/student251/CVPR
mkdir -p logs

$PY eval_camus_fewshot.py \
    --camus_root $CAMUS_ROOT \
    --ckpt_dir folds \
    --view 2CH \
    --epochs 40 \
    --lr 5e-5

echo "CAMUS FEW-SHOT DONE"
