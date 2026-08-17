#!/bin/bash
#SBATCH --job-name=pulse_attn
#SBATCH --output=/SLURM/home/slurm_g202518690/student251/CVPR/logs/attn_%j.log
#SBATCH --error=/SLURM/home/slurm_g202518690/student251/CVPR/logs/attn_%j.err
#SBATCH --partition=RTX3090,A100
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=12G --time=00:30:00

PY=/SLURM/home/slurm_g202518690/student251/conda_envs/dinov3/bin/python3
export TRANSFORMERS_OFFLINE=1
export ACDC_ROOT=/SLURM/home/slurm_g202518690/student251/ACDC/database
cd /SLURM/home/slurm_g202518690/student251/CVPR
mkdir -p logs

$PY generate_attention_maps.py \
    --ckpt folds/fold_4/best_model.pth \
    --out paper_contents/fig_attention_maps.png

echo "ATTENTION MAPS DONE"
