#!/bin/bash
#SBATCH --job-name=camus_RTX
#SBATCH --output=/SLURM/home/slurm_g202518690/student251/CVPR/logs/camus_RTX_%j.out
#SBATCH --error=/SLURM/home/slurm_g202518690/student251/CVPR/logs/camus_RTX_%j.err
#SBATCH --partition=RTX3090
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=06:00:00
PY=/SLURM/home/slurm_g202518690/student251/conda_envs/dinov3/bin/python3
export TRANSFORMERS_OFFLINE=1
export CAMUS_ROOT=/SLURM/home/slurm_g202518690/student251/CAMUS/database_nifti
cd /SLURM/home/slurm_g202518690/student251/CVPR
mkdir -p logs
$PY eval_camus_fewshot.py --ckpt_dir folds --view 2CH
echo "CAMUS FEWSHOT DONE"
