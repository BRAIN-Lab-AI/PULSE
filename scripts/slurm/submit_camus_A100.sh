#!/bin/bash
#SBATCH --job-name=camus_A100
#SBATCH --output=/SLURM/home/slurm_g202518690/student251/CVPR/logs/camus_A100_%j.out
#SBATCH --error=/SLURM/home/slurm_g202518690/student251/CVPR/logs/camus_A100_%j.err
#SBATCH --partition=A100
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=40G --time=06:00:00
PY=/SLURM/home/slurm_g202518690/student251/conda_envs/dinov3/bin/python3
export TRANSFORMERS_OFFLINE=1
export CAMUS_ROOT=/SLURM/home/slurm_g202518690/student251/CAMUS/database_nifti
cd /SLURM/home/slurm_g202518690/student251/CVPR
mkdir -p logs
$PY eval_camus_fewshot.py --ckpt_dir folds --view 2CH
echo "CAMUS FEWSHOT DONE"
