#!/bin/bash
#SBATCH --job-name=eval_tta
#SBATCH --partition=RTX3090
#SBATCH --account=grp_muzammilbehzad
#SBATCH --qos=overrideJobsPA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/SLURM/home/slurm_g202518690/student251/CVPR/eval_tta_%j.log

cd /SLURM/home/slurm_g202518690/student251/CVPR
PYTHON=/SLURM/home/slurm_g202518690/.conda/envs/DETRIS/bin/python

$PYTHON eval_tta_contribution.py \
    --data_root /SLURM/home/slurm_g202518690/student251/ACDC/database \
    --folds_dir /SLURM/home/slurm_g202518690/student251/CVPR/folds
