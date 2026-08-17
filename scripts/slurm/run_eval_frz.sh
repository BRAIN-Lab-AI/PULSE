#!/bin/bash
#SBATCH --job-name=eval_frz
#SBATCH --partition=RTX3090
#SBATCH --account=grp_muzammilbehzad
#SBATCH --qos=overrideJobsPA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/SLURM/home/slurm_g202518690/student251/CVPR/eval_frz_%j.log

cd /SLURM/home/slurm_g202518690/student251/CVPR
PYTHON=/SLURM/home/slurm_g202518690/.conda/envs/DETRIS/bin/python

$PYTHON eval_new_ckpts.py \
    --data_root /SLURM/home/slurm_g202518690/student251/ACDC/database \
    --ablation_dir /SLURM/home/slurm_g202518690/student251/CVPR/ablation_ckpts
