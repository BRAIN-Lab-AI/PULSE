#!/bin/bash
#SBATCH --job-name=abl_vanilla_dino
#SBATCH --partition=A100
#SBATCH --account=grp_muzammilbehzad
#SBATCH --qos=overrideJobsPA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/SLURM/home/slurm_g202518690/student251/CVPR/vanilla_dino_%j.log

cd /SLURM/home/slurm_g202518690/student251/CVPR
PYTHON=/SLURM/home/slurm_g202518690/student251/conda_envs/dinov3/bin/python3
DATA=/SLURM/home/slurm_g202518690/student251/ACDC/database
export TRANSFORMERS_OFFLINE=1
export ACDC_ROOT=${DATA}

$PYTHON ablation_vanilla_dino.py \
    --fold 0 --nfolds 5 --epochs 120 --bs 16 \
    --data_root ${DATA} \
    --ckpt_dir ablation_ckpts/vanilla_dino
