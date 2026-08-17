#!/bin/bash
#SBATCH --job-name=pulse_ens
#SBATCH --output=/SLURM/home/slurm_g202518690/student251/CVPR/logs/ensemble_%j.log
#SBATCH --error=/SLURM/home/slurm_g202518690/student251/CVPR/logs/ensemble_%j.err
#SBATCH --partition=A100
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=40G --time=05:00:00

PY=/SLURM/home/slurm_g202518690/student251/conda_envs/dinov3/bin/python3
export TRANSFORMERS_OFFLINE=1
export ACDC_ROOT=/SLURM/home/slurm_g202518690/student251/ACDC/database
cd /SLURM/home/slurm_g202518690/student251/CVPR
mkdir -p logs

$PY ensemble_eval.py \
    --folds_dir folds \
    --img_size 256 \
    --out diagnosis_features_ensemble.json \
    --data_root /SLURM/home/slurm_g202518690/student251/ACDC/database \
  && $PY diagnosis_classify.py \
    --features diagnosis_features_ensemble.json \
    --seeds 10

echo "ENSEMBLE + DIAGNOSIS DONE"
