#!/bin/bash
#SBATCH --job-name=segf3090
#SBATCH --output=/SLURM/home/slurm_g202518690/student251/CVPR/folds/fold_%a/r3090_%A_%a.log
#SBATCH --error=/SLURM/home/slurm_g202518690/student251/CVPR/folds/fold_%a/r3090_%A_%a.err
#SBATCH --partition=RTX3090
#SBATCH --nodelist=jrcai02
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=6 --mem=18G --time=08:00:00
#SBATCH --array=0-4
PY=/SLURM/home/slurm_g202518690/student251/conda_envs/dinov3/bin/python3
export OMP_NUM_THREADS=4 TRANSFORMERS_OFFLINE=1 HF_HOME=/SLURM/home/slurm_g202518690/.cache/huggingface
cd /SLURM/home/slurm_g202518690/student251/CVPR
OUT=/SLURM/home/slurm_g202518690/student251/CVPR/folds/fold_${SLURM_ARRAY_TASK_ID}
mkdir -p $OUT
if [ -f $OUT/best_model.pth ] && grep -q '"epoch": 120' $OUT/history.json 2>/dev/null; then
  echo "Fold ${SLURM_ARRAY_TASK_ID} already complete -> skip"; exit 0
fi
$PY pulse_seg.py --fold $SLURM_ARRAY_TASK_ID --nfolds 5 --epochs 120 --bs 8 --ckpt_dir $OUT
echo "Fold ${SLURM_ARRAY_TASK_ID} done."
