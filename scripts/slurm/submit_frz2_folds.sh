#!/bin/bash
# Submit PULSE folds 1-4 with freeze_2blk — all on A100 in parallel.
# frz2 fold-0 already done at ablation_ckpts/freeze_2blk/ (87.8% mean Dice).
# These 4 jobs + fold-0 form the new full 5-fold PULSE ensemble.

BASE=/SLURM/home/slurm_g202518690/student251/CVPR
PYTHON=/SLURM/home/slurm_g202518690/student251/conda_envs/dinov3/bin/python3
DATA=/SLURM/home/slurm_g202518690/student251/ACDC/database
export TRANSFORMERS_OFFLINE=1
export ACDC_ROOT=${DATA}

echo "=== Submitting PULSE frz2 folds 1-4 on A100 (parallel) ==="

for FOLD in 1 2 3 4; do
  JOB=$(sbatch --parsable \
    --job-name=pulse_frz2_f${FOLD} \
    --partition=A100 \
    --account=grp_muzammilbehzad \
    --qos=overrideJobsPA \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=48G \
    --time=06:00:00 \
    --output=${BASE}/logs/frz2_fold${FOLD}_%j.log \
    --wrap="cd ${BASE} && ${PYTHON} ablation_freeze_2blk.py \
      --fold ${FOLD} --nfolds 5 --epochs 120 --bs 16 \
      --data_root ${DATA} \
      --ckpt_dir folds_frz2/fold_${FOLD}")
  echo "  fold ${FOLD} → job ${JOB}"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    ${BASE}/logs/frz2_fold*"
echo "Output:  ${BASE}/folds_frz2/fold_{1..4}/best_model.pth"
echo ""
echo "After all 4 finish, run 5-fold eval with:"
echo "  python ensemble_eval.py --folds_dir folds_frz2 --fold0 ablation_ckpts/freeze_2blk"
