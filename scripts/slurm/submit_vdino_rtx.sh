#!/bin/bash
# ============================================================
# submit_vdino_rtx.sh — Move pending ablation + CAMUS jobs
# from A100 (hit 5-job limit) to RTX3090 (separate QoS/limit)
#
# Cancels: pending A100 jobs 14895-14902 (8 ablations + CAMUS)
#          and the stale ablation eval job 14906
# Resubmits: all 8 ablations + CAMUS to RTX3090
# Fixes:  ablation eval dependency with new RTX job IDs
#
# Usage:  bash submit_vdino_rtx.sh
# ============================================================

set -euo pipefail

BASE=/SLURM/home/slurm_g202518690/student251/CVPR
TRAIN_PY=/SLURM/home/slurm_g202518690/student251/conda_envs/dinov3/bin/python3
EVAL_PY=/SLURM/home/slurm_g202518690/.conda/envs/DETRIS/bin/python
DATA=/SLURM/home/slurm_g202518690/student251/ACDC/database
CAMUS_ROOT=/SLURM/home/slurm_g202518690/student251/CAMUS/database_nifti

# A100 fold training jobs that are STILL RUNNING (keep them)
# 14890 vdino_fold1, 14891 vdino_fold2, 14892 vdino_fold3,
# 14893 vdino_fold4 — these stay on A100

echo "=== Cancelling pending A100 jobs (ablations + CAMUS + stale eval) ==="
# Pending A100 ablation training jobs
for JID in 14895 14896 14897 14898 14899 14900 14901; do
    scancel ${JID} 2>/dev/null && echo "  cancelled ${JID}" || echo "  ${JID} already gone"
done
# CAMUS (was pending on A100)
scancel 14902 2>/dev/null && echo "  cancelled 14902 (camus)" || true
# Stale ablation eval (depended on now-cancelled A100 job IDs)
scancel 14906 2>/dev/null && echo "  cancelled 14906 (abl eval)" || true

echo ""
echo "=== Submitting ablations + CAMUS to RTX3090 (separate job limit) ==="

# Common RTX flags — no overrideJobsPA QoS needed
RTX="--parsable \
     --partition=RTX3090 \
     --account=grp_muzammilbehzad \
     --gres=gpu:1 --cpus-per-task=8 --mem=40G --time=12:00:00"

TENV="export TRANSFORMERS_OFFLINE=1 ACDC_ROOT='${DATA}'"

mkdir -p "${BASE}/logs"

ABL_NAMES=(no_bnd no_cur aug_none aug_weak base_dpt freeze4 freeze6)

ABL_IDS_RTX=()
for NAME in "${ABL_NAMES[@]}"; do
    JID=$(sbatch ${RTX} \
        --job-name=vdino_${NAME} \
        --output="${BASE}/logs/vdino_${NAME}_%j.log" \
        --wrap="cd '${BASE}' && ${TENV} && ${TRAIN_PY} vdino_abl_${NAME}.py \
                --fold 0 --nfolds 5 --epochs 120 --bs 16 \
                --data_root '${DATA}' \
                --ckpt_dir vdino_ablation_ckpts/${NAME}")
    ABL_IDS_RTX+=("${JID}")
    echo "  abl/${NAME}   -> RTX job ${JID}"
done

# no_ds is already running on A100 (14894) — add its ID to the dependency list
# so the ablation eval waits for it too
A100_NO_DS_JID=14894

echo ""
echo "=== Submitting CAMUS eval to RTX3090 (fold_0 ready) ==="
CAMUS_JID=$(sbatch ${RTX} \
    --job-name=vdino_camus \
    --mem=32G --time=06:00:00 \
    --output="${BASE}/logs/vdino_camus_%j.log" \
    --wrap="cd '${BASE}' && export TRANSFORMERS_OFFLINE=1 && \
            ${EVAL_PY} eval_camus_vdino.py \
            --ckpt_dir folds_vdino --camus_root '${CAMUS_ROOT}'")
echo "  CAMUS eval   -> RTX job ${CAMUS_JID}"

echo ""
echo "=== Resubmitting ablation eval with corrected dependencies ==="

# Ablation eval must wait for:
#   - A100 fold jobs:   14890 14891 14892 14893 (fold1-4, still running)
#   - A100 no_ds job:   14894 (still running)
#   - RTX ablation jobs: all ABL_IDS_RTX
A100_FOLDS="14890:14891:14892:14893"
ALL_ABL_DEP="afterok:${A100_FOLDS}:${A100_NO_DS_JID}:$(IFS=:; echo "${ABL_IDS_RTX[*]}")"

ABL_EVAL_JID=$(sbatch --parsable \
    --partition=RTX3090 \
    --account=grp_muzammilbehzad \
    --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=06:00:00 \
    --job-name=vdino_eval_abl \
    --dependency="${ALL_ABL_DEP}" \
    --output="${BASE}/logs/vdino_eval_abl_%j.log" \
    --wrap="cd '${BASE}' && export TRANSFORMERS_OFFLINE=1 && \
            ${EVAL_PY} eval_vdino_final.py \
            --data_root '${DATA}' \
            --folds_dir folds_vdino \
            --abl_dir vdino_ablation_ckpts")
echo "  Ablation eval -> RTX job ${ABL_EVAL_JID}  (waits for all folds + all ablations)"

echo ""
echo "==================================================================="
echo " SUMMARY"
echo "==================================================================="
echo ""
echo "  Still running on A100:"
echo "    vdino_fold1 (14890), vdino_fold2 (14891)"
echo "    vdino_fold3 (14892), vdino_fold4 (14893)"
echo "    vdino_no_ds (14894)"
echo ""
printf "  RTX3090 ablation jobs : %s\n" "${ABL_IDS_RTX[*]}"
printf "  RTX3090 CAMUS eval    : %s\n" "${CAMUS_JID}"
printf "  RTX3090 ablation eval : %s\n" "${ABL_EVAL_JID}"
echo ""
echo "  Eval jobs already queued on A100 (unchanged):"
echo "    vdino_eval_acdc (14903), vdino_eval_mnm (14904)"
echo "    vdino_eval_sb   (14905)  — all depend on fold1-4"
echo ""
echo "  Monitor : squeue -u \$USER"
echo "  Note: RTX3090 is slower than A100 (~2x longer per job)"
echo "        but gives you 7 more parallel slots right now"
