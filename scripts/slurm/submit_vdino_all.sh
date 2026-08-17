#!/bin/bash
# ============================================================
# submit_vdino_all.sh — Full vanilla DINOv2 PULSE pipeline
#
# Submits ALL jobs to SLURM simultaneously. Jobs queue and run
# as GPUs free up. Eval jobs use --dependency=afterok so they
# auto-start when their training deps finish — no manual steps.
#
# Usage:
#   cd /SLURM/home/slurm_g202518690/student251/CVPR
#   bash submit_vdino_all.sh
# Monitor:
#   squeue -u $USER
#   tail -f logs/vdino_*_%j.log
# ============================================================

set -euo pipefail

BASE=/SLURM/home/slurm_g202518690/student251/CVPR
TRAIN_PY=/SLURM/home/slurm_g202518690/student251/conda_envs/dinov3/bin/python3
EVAL_PY=/SLURM/home/slurm_g202518690/.conda/envs/DETRIS/bin/python
DATA=/SLURM/home/slurm_g202518690/student251/ACDC/database
MNM_ROOT=/SLURM/home/slurm_g202518690/student251/MnMs_Extracted/MnM2/dataset
# Note: SB_ROOT has a space — we pass it via env var, NOT as a CLI arg
SB_ROOT="/SLURM/home/slurm_g202518690/student251/SUNNYBROOK_EXTRACTED/Cardiac MRI"
CAMUS_ROOT=/SLURM/home/slurm_g202518690/student251/CAMUS/database_nifti

mkdir -p "${BASE}/logs"
mkdir -p "${BASE}/folds_vdino/fold_0"
mkdir -p "${BASE}/vdino_ablation_ckpts"

# ── Symlink fold-0 (already trained vanilla DINOv2, best_dice=0.9137) ─────────
FOLD0_CKPT="${BASE}/ablation_ckpts/vanilla_dino/best_model.pth"
FOLD0_LINK="${BASE}/folds_vdino/fold_0/best_model.pth"
if [ ! -e "${FOLD0_LINK}" ]; then
    ln -sf "${FOLD0_CKPT}" "${FOLD0_LINK}"
    echo "[setup] linked: folds_vdino/fold_0/best_model.pth -> ${FOLD0_CKPT}"
else
    echo "[setup] fold_0 link already exists"
fi

# ── SLURM common flags ────────────────────────────────────────────────────────
PART=A100
ACCT=grp_muzammilbehzad
QOS=overrideJobsPA

sbatch_train() {
    # sbatch_train <job-name> <log-suffix> <wrap-cmd>
    sbatch --parsable \
        --partition="${PART}" --account="${ACCT}" --qos="${QOS}" \
        --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=08:00:00 \
        --job-name="$1" \
        --output="${BASE}/logs/${2}_%j.log" \
        --wrap="$3"
}

sbatch_eval() {
    # sbatch_eval <job-name> <log-suffix> <dependency> <wrap-cmd>
    sbatch --parsable \
        --partition="${PART}" --account="${ACCT}" --qos="${QOS}" \
        --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=06:00:00 \
        --job-name="$1" \
        --output="${BASE}/logs/${2}_%j.log" \
        --dependency="$3" \
        --wrap="$4"
}

TENV="export TRANSFORMERS_OFFLINE=1 ACDC_ROOT='${DATA}'"

echo ""
echo "====================================================================="
echo " Phase 1 — Training: 4 main folds + 8 ablation variants"
echo "====================================================================="

# ── Main folds 1–4 (fold 0 already done: 88.6%) ──────────────────────────────
FOLD_IDS=()
for FOLD in 1 2 3 4; do
    JID=$(sbatch_train "vdino_fold${FOLD}" "vdino_fold${FOLD}" \
        "cd '${BASE}' && ${TENV} && ${TRAIN_PY} vdino_train_main.py \
         --fold ${FOLD} --nfolds 5 --epochs 120 --bs 16 \
         --data_root '${DATA}' --ckpt_dir folds_vdino/fold_${FOLD}")
    FOLD_IDS+=("${JID}")
    echo "  fold ${FOLD}       -> job ${JID}"
done

# ── Ablations: one fold-0 job per variant ─────────────────────────────────────
ABL_NAMES=(no_ds no_bnd no_cur aug_none aug_weak base_dpt freeze4 freeze6)
ABL_IDS=()
for NAME in "${ABL_NAMES[@]}"; do
    JID=$(sbatch_train "vdino_${NAME}" "vdino_${NAME}" \
        "cd '${BASE}' && ${TENV} && ${TRAIN_PY} vdino_abl_${NAME}.py \
         --fold 0 --nfolds 5 --epochs 120 --bs 16 \
         --data_root '${DATA}' --ckpt_dir vdino_ablation_ckpts/${NAME}")
    ABL_IDS+=("${JID}")
    echo "  abl/${NAME}   -> job ${JID}"
done

echo ""
echo "====================================================================="
echo " Phase 1B — CAMUS few-shot eval (fold_0 ready -> starts immediately)"
echo "====================================================================="

CAMUS_JID=$(sbatch --parsable \
    --partition="${PART}" --account="${ACCT}" --qos="${QOS}" \
    --gres=gpu:1 --cpus-per-task=8 --mem=40G --time=05:00:00 \
    --job-name=vdino_camus \
    --output="${BASE}/logs/vdino_camus_%j.log" \
    --wrap="cd '${BASE}' && export TRANSFORMERS_OFFLINE=1 && \
            ${EVAL_PY} eval_camus_vdino.py \
            --ckpt_dir folds_vdino --camus_root '${CAMUS_ROOT}'")
echo "  CAMUS eval   -> job ${CAMUS_JID}  (no dependency)"

echo ""
echo "====================================================================="
echo " Phase 2 — ACDC / M&Ms / Sunnybrook evals  (wait for folds 1-4)"
echo "====================================================================="

FOLD_DEP="afterok:$(IFS=:; echo "${FOLD_IDS[*]}")"

ACDC_JID=$(sbatch_eval "vdino_eval_acdc" "vdino_eval_acdc" "${FOLD_DEP}" \
    "cd '${BASE}' && export TRANSFORMERS_OFFLINE=1 && \
     ${EVAL_PY} eval_vdino_final.py \
     --data_root '${DATA}' \
     --folds_dir folds_vdino \
     --abl_dir vdino_ablation_ckpts")
echo "  ACDC eval    -> job ${ACDC_JID}  (waits for folds 1-4)"

MNM_JID=$(sbatch_eval "vdino_eval_mnm" "vdino_eval_mnm" "${FOLD_DEP}" \
    "cd '${BASE}' && export TRANSFORMERS_OFFLINE=1 && \
     ${EVAL_PY} eval_mnm_vdino.py \
     --folds_dir folds_vdino --mnm_root '${MNM_ROOT}'")
echo "  M&Ms eval    -> job ${MNM_JID}   (waits for folds 1-4)"

SB_JID=$(sbatch_eval "vdino_eval_sb" "vdino_eval_sb" "${FOLD_DEP}" \
    "cd '${BASE}' && export TRANSFORMERS_OFFLINE=1 && \
     export SB_ROOT='${SB_ROOT}' && \
     ${EVAL_PY} eval_sunnybrook_vdino.py \
     --folds_dir folds_vdino")
echo "  Sunnybrook   -> job ${SB_JID}   (waits for folds 1-4)"

echo ""
echo "====================================================================="
echo " Phase 3 — Ablation table eval  (waits for all 8 ablation trains)"
echo "====================================================================="

# Ablation eval must wait for ALL folds (loads them for ensemble) AND all ablations
ALL_TRAIN_DEP="afterok:$(IFS=:; echo "${FOLD_IDS[*]}"):$(IFS=:; echo "${ABL_IDS[*]}")"

ABL_EVAL_JID=$(sbatch_eval "vdino_eval_abl" "vdino_eval_abl" "${ALL_TRAIN_DEP}" \
    "cd '${BASE}' && export TRANSFORMERS_OFFLINE=1 && \
     ${EVAL_PY} eval_vdino_final.py \
     --data_root '${DATA}' \
     --folds_dir folds_vdino \
     --abl_dir vdino_ablation_ckpts")
echo "  Ablation eval -> job ${ABL_EVAL_JID}  (waits for folds 1-4 + all 8 ablations)"

echo ""
echo "====================================================================="
echo " SUMMARY — all $(( 4 + ${#ABL_NAMES[@]} + 4 )) jobs submitted"
echo "====================================================================="
echo ""
printf "  %-26s: %s\n" "Main folds 1-4" "${FOLD_IDS[*]}"
printf "  %-26s: %s\n" "Ablations (8)" "${ABL_IDS[*]}"
printf "  %-26s: %s\n" "CAMUS few-shot" "${CAMUS_JID}"
printf "  %-26s: %s\n" "ACDC ensemble eval" "${ACDC_JID}"
printf "  %-26s: %s\n" "M&Ms zero-shot eval" "${MNM_JID}"
printf "  %-26s: %s\n" "Sunnybrook eval" "${SB_JID}"
printf "  %-26s: %s\n" "Ablation table eval" "${ABL_EVAL_JID}"
echo ""
echo "  Monitor  : squeue -u \$USER"
echo "  Logs dir : ${BASE}/logs/vdino_*"
echo ""
echo "  Estimated total wall time (with enough free GPUs):"
echo "    Training jobs      : ~5-8h  (12 jobs in parallel)"
echo "    CAMUS eval         : ~4-5h  (starts now)"
echo "    ACDC/M&Ms/SB evals : ~2-4h  (after training)"
echo "    End-to-end         : ~8-12h"
