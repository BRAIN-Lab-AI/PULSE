#!/usr/bin/env python3
"""
Evaluate ensemble scaling (1-fold no TTA, 1/2/3/4 folds with TTA).
Uses IDENTICAL evaluation methodology to eval_vdino_final.py:
  - predict_ensemble: 3D volume inference per NIfTI file
  - postprocess: largest-connected-component filtering
  - dice3d: 3D Dice over full volume
  - patient-level averaging (mean over phases, then over patients)
"""
import argparse, os, sys
from pathlib import Path
import torch, numpy as np, nibabel as nib
sys.path.insert(0, str(Path(__file__).parent))
import pulse_seg as PS
PS.BACKBONE = "facebook/dinov2-base"
PS.FREEZE_BLK = 2

from eval_vdino_final import load_model, predict_ensemble, dice3d
import diagnosis_features as _df

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_tta(models, patients, use_tta):
    pc = {n: [] for n in ("RV", "Myo", "LV")}
    for pid, rec in sorted(patients.items()):
        pat = {n: [] for n in ("RV", "Myo", "LV")}
        for phase, info in rec["phases"].items():
            pred = predict_ensemble(models, info["img"], use_tta=use_tta)
            gt   = nib.load(str(info["gt"])).get_fdata().astype(np.int64)
            for c, n in ((1, "RV"), (2, "Myo"), (3, "LV")):
                pat[n].append(dice3d(pred, gt, c))
        for n in ("RV", "Myo", "LV"):
            if pat[n]:
                pc[n].append(float(np.mean(pat[n])))
    rv  = 100 * np.mean(pc["RV"])
    myo = 100 * np.mean(pc["Myo"])
    lv  = 100 * np.mean(pc["LV"])
    mn  = (rv + myo + lv) / 3
    return {"RV": rv, "Myo": myo, "LV": lv, "Mean": mn}


def main():
    ap = argparse.ArgumentParser(description="Ensemble-size and TTA scaling on ACDC.")
    ap.add_argument("--folds_dir", default=os.environ.get("FOLDS_DIR", "checkpoints/folds_vdino"),
                    help="Folder with fold_*/best_model.pth (default: checkpoints/folds_vdino).")
    ap.add_argument("--data_root", default=os.environ.get("ACDC_ROOT"),
                    help="ACDC database root containing testing/. Or export ACDC_ROOT.")
    args = ap.parse_args()
    if not args.data_root:
        ap.error("Set --data_root /path/to/ACDC/database (or export ACDC_ROOT=/path/to/ACDC/database).")
    _df.TEST_ROOT = Path(args.data_root) / "testing"
    patients  = _df.list_patients(_df.TEST_ROOT)
    fold_ckpts = sorted(Path(args.folds_dir).glob("fold_*/best_model.pth"),
                        key=lambda p: p.parent.name)
    if not fold_ckpts:
        ap.error(f"No checkpoints in {args.folds_dir}. Run: bash checkpoints/download_weights.sh")

    all_models = []
    for ck in fold_ckpts:
        all_models.append(load_model(ck, freeze_blk=2))

    print(f"\n{'Config':<26}  {'RV':>6}  {'Myo':>6}  {'LV':>6}  {'Mean':>6}  TTA")
    print("-" * 62)

    for n, tta, label in [
        (1, False, "1 fold (fold 0)"),
        (1, True,  "1 fold (fold 0)"),
        (2, True,  "2-fold ensemble"),
        (3, True,  "3-fold ensemble"),
        (4, True,  "4-fold ensemble"),
    ]:
        print(f"  Evaluating {label} (TTA={tta})...")
        res = evaluate_tta(all_models[:n], patients, use_tta=tta)
        tta_str = "Yes" if tta else "No"
        print(f"{label:<26}  {res['RV']:6.1f}  {res['Myo']:6.1f}  "
              f"{res['LV']:6.1f}  {res['Mean']:6.1f}  {tta_str}")

    print("Done")


if __name__ == "__main__":
    main()
