#!/usr/bin/env python3
"""
Compute per-disease per-class Dice using the current 5-fold ensemble
(same methodology as eval_vdino_final.py: 3D volumes, postprocess, dice3d).
"""
import sys
from pathlib import Path
import numpy as np, json
sys.path.insert(0, str(Path(__file__).parent))
import pulse_seg as PS
PS.BACKBONE = "facebook/dinov2-base"
PS.FREEZE_BLK = 2

from eval_vdino_final import load_model, predict_ensemble, dice3d
import diagnosis_features as _df

DEVICE   = __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")
BASE     = Path(__file__).parent
FOLDS    = BASE / "folds_vdino"
DATA     = Path("/SLURM/home/slurm_g202518690/student251/ACDC/database")
_df.TEST_ROOT = DATA / "testing"

import nibabel as nib

def main():
    fold_ckpts = sorted(FOLDS.glob("fold_*/best_model.pth"), key=lambda p: p.parent.name)
    models = [load_model(ck, freeze_blk=2) for ck in fold_ckpts]
    patients = _df.list_patients(_df.TEST_ROOT)

    # Group by disease
    per_disease = {grp: {s: [] for s in ("RV","Myo","LV")} for grp in ("NOR","DCM","HCM","MINF","RV")}

    for pid, rec in sorted(patients.items()):
        # Get disease group from Info.cfg
        info_cfg = _df.TEST_ROOT / pid / "Info.cfg"
        info = {}
        for line in info_cfg.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
        group = info.get("Group", None)
        if group not in per_disease:
            print(f"  Unknown group for {pid}: {group}")
            continue

        pat_dice = {s: [] for s in ("RV","Myo","LV")}
        for phase, pinfo in rec["phases"].items():
            pred = predict_ensemble(models, pinfo["img"], use_tta=True)
            gt   = nib.load(str(pinfo["gt"])).get_fdata().astype(int)
            for c, name in ((1,"RV"),(2,"Myo"),(3,"LV")):
                pat_dice[name].append(dice3d(pred, gt, c))

        for s in ("RV","Myo","LV"):
            per_disease[group][s].append(float(np.mean(pat_dice[s])))

    print("\nPer-disease per-class Dice (5-fold ensemble + TTA):")
    print(f"{'Group':<8}  {'LV':>6}  {'Myo':>6}  {'RV':>6}  {'Mean':>6}  N")
    print("-"*45)
    all_lv, all_myo, all_rv = [], [], []
    for grp in ("NOR","DCM","HCM","MINF","RV"):
        d = per_disease[grp]
        if not d["LV"]: continue
        lv  = 100*np.mean(d["LV"])
        myo = 100*np.mean(d["Myo"])
        rv  = 100*np.mean(d["RV"])
        mn  = (lv+myo+rv)/3
        n   = len(d["LV"])
        print(f"{grp:<8}  {lv:6.1f}  {myo:6.1f}  {rv:6.1f}  {mn:6.1f}  {n}")
        all_lv += d["LV"]; all_myo += d["Myo"]; all_rv += d["RV"]

    lv = 100*np.mean(all_lv); myo = 100*np.mean(all_myo); rv = 100*np.mean(all_rv)
    print(f"\nOVERALL: LV={lv:.1f}  Myo={myo:.1f}  RV={rv:.1f}  Mean={(lv+myo+rv)/3:.1f}")
    print("\n=== JSON OUTPUT ===")
    print(json.dumps({grp: {s: 100*float(np.mean(per_disease[grp][s])) for s in ("LV","Myo","RV")} for grp in ("NOR","DCM","HCM","MINF","RV")}))

if __name__ == "__main__":
    main()
