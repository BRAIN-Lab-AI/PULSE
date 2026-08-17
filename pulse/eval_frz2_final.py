#!/usr/bin/env python3
"""
Final evaluation:
  1. frz2 5-fold ensemble (fold-0 from ablation_ckpts/freeze_2blk + folds 1-4 from folds_frz2/)
  2. frz2 per-fold results (for fold results table)
  3. vanilla DINOv2 fold-0 (3D patient-level, for backbone ablation table)
"""
import sys, warnings, argparse
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import nibabel as nib
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import pulse_seg as PS
import diagnosis_features as _df
from diagnosis_features import zscore
from postprocess import postprocess

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256


def load_model(ckpt_path, freeze_blk=2):
    PS.FREEZE_BLK = freeze_blk
    m = PS.PULSESeg(img_size=IMG_SIZE, freeze_blocks=freeze_blk).to(DEVICE)
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ck["model"] if "model" in ck else ck
    m.load_state_dict(state)
    m.eval()
    bd = ck.get("best_dice", "?")
    print(f"  loaded {Path(ckpt_path).parent.name}/{Path(ckpt_path).name}  best_dice={bd:.4f}" if isinstance(bd, float) else f"  loaded {ckpt_path}")
    return m


@torch.no_grad()
def predict_ensemble(models, img_path, use_tta=True):
    vol = nib.load(str(img_path)).get_fdata().astype(np.float32)
    H, W, S = vol.shape
    pred_vol = np.zeros((H, W, S), np.uint8)
    for z in range(S):
        z0, z1, z2 = max(z-1,0), z, min(z+1,S-1)
        sl = [np.ascontiguousarray(zscore(vol[..., zi])) for zi in (z0, z1, z2)]
        r  = [cv2.resize(s, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR) for s in sl]
        x0 = torch.from_numpy(np.stack(r,-1).transpose(2,0,1)).float().unsqueeze(0).to(DEVICE)
        all_probs = []
        for model in models:
            if use_tta:
                probs = []
                for fh, fv in [(False,False),(True,False),(False,True),(True,True)]:
                    xi = x0.clone()
                    if fh: xi = xi.flip(-1)
                    if fv: xi = xi.flip(-2)
                    with torch.amp.autocast(DEVICE.type):
                        out = model(xi); main = out[0] if isinstance(out, tuple) else out
                    p = F.softmax(main, 1)
                    if fh: p = p.flip(-1)
                    if fv: p = p.flip(-2)
                    probs.append(p)
                all_probs.append(torch.stack(probs).mean(0))
            else:
                with torch.amp.autocast(DEVICE.type):
                    out = model(x0); main = out[0] if isinstance(out, tuple) else out
                all_probs.append(F.softmax(main, 1))
        prob = torch.stack(all_probs).mean(0)
        pred = prob[0].argmax(0).cpu().numpy().astype(np.uint8)
        pred_vol[..., z] = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)
    return postprocess(pred_vol)


def dice3d(pred, gt, c):
    p = (pred == c); g = (gt == c)
    inter = (p & g).sum(); denom = p.sum() + g.sum()
    return 0.0 if denom == 0 else float(2 * inter / denom)


def evaluate_models(label, models, patients):
    print(f"\n{'='*60}\n[{label}]  {len(models)} model(s), {len(patients)} patients, TTA=True")
    per_class = {n: [] for n in ("RV","Myo","LV")}
    for pid, rec in sorted(patients.items()):
        pat = {n: [] for n in ("RV","Myo","LV")}
        for phase, info in rec["phases"].items():
            pred = predict_ensemble(models, info["img"], use_tta=True)
            gt   = nib.load(str(info["gt"])).get_fdata().astype(np.int64)
            for c, n in ((1,"RV"),(2,"Myo"),(3,"LV")):
                pat[n].append(dice3d(pred, gt, c))
        for n in ("RV","Myo","LV"):
            if pat[n]: per_class[n].append(float(np.mean(pat[n])))
    rv  = 100*np.mean(per_class["RV"])
    myo = 100*np.mean(per_class["Myo"])
    lv  = 100*np.mean(per_class["LV"])
    mn  = (rv+myo+lv)/3
    print(f"  RESULT: RV={rv:.1f}  Myo={myo:.1f}  LV={lv:.1f}  Mean={mn:.1f}")
    return {"RV":rv,"Myo":myo,"LV":lv,"Mean":mn}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--frz2_fold0",  default="ablation_ckpts/freeze_2blk/best_model.pth")
    ap.add_argument("--frz2_dir",    default="folds_frz2")
    ap.add_argument("--vanilla_ckpt",default="ablation_ckpts/vanilla_dino/best_model.pth")
    args = ap.parse_args()

    BASE = Path(__file__).parent
    _df.TEST_ROOT = Path(args.data_root) / "testing"
    patients = _df.list_patients(_df.TEST_ROOT)

    results = {}

    # ── 1. frz2 per-fold (for fold results table) ─────────────────────────────
    fold_ckpts = [Path(args.frz2_fold0)] + sorted((BASE/args.frz2_dir).glob("fold_*/best_model.pth"))
    print(f"\nFound {len(fold_ckpts)} frz2 fold checkpoints")
    fold_results = []
    for i, ckpt in enumerate(fold_ckpts):
        m = load_model(ckpt, freeze_blk=2)
        r = evaluate_models(f"frz2 fold {i}", [m], patients)
        fold_results.append(r)
        del m; torch.cuda.empty_cache()

    print("\n\n" + "="*60)
    print("FRZ2 PER-FOLD RESULTS (for tab:foldresults)")
    print("="*60)
    means = []
    for i, r in enumerate(fold_results):
        print(f"  Fold {i}: RV={r['RV']:.1f}  Myo={r['Myo']:.1f}  LV={r['LV']:.1f}  Mean={r['Mean']:.1f}")
        means.append(r['Mean'])
    print(f"  Cross-fold mean: {np.mean(means):.1f} ± {np.std(means):.1f}")

    # ── 2. frz2 5-fold ensemble ───────────────────────────────────────────────
    print("\nLoading all 5 frz2 models for ensemble...")
    ens_models = [load_model(ck, freeze_blk=2) for ck in fold_ckpts]
    r_ens = evaluate_models("frz2 5-fold ENSEMBLE", ens_models, patients)
    results["frz2_ensemble"] = r_ens
    for m in ens_models: del m
    torch.cuda.empty_cache()

    # ── 3. Vanilla DINOv2 fold-0 3D eval ─────────────────────────────────────
    vdino_ckpt = BASE / args.vanilla_ckpt
    if vdino_ckpt.exists():
        _saved_backbone = PS.BACKBONE
        PS.BACKBONE = "facebook/dinov2-base"
        m_vd = load_model(vdino_ckpt, freeze_blk=2)
        r_vd = evaluate_models("Vanilla DINOv2 fold-0", [m_vd], patients)
        results["vanilla_dino"] = r_vd
        del m_vd; torch.cuda.empty_cache()
        PS.BACKBONE = _saved_backbone  # restore to prevent contamination
    else:
        print(f"\nVanilla DINOv2 checkpoint not found at {vdino_ckpt}")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print("\n\n" + "="*65)
    print("FINAL SUMMARY — copy into main.tex")
    print("="*65)
    hdr = f"{'Configuration':<35} {'RV':>6} {'Myo':>6} {'LV':>6} {'Mean':>7}"
    print(hdr); print("-"*len(hdr))
    for fold_i, r in enumerate(fold_results):
        print(f"{'frz2 fold '+str(fold_i):<35} {r['RV']:>5.1f}% {r['Myo']:>5.1f}% {r['LV']:>5.1f}% {r['Mean']:>6.1f}%")
    print(f"{'frz2 cross-fold mean':<35} {np.mean([r['RV'] for r in fold_results]):>5.1f}% {np.mean([r['Myo'] for r in fold_results]):>5.1f}% {np.mean([r['LV'] for r in fold_results]):>5.1f}% {np.mean(means):>5.1f}±{np.std(means):.1f}%")
    print(f"{'frz2 5-fold ensemble':<35} {r_ens['RV']:>5.1f}% {r_ens['Myo']:>5.1f}% {r_ens['LV']:>5.1f}% {r_ens['Mean']:>6.1f}%")
    if "vanilla_dino" in results:
        r_vd = results["vanilla_dino"]
        delta = r_ens["Mean"] - r_vd["Mean"]
        print(f"{'Vanilla DINOv2 fold-0':<35} {r_vd['RV']:>5.1f}% {r_vd['Myo']:>5.1f}% {r_vd['LV']:>5.1f}% {r_vd['Mean']:>6.1f}%")
        print(f"\nRAD-DINOv2 vs vanilla DINOv2 gain: +{delta:.1f}% (ensemble vs fold-0)")
    print("="*65)
