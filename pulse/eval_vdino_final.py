#!/usr/bin/env python3
"""
Final vanilla DINOv2 evaluation:
  1. Per-fold results (folds 0-4) with TTA
  2. 5-fold ensemble result
  3. Component ablation evals (fold-0 only ablation checkpoints)
"""
import sys, warnings, argparse
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, nibabel as nib, cv2, torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import pulse_seg as PS
PS.BACKBONE = "facebook/dinov2-base"   # must be set BEFORE any PULSESeg is constructed

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
    m.load_state_dict(state); m.eval()
    bd = ck.get("best_dice", "?")
    label = f"{Path(ckpt_path).parent.parent.name}/{Path(ckpt_path).parent.name}"
    print(f"  loaded {label}  best_dice={bd:.4f}" if isinstance(bd, float) else f"  loaded {label}")
    return m


@torch.no_grad()
def predict_ensemble(models, img_path, use_tta=True):
    vol = nib.load(str(img_path)).get_fdata().astype("float32")
    H, W, S = vol.shape; pred_vol = np.zeros((H, W, S), np.uint8)
    for z in range(S):
        z0, z1, z2 = max(z-1,0), z, min(z+1,S-1)
        sl = [np.ascontiguousarray(zscore(vol[..., zi])) for zi in (z0,z1,z2)]
        r  = [cv2.resize(s,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_LINEAR) for s in sl]
        x0 = torch.from_numpy(np.stack(r,-1).transpose(2,0,1)).float().unsqueeze(0).to(DEVICE)
        all_probs = []
        for model in models:
            if use_tta:
                probs = []
                for fh,fv in [(False,False),(True,False),(False,True),(True,True)]:
                    xi = x0.clone()
                    if fh: xi = xi.flip(-1)
                    if fv: xi = xi.flip(-2)
                    with torch.amp.autocast(DEVICE.type):
                        out = model(xi); main = out[0] if isinstance(out,tuple) else out
                    p = F.softmax(main,1)
                    if fh: p=p.flip(-1)
                    if fv: p=p.flip(-2)
                    probs.append(p)
                all_probs.append(torch.stack(probs).mean(0))
            else:
                with torch.amp.autocast(DEVICE.type):
                    out = model(x0); main = out[0] if isinstance(out,tuple) else out
                all_probs.append(F.softmax(main,1))
        prob = torch.stack(all_probs).mean(0)
        pred = prob[0].argmax(0).cpu().numpy().astype(np.uint8)
        pred_vol[...,z] = cv2.resize(pred,(W,H),interpolation=cv2.INTER_NEAREST)
    return postprocess(pred_vol)


def dice3d(pred, gt, c):
    p=(pred==c); g=(gt==c); i=(p&g).sum(); d=p.sum()+g.sum()
    return 0.0 if d==0 else float(2*i/d)


def evaluate(label, models, patients):
    print(f"\n{'='*60}\n[{label}]  {len(models)} model(s), {len(patients)} patients")
    pc = {n:[] for n in ("RV","Myo","LV")}
    for pid, rec in sorted(patients.items()):
        pat = {n:[] for n in ("RV","Myo","LV")}
        for phase, info in rec["phases"].items():
            pred = predict_ensemble(models, info["img"])
            gt   = nib.load(str(info["gt"])).get_fdata().astype(np.int64)
            for c,n in ((1,"RV"),(2,"Myo"),(3,"LV")): pat[n].append(dice3d(pred,gt,c))
        for n in ("RV","Myo","LV"):
            if pat[n]: pc[n].append(float(np.mean(pat[n])))
    rv=100*np.mean(pc["RV"]); myo=100*np.mean(pc["Myo"]); lv=100*np.mean(pc["LV"])
    mn=(rv+myo+lv)/3
    print(f"  RESULT: RV={rv:.1f}  Myo={myo:.1f}  LV={lv:.1f}  Mean={mn:.1f}")
    return {"RV":rv,"Myo":myo,"LV":lv,"Mean":mn}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",  required=True)
    ap.add_argument("--folds_dir",  default="folds_vdino",
                    help="Directory containing fold_0..fold_4/best_model.pth")
    ap.add_argument("--abl_dir",    default="vdino_ablation_ckpts",
                    help="Directory containing per-ablation best_model.pth")
    args = ap.parse_args()

    BASE = Path(__file__).parent
    _df.TEST_ROOT = Path(args.data_root) / "testing"
    patients = _df.list_patients(_df.TEST_ROOT)

    # ── 1. Per-fold results ────────────────────────────────────────────────────
    fold_ckpts = sorted((BASE / args.folds_dir).glob("fold_*/best_model.pth"))
    print(f"\nFound {len(fold_ckpts)} fold checkpoints in {args.folds_dir}")
    fold_results = []
    for i, ckpt in enumerate(sorted(fold_ckpts, key=lambda p: p.parent.name)):
        m = load_model(ckpt, freeze_blk=2)
        r = evaluate(f"fold {i}", [m], patients)
        fold_results.append(r); del m; torch.cuda.empty_cache()

    means = [r["Mean"] for r in fold_results]
    print(f"\n  Cross-fold mean: {np.mean(means):.1f} ± {np.std(means):.1f}")

    # ── 2. 5-fold ensemble ─────────────────────────────────────────────────────
    print("\nLoading all folds for ensemble...")
    ens_models = [load_model(ck, freeze_blk=2) for ck in sorted(fold_ckpts, key=lambda p: p.parent.name)]
    r_ens = evaluate("5-fold ENSEMBLE", ens_models, patients)
    for m in ens_models: del m
    torch.cuda.empty_cache()

    # ── 3. Ablation evals ─────────────────────────────────────────────────────
    abl_results = {}
    abl_dir = BASE / args.abl_dir
    # freeze_blk used at EVAL time must match what was used during training
    ABL_CONFIGS = {
        "no_ds":    (2,),   # trained with FREEZE_BLK=2, only DS removed
        "no_bnd":   (2,),   # trained with FREEZE_BLK=2, only bnd loss removed
        "no_cur":   (0,),   # trained with FREEZE_BLK=0 (no curriculum = no freeze)
        "aug_none": (2,),   # trained with FREEZE_BLK=2, no augmentation
        "aug_weak": (2,),   # trained with FREEZE_BLK=2, weak augmentation
        "base_dpt": (0,),   # trained with FREEZE_BLK=0 (DPT baseline, no innovations)
        "freeze4":  (4,),   # trained with FREEZE_BLK=4
        "freeze6":  (6,),   # trained with FREEZE_BLK=6
    }
    for name, (fblk,) in ABL_CONFIGS.items():
        ckpt = abl_dir / name / "best_model.pth"
        if not ckpt.exists():
            print(f"\n  SKIP {name} — checkpoint not found"); continue
        m = load_model(ckpt, freeze_blk=fblk)
        abl_results[name] = evaluate(f"abl/{name}", [m], patients)
        del m; torch.cuda.empty_cache()

    # ── FINAL SUMMARY ──────────────────────────────────────────────────────────
    print("\n\n" + "="*65)
    print("FINAL SUMMARY — copy into main.tex (Vanilla DINOv2)")
    print("="*65)
    hdr = f"{'Configuration':<35} {'RV':>6} {'Myo':>6} {'LV':>6} {'Mean':>7}"
    print(hdr); print("-"*len(hdr))
    for i, r in enumerate(fold_results):
        print(f"{'Fold '+str(i):<35} {r['RV']:>5.1f}% {r['Myo']:>5.1f}% {r['LV']:>5.1f}% {r['Mean']:>6.1f}%")
    print(f"{'Cross-fold mean':<35} {np.mean([r['RV'] for r in fold_results]):>5.1f}% "
          f"{np.mean([r['Myo'] for r in fold_results]):>5.1f}% "
          f"{np.mean([r['LV'] for r in fold_results]):>5.1f}% {np.mean(means):>5.1f}±{np.std(means):.1f}%")
    print(f"{'5-fold ensemble':<35} {r_ens['RV']:>5.1f}% {r_ens['Myo']:>5.1f}% {r_ens['LV']:>5.1f}% {r_ens['Mean']:>6.1f}%")
    if abl_results:
        print("\nAblations:")
        for name, r in abl_results.items():
            print(f"  {name:<31} {r['RV']:>5.1f}% {r['Myo']:>5.1f}% {r['LV']:>5.1f}% {r['Mean']:>6.1f}%")
    print("="*65)
