#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PULSE-Seg — segmentation-only 2.5D cardiac model with deep supervision + 5-fold CV.
================================================================================
Rebuild of the segmentation core for the honest paper. Diagnosis is now derived
post-hoc from the predicted masks (see diagnosis_features.py), so training is a
clean single-task segmentation problem -> simpler, stronger, reproducible.

Improvements over pulse_v3 aimed at closing the gap to ACDC SOTA (~0.91-0.92):
  * DEEP SUPERVISION: auxiliary seg heads at 2 intermediate decoder scales,
    loss on downsampled GT (weights 0.5, 0.25) -> sharper Myo/RV boundaries.
  * 5-FOLD CROSS-VALIDATION on the 100 official ACDC training patients
    (standard protocol). Fold models are ensembled at test time for accuracy
    AND give mean+/-std for statistical rigor. Test = 50 official test patients.
  * Keeps what worked: 2.5D input (3 adjacent slices), class-weighted Dice
    (Myo x3, RV x2), Lovasz, boundary loss, two-stage backbone unfreezing, TTA.

Usage (one fold):
  python pulse_seg.py --fold 0 --nfolds 5 --epochs 120 --bs 16 \
         --ckpt_dir folds/fold_0
"""
import os, math, random, warnings, argparse, json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import nibabel as nib
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel

try:
    import albumentations as A
    HAS_ALB = True
except ImportError:
    HAS_ALB = False

# Data root — set ACDC_ROOT env var or pass --data_root at runtime.
# Expected structure: <data_root>/training/patient001/ and <data_root>/testing/patient101/
ROOT       = Path(os.environ.get("ACDC_ROOT",
                  str(Path(__file__).resolve().parent / "ACDC" / "database")))
TRAIN_ROOT = ROOT / "training"
TEST_ROOT  = ROOT / "testing"
BACKBONE   = "facebook/dinov2-base"

IMG_SIZE   = 256
EPOCHS     = 120
WARMUP     = 8
STAGE1     = 70          # seg-only warmup with partial freeze; unfreeze rest after
STAGE2_LR  = 2e-5
LR         = 3e-4
WD         = 0.05
FREEZE_BLK = 2
DEC_CH     = 256

LAMBDA_LOV = 0.5
LAMBDA_BND = 0.5
DS_WEIGHTS = [1.0, 0.5, 0.25]   # main + 2 deep-supervision scales

SEG_CLASSES = 4
SEG_NAMES   = ["BG", "RV", "Myo", "LV"]
CLS_MAP     = {"NOR": 0, "DCM": 1, "HCM": 2, "MINF": 3, "RV": 4}
DICE_CLASS_W = torch.tensor([0.0, 2.0, 3.0, 1.0])
CE_CLASS_W   = torch.tensor([0.1, 2.0, 3.0, 1.0])


# ── Data listing + stratified k-fold ──────────────────────────────────────────
def parse_info_cfg(path: Path) -> Dict:
    info = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            for key in ("ED", "ES", "Group", "NbFrame"):
                if line.startswith(f"{key}:"):
                    v = line.split(":", 1)[1].strip()
                    info[key] = int(v) if key in {"ED", "ES", "NbFrame"} else v
    return info


def list_acdc(root: Path) -> List[Dict]:
    recs = []
    for pdir in sorted(root.glob("patient*/")):
        cfg = pdir / "Info.cfg"
        if not cfg.exists():
            continue
        info = parse_info_cfg(cfg)
        if "ED" not in info or "ES" not in info:
            continue
        grp = info.get("Group", "NOR")
        for phase in ("ED", "ES"):
            idx = info[phase]
            img = pdir / f"{pdir.name}_frame{idx:02d}.nii"
            gt  = pdir / f"{pdir.name}_frame{idx:02d}_gt.nii"
            if not img.exists(): img = img.with_suffix(".nii.gz")
            if not gt.exists():  gt  = gt.with_suffix(".nii.gz")
            if img.exists() and gt.exists():
                recs.append({"pid": pdir.name, "phase": phase, "img": str(img),
                             "gt": str(gt), "group": grp})
    return recs


def stratified_kfold_pids(recs, nfolds, fold, seed=42):
    """Patient-level stratified split by disease group. Returns (train_pids, val_pids)."""
    by_group: Dict[str, List[str]] = {}
    for r in recs:
        by_group.setdefault(r["group"], [])
        if r["pid"] not in by_group[r["group"]]:
            by_group[r["group"]].append(r["pid"])
    val_pids = set()
    rng = random.Random(seed)
    for grp, pids in by_group.items():
        pids = sorted(pids); rng.shuffle(pids)
        folds = [pids[i::nfolds] for i in range(nfolds)]
        val_pids.update(folds[fold])
    train_pids = {r["pid"] for r in recs} - val_pids
    return train_pids, val_pids


def _zscore(x): return (x - x.mean()) / (x.std() + 1e-6)


def _build_aug():
    if not HAS_ALB: return None
    return A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=0.1, scale=(0.85, 1.15), rotate=(-30, 30),
                 border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.ElasticTransform(alpha=80, sigma=4, p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.25, p=0.3),
        A.RandomBrightnessContrast(0.25, 0.25, p=0.5),
        A.GaussNoise(p=0.3), A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])


class ACDCDataset(Dataset):
    def __init__(self, records, img_size=256, train=True, bg_max=3):
        self.img_size, self.train = img_size, train
        self.aug = _build_aug() if train else None
        self.slices = []
        for rec in records:
            gt_vol = nib.load(rec["gt"]).get_fdata().astype(np.int64)
            S = gt_vol.shape[-1]
            fg = [z for z in range(S) if (gt_vol[..., z] > 0).any()]
            bg = [z for z in range(S) if not (gt_vol[..., z] > 0).any()]
            keep = bg if len(bg) <= bg_max else random.sample(bg, bg_max)
            for z in fg + keep:
                self.slices.append((rec, z, S))
        random.shuffle(self.slices)

    def __len__(self): return len(self.slices)

    def __getitem__(self, i):
        rec, z, S = self.slices[i]
        img = nib.load(rec["img"]).get_fdata().astype(np.float32)
        gt  = nib.load(rec["gt"]).get_fdata().astype(np.int64)
        z0, z1, z2 = max(z-1, 0), z, min(z+1, S-1)
        sl = [np.ascontiguousarray(_zscore(img[..., zi])) for zi in (z0, z1, z2)]
        g  = np.ascontiguousarray(gt[..., z1])
        r  = [cv2.resize(s, (self.img_size,)*2, interpolation=cv2.INTER_LINEAR) for s in sl]
        gr = cv2.resize(g.astype(np.uint8), (self.img_size,)*2, interpolation=cv2.INTER_NEAREST).astype(np.int64)
        x  = np.stack(r, -1).astype(np.float32)
        if self.train and self.aug is not None:
            o = self.aug(image=x, mask=gr.astype(np.uint8)); x = o["image"]; gr = o["mask"].astype(np.int64)
        x = np.ascontiguousarray(x); gr = np.ascontiguousarray(gr)
        return {"img": torch.from_numpy(x.transpose(2, 0, 1)).float(),
                "mask": torch.from_numpy(gr).long()}


# ── Losses ────────────────────────────────────────────────────────────────────
def _lovasz_grad(gt_sorted):
    gts = gt_sorted.sum(); inter = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0); jac = 1. - inter / union
    if len(jac) > 1: jac[1:] = jac[1:] - jac[:-1]
    return jac


def lovasz_softmax(logits, targets, ignore_bg=True):
    probas = F.softmax(logits, 1); B, C, H, W = probas.shape
    pf = probas.permute(0, 2, 3, 1).reshape(-1, C); tf = targets.reshape(-1)
    losses = []
    for c in range(1 if ignore_bg else 0, C):
        fg = (tf == c).float()
        if fg.sum() == 0: continue
        err = (fg - pf[:, c]).abs(); es, perm = torch.sort(err, descending=True)
        losses.append(torch.dot(es, _lovasz_grad(fg[perm])))
    return torch.stack(losses).mean() if losses else logits.sum() * 0.


def boundary_loss(logits, targets):
    probs = F.softmax(logits, 1); myo = probs[:, 2]; myo_gt = (targets == 2).float()
    dil = F.max_pool2d(myo_gt.unsqueeze(1), 3, 1, 1).squeeze(1)
    ero = -F.max_pool2d(-myo_gt.unsqueeze(1), 3, 1, 1).squeeze(1)
    bnd = (dil - ero).clamp(0, 1); w = 1.0 + 4.0 * bnd
    pc = myo.clamp(1e-6, 1 - 1e-6)
    bce = -(myo_gt * torch.log(pc) + (1 - myo_gt) * torch.log(1 - pc))
    return (w * bce).mean()


class WeightedDice(nn.Module):
    def __init__(self, w, eps=1e-6):
        super().__init__(); self.register_buffer("w", w); self.C = len(w); self.eps = eps
    def forward(self, logits, target):
        probs = F.softmax(logits, 1)
        oh = F.one_hot(target, self.C).permute(0, 3, 1, 2).float()
        inter = (probs * oh).sum((0, 2, 3)); denom = probs.sum((0, 2, 3)) + oh.sum((0, 2, 3))
        dice = (2 * inter + self.eps) / (denom + self.eps)
        return 1 - (self.w[1:] * dice[1:]).sum() / self.w[1:].sum().clamp_min(1e-6)


def seg_loss(logits, target, dice_fn, ce_fn):
    return (dice_fn(logits, target) + ce_fn(logits, target)
            + LAMBDA_LOV * lovasz_softmax(logits, target)
            + LAMBDA_BND * boundary_loss(logits, target))


# ── Model: DPT decoder with deep supervision ──────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))
    def forward(self, x): return F.gelu(x + self.net(x))


class DPTDecoderDS(nn.Module):
    """DPT decoder returning main logits + 2 deep-supervision logits (coarser)."""
    def __init__(self, embed_dim=768, ch=256, num_classes=4, out_size=256):
        super().__init__()
        self.out_size = out_size
        self.projs = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, ch), nn.GELU()) for _ in range(4)])
        self.res = nn.ModuleList([ResBlock(ch) for _ in range(4)])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(ch, ch, 2, stride=2, bias=False) for _ in range(4)])
        self.head = nn.Sequential(nn.Conv2d(ch, 64, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(64), nn.GELU(), nn.Conv2d(64, num_classes, 1))
        self.aux1 = nn.Conv2d(ch, num_classes, 1)   # after 2nd upsample stage
        self.aux2 = nn.Conv2d(ch, num_classes, 1)   # after 3rd upsample stage

    def _reassemble(self, hidden, proj):
        t = hidden[:, 1:]; B, N, _ = t.shape; hw = int(math.sqrt(N))
        return proj(t).transpose(1, 2).reshape(B, -1, hw, hw)

    def forward(self, hs):
        f = [self._reassemble(h, self.projs[i]) for i, h in enumerate(hs)]
        x = self.res[3](f[3]); x = self.ups[3](x)
        x = x + F.interpolate(f[2], size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = self.res[2](x); x = self.ups[2](x); a1 = self.aux1(x)
        x = x + F.interpolate(f[1], size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = self.res[1](x); x = self.ups[1](x); a2 = self.aux2(x)
        x = x + F.interpolate(f[0], size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = self.res[0](x); x = self.ups[0](x); x = self.head(x)
        main = F.interpolate(x, (self.out_size,)*2, mode='bilinear', align_corners=False)
        return main, a1, a2


class PULSESeg(nn.Module):
    EXTRACT = [3, 6, 9, 12]
    def __init__(self, img_size=256, freeze_blocks=None):
        super().__init__()
        if freeze_blocks is None:
            freeze_blocks = FREEZE_BLK  # read global at call time, not definition time
        self.stem = nn.Conv2d(3, 3, 1, bias=False)
        self.vit = AutoModel.from_pretrained(BACKBONE)
        ed = self.vit.config.hidden_size
        blocks = self.vit.encoder.layer if hasattr(self.vit, "encoder") else self.vit.vit.encoder.layer
        self.backbone_blocks = blocks
        for blk in blocks[:min(freeze_blocks, len(blocks))]:
            for p in blk.parameters(): p.requires_grad = False
        self.decoder = DPTDecoderDS(ed, DEC_CH, SEG_CLASSES, img_size)

    def forward(self, x):
        x = self.stem(x)
        out = self.vit(pixel_values=x, output_hidden_states=True)
        hs = [out.hidden_states[i] for i in self.EXTRACT]
        return self.decoder(hs)


# ── Metrics / TTA ─────────────────────────────────────────────────────────────
def dice_per_class(probs, target):
    if probs.shape[2:] != target.shape[1:]:
        probs = F.interpolate(probs, target.shape[1:], mode='bilinear', align_corners=False)
    pred = probs.argmax(1); out = []
    for c in range(1, SEG_CLASSES):
        p = (pred == c).cpu(); t = (target == c).cpu()
        inter = (p & t).sum().item(); denom = p.sum().item() + t.sum().item()
        out.append(0. if denom == 0 else 2 * inter / denom)
    return np.array(out)


@torch.no_grad()
def eval_epoch(model, loader, device, tta=False):
    model.eval(); dices = []
    for b in loader:
        x, y = b["img"].to(device), b["mask"].to(device)
        if tta:
            ps = []
            for fh in (False, True):
                for fv in (False, True):
                    xi = x.clone()
                    if fh: xi = xi.flip(-1)
                    if fv: xi = xi.flip(-2)
                    with torch.amp.autocast(device.type): m, _, _ = model(xi)
                    if fh: m = m.flip(-1)
                    if fv: m = m.flip(-2)
                    ps.append(F.softmax(m, 1))
            prob = torch.stack(ps).mean(0)
        else:
            with torch.amp.autocast(device.type): m, _, _ = model(x)
            prob = F.softmax(m, 1)
        dices.append(dice_per_class(prob, y))
    return np.stack(dices).mean(0)


def set_lr(opt, epoch, base, s2):
    if epoch < STAGE1:
        if epoch < WARMUP: scale = (epoch + 1) / WARMUP
        else:
            t = (epoch - WARMUP) / max(1, STAGE1 - WARMUP); scale = 0.5 * (1 + math.cos(math.pi * t))
        opt.param_groups[0]["lr"] = base * scale * 0.1
        opt.param_groups[1]["lr"] = base * scale
    else:
        opt.param_groups[0]["lr"] = s2 * 0.1
        opt.param_groups[1]["lr"] = s2
        # Group 2 is the early backbone blocks unfrozen at Stage-2; keep at fixed low LR.
        if len(opt.param_groups) > 2:
            opt.param_groups[2]["lr"] = s2 * 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--nfolds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--data_root", type=str, default=None,
                    help="Path to ACDC database/ dir (contains training/ and testing/)."
                         " Overrides the ACDC_ROOT env var.")
    args = ap.parse_args()

    global TRAIN_ROOT, TEST_ROOT
    if args.data_root:
        TRAIN_ROOT = Path(args.data_root) / "training"
        TEST_ROOT  = Path(args.data_root) / "testing"

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_recs = list_acdc(TRAIN_ROOT)
    test_recs = list_acdc(TEST_ROOT)
    tr_pids, va_pids = stratified_kfold_pids(all_recs, args.nfolds, args.fold, seed=42)
    tr = [r for r in all_recs if r["pid"] in tr_pids]
    va = [r for r in all_recs if r["pid"] in va_pids]
    print(f"Fold {args.fold}/{args.nfolds} | train pids {len(tr_pids)} val pids {len(va_pids)} "
          f"| train slices≈ (built below) | test recs {len(test_recs)}")

    tr_ds = ACDCDataset(tr, IMG_SIZE, train=True)
    va_ds = ACDCDataset(va, IMG_SIZE, train=False)
    te_ds = ACDCDataset(test_recs, IMG_SIZE, train=False)
    tr_dl = DataLoader(tr_ds, args.bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, args.bs, shuffle=False, num_workers=4, pin_memory=True)
    te_dl = DataLoader(te_ds, args.bs, shuffle=False, num_workers=4, pin_memory=True)

    model = PULSESeg(IMG_SIZE).to(device)
    bb = model.backbone_blocks
    bb_train = [p for blk in bb[FREEZE_BLK:] for p in blk.parameters() if p.requires_grad]
    other = [p for n, p in model.named_parameters()
             if p.requires_grad and not any(p is bp for blk in bb[FREEZE_BLK:] for bp in blk.parameters())]
    opt = torch.optim.AdamW([{"params": bb_train, "lr": args.lr * 0.1},
                             {"params": other, "lr": args.lr}], weight_decay=WD)
    scaler = torch.amp.GradScaler(device.type)
    dice_fn = WeightedDice(DICE_CLASS_W.to(device)).to(device)
    ce_fn = nn.CrossEntropyLoss(weight=CE_CLASS_W.to(device))

    best = 0.; unfroze = False; hist = []
    for epoch in range(1, args.epochs + 1):
        if epoch == STAGE1 + 1 and not unfroze:
            fp = []
            for blk in bb[:FREEZE_BLK]:
                for p in blk.parameters(): p.requires_grad = True; fp.append(p)
            opt.add_param_group({"params": fp, "lr": STAGE2_LR * 0.05}); unfroze = True
            print(f"  >> Stage2: unfroze blocks 1-{FREEZE_BLK}")
        set_lr(opt, epoch - 1, args.lr, STAGE2_LR)
        model.train(); tot = 0.
        for b in tqdm(tr_dl, desc=f"ep{epoch} train", leave=False):
            x, y = b["img"].to(device), b["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type):
                main, a1, a2 = model(x)
                y1 = F.interpolate(y.unsqueeze(1).float(), a1.shape[-2:], mode='nearest').squeeze(1).long()
                y2 = F.interpolate(y.unsqueeze(1).float(), a2.shape[-2:], mode='nearest').squeeze(1).long()
                loss = (DS_WEIGHTS[0] * seg_loss(main, y, dice_fn, ce_fn)
                        + DS_WEIGHTS[1] * seg_loss(a2, y2, dice_fn, ce_fn)
                        + DS_WEIGHTS[2] * seg_loss(a1, y1, dice_fn, ce_fn))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); tot += loss.item()
        v = eval_epoch(model, va_dl, device, tta=(epoch > STAGE1))
        vm = float(v.mean())
        hist.append({"epoch": epoch, "loss": tot / len(tr_dl),
                     "val_RV": float(v[0]), "val_Myo": float(v[1]), "val_LV": float(v[2]), "val_mean": vm})
        print(f"Ep{epoch:03d} loss={tot/len(tr_dl):.4f} | val RV={v[0]:.3f} Myo={v[1]:.3f} LV={v[2]:.3f} mean={vm:.3f}")
        if vm > best:
            best = vm
            torch.save({"epoch": epoch, "model": model.state_dict(), "best_dice": best,
                        "fold": args.fold}, ckpt_dir / "best_model.pth")
            print(f"  >> saved best mean={best:.4f}")
    json.dump(hist, open(ckpt_dir / "history.json", "w"), indent=2)

    ck = torch.load(ckpt_dir / "best_model.pth", map_location=device)
    model.load_state_dict(ck["model"])
    t = eval_epoch(model, te_dl, device, tta=True)
    print(f"\n[FOLD {args.fold} TEST+TTA] RV={t[0]:.3f} Myo={t[1]:.3f} LV={t[2]:.3f} mean={t.mean():.3f}")


if __name__ == "__main__":
    main()
