"""
Few-shot CAMUS echocardiography evaluation.

CAMUS 2CH label convention:
  0 = background
  1 = LV endocardium (cavity)
  2 = LV myocardium
  3 = LA (left atrium)

PULSE label convention (ACDC):
  0 = BG, 1 = RV, 2 = Myo, 3 = LV

Remapping for fine-tuning:
  CAMUS 1 (LV endo) -> PULSE 3 (LV)
  CAMUS 2 (Myo)     -> PULSE 2 (Myo)
  CAMUS 3 (LA)      -> 0 (background; ignored)

Temporal 2.5D input: frames (t-1, t, t+1) from the half-sequence cine,
clamped at sequence boundaries. ED and ES frame indices read from Info_2CH.cfg.

Evaluation metrics: LV Dice (PULSE class 3) and Myo Dice (PULSE class 2).
"""

import argparse, configparser, glob, os, random, re, sys, copy
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import cv2

sys.path.insert(0, str(Path(__file__).parent))
import pulse_seg as ps

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
NUM_CLASSES = 4
SEEDS = [0, 1, 2, 3, 4]
N_SHOTS = [5, 10, 20]

CAMUS_ROOT = Path(os.environ.get(
    "CAMUS_ROOT", "data/CAMUS/database_nifti"
))


def parse_cfg(cfg_path: Path) -> dict:
    """Parse CAMUS Info_2CH.cfg, returning dict of key->value."""
    info = {}
    with open(cfg_path) as f:
        for line in f:
            if ":" in line:
                k, v = line.strip().split(":", 1)
                info[k.strip()] = v.strip()
    return info


def remap_camus_label(label: np.ndarray) -> np.ndarray:
    """Remap CAMUS label convention to PULSE/ACDC convention."""
    out = np.zeros_like(label)
    out[label == 1] = 3   # LV endo -> LV
    out[label == 2] = 2   # Myo     -> Myo
    out[label == 3] = 0   # LA      -> background (ignore)
    return out


def zscore(img: np.ndarray) -> np.ndarray:
    m, s = img.mean(), img.std()
    return (img - m) / (s + 1e-8)


def resize2d(arr: np.ndarray, size: int, interp) -> np.ndarray:
    return cv2.resize(arr.astype(np.float32), (size, size), interpolation=interp)


def load_camus_sample(pat_dir: Path, view: str = "2CH"):
    """Load ED and ES samples with temporal 2.5D context.

    Returns list of (x, gt) where:
        x  : (3, H, W) float32 — three temporal frames
        gt : (H, W) int64      — remapped label (PULSE convention)
    """
    pid = pat_dir.name
    cfg = parse_cfg(pat_dir / f"Info_{view}.cfg")
    ed_idx = int(cfg["ED"]) - 1   # 1-based -> 0-based
    es_idx = int(cfg["ES"]) - 1
    n_frames = int(cfg["NbFrame"])

    seq_f = pat_dir / f"{pid}_{view}_half_sequence.nii.gz"
    seq = nib.load(str(seq_f)).get_fdata()   # (H, W, NbFrame)

    samples = []
    for t_idx, gt_key in [(ed_idx, "ED"), (es_idx, "ES")]:
        gt_f = pat_dir / f"{pid}_{view}_{gt_key}_gt.nii.gz"
        gt_raw = nib.load(str(gt_f)).get_fdata().squeeze().astype(np.int64)
        gt_r = resize2d(gt_raw.astype(np.float32), IMG_SIZE, cv2.INTER_NEAREST).astype(np.int64)
        gt_r = remap_camus_label(gt_r)

        frames = []
        for delta in [-1, 0, 1]:
            ti = np.clip(t_idx + delta, 0, n_frames - 1)
            f  = seq[..., ti]
            f  = resize2d(f, IMG_SIZE, cv2.INTER_LINEAR)
            f  = zscore(f)
            frames.append(f)
        x = np.stack(frames, axis=0).astype(np.float32)  # (3, H, W)
        samples.append((x, gt_r))
    return samples


def collect_all_samples(camus_root: Path, view: str):
    """Returns {patient_path: [(x, gt), ...]} for all patients."""
    patients = sorted(camus_root.glob("patient*/"))
    data = {}
    for p in patients:
        try:
            s = load_camus_sample(p, view=view)
            if s:
                data[p] = s
        except Exception as e:
            print(f"  Warning: skipping {p.name}: {e}")
    return data


class CAMUSDataset(Dataset):
    def __init__(self, samples, augment=False):
        # samples: list of (x, gt) where x:(3,H,W), gt:(H,W)
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, gt = self.samples[idx]
        x  = x.copy()
        gt = gt.copy()
        if self.augment:
            if random.random() > 0.5:
                x  = x[:, :, ::-1].copy()
                gt = gt[:, ::-1].copy()
            if random.random() > 0.5:
                x  = x[:, ::-1, :].copy()
                gt = gt[::-1, :].copy()
        return torch.tensor(x), torch.tensor(gt, dtype=torch.long)


def dice_per_class(pred: np.ndarray, gt: np.ndarray):
    """Returns dict with Myo and LV Dice."""
    results = {}
    for cls, name in [(2, "Myo"), (3, "LV")]:
        p = (pred == cls)
        g = (gt   == cls)
        inter = (p & g).sum()
        denom = p.sum() + g.sum()
        results[name] = float(2 * inter / denom) if denom > 0 else 1.0
    return results


def load_model(ckpt_path: str) -> ps.PULSESeg:
    ps.FREEZE_BLK = 2
    model = ps.PULSESeg(freeze_blocks=2).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    return model


def finetune(model, train_samples, epochs=40, lr=2e-5, bs=8):
    """Few-shot fine-tuning: freeze ViT backbone + stem, fine-tune decoder only."""
    for n, p in model.named_parameters():
        # Freeze backbone (vit.*) and stem; only decoder adapts
        p.requires_grad = not (n.startswith("vit.") or n.startswith("stem."))
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    ds = CAMUSDataset(train_samples, augment=True)
    dl = DataLoader(ds, batch_size=min(bs, len(ds)), shuffle=True,
                    num_workers=2, drop_last=False)
    model.train()
    ce_loss = nn.CrossEntropyLoss()
    dice_fn = ps.WeightedDice(w=torch.ones(NUM_CLASSES).to(DEVICE))
    for ep in range(epochs):
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast(DEVICE.type):
                out = model(x)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                loss = 0.5 * ce_loss(logits, y) + 0.5 * dice_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
    return model


def predict_tta(model, x: np.ndarray) -> np.ndarray:
    """4-way TTA for a single (3, H, W) input."""
    model.eval()
    flips = [(False, False), (True, False), (False, True), (True, True)]
    logits_sum = None
    for fh, fv in flips:
        xi = x.copy()
        if fh: xi = xi[:, :, ::-1].copy()
        if fv: xi = xi[:, ::-1, :].copy()
        t = torch.tensor(xi[None]).to(DEVICE)
        with torch.no_grad(), torch.amp.autocast(DEVICE.type):
            out = model(t)
            logits = out[0] if isinstance(out, (tuple, list)) else out
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        if fh: probs = probs[:, :, ::-1].copy()
        if fv: probs = probs[:, ::-1, :].copy()
        logits_sum = probs if logits_sum is None else logits_sum + probs
    return logits_sum.argmax(0).astype(np.int64)


def evaluate(model, test_samples):
    all_dice = []
    for x, gt in test_samples:
        pred = predict_tta(model, x)
        d    = dice_per_class(pred, gt)
        all_dice.append(d)
    myo  = np.mean([d["Myo"] for d in all_dice])
    lv   = np.mean([d["LV"]  for d in all_dice])
    mean = (myo + lv) / 2.0
    return {"Myo": myo, "LV": lv, "Mean": mean}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camus_root", default=str(CAMUS_ROOT))
    ap.add_argument("--ckpt_dir",   default="folds")
    ap.add_argument("--view",       default="2CH", choices=["2CH", "4CH"])
    ap.add_argument("--epochs",     type=int,   default=40)
    ap.add_argument("--lr",         type=float, default=5e-5)
    args = ap.parse_args()

    camus_root = Path(args.camus_root)
    print(f"Loading CAMUS data from {camus_root} (view={args.view}) ...")
    all_data = collect_all_samples(camus_root, view=args.view)
    patients  = sorted(all_data.keys())
    print(f"Total CAMUS patients with data: {len(patients)}")
    if not patients:
        raise RuntimeError(f"No CAMUS data found at {camus_root}")

    ckpt_path = Path(args.ckpt_dir) / "fold_0" / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"\n{'N':>6}  {'Myo':>7}  {'LV':>7}  {'Mean':>7}")
    print("-" * 35)

    # Adaptive epochs: more passes for smaller N to compensate for fewer samples
    EPOCHS_BY_N = {5: 60, 10: 50, 20: 40}

    results_by_n = {}
    for n_shot in N_SHOTS:
        seed_results = []
        ep = EPOCHS_BY_N.get(n_shot, args.epochs)
        for seed in SEEDS:
            rng = random.Random(seed)
            pats = list(patients)
            rng.shuffle(pats)
            train_pats = pats[:n_shot]
            test_pats  = pats[n_shot:]

            train_samples = [s for p in train_pats for s in all_data[p]]
            test_samples  = [s for p in test_pats  for s in all_data[p]]

            model = load_model(str(ckpt_path))
            model = finetune(model, train_samples, epochs=ep, lr=args.lr)
            res   = evaluate(model, test_samples)
            seed_results.append(res)
            print(f"  N={n_shot:>3} ep={ep} seed={seed}  Myo={res['Myo']:.3f}  LV={res['LV']:.3f}  Mean={res['Mean']:.3f}")

        myo  = np.mean([r["Myo"]  for r in seed_results])
        lv   = np.mean([r["LV"]   for r in seed_results])
        mean = np.mean([r["Mean"] for r in seed_results])
        results_by_n[n_shot] = {"Myo": myo, "LV": lv, "Mean": mean}

    print("\n=== CAMUS FEW-SHOT FINAL RESULTS (avg over 5 seeds) ===")
    print(f"{'N':>4}  {'Myo':>7}  {'LV':>7}  {'Mean':>7}")
    for n, r in results_by_n.items():
        print(f"{n:>4}  {r['Myo']:.3f}  {r['LV']:.3f}  {r['Mean']:.3f}")
    print()
    print("NOTE: Evaluates LV and Myo Dice only (CAMUS has no RV labels in standard release).")


if __name__ == "__main__":
    main()
