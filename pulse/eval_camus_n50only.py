"""Run CAMUS few-shot for N=50 only, with adaptive 20 epochs."""
import sys
sys.path.insert(0, '.')
# Override N_SHOTS before importing eval
import eval_camus_fewshot as ev
ev.N_SHOTS = [50]
# Override the adaptive epochs so N=50 uses 20 epochs
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--camus_root", default=str(ev.CAMUS_ROOT))
    ap.add_argument("--ckpt_dir",   default="folds")
    ap.add_argument("--view",       default="2CH")
    ap.add_argument("--epochs",     type=int, default=20)
    ap.add_argument("--lr",         type=float, default=5e-5)
    args = ap.parse_args()
    ev.main.__code__  # force import
    
# Actually just call main directly
import importlib, types
# Just run main from eval_camus_fewshot with N_SHOTS=[50]
