#!/usr/bin/env python3
"""CAMUS few-shot eval with vanilla DINOv2 backbone.
Sets BACKBONE before importing eval_camus_fewshot so PULSESeg is built correctly.
"""
import sys
from pathlib import Path
import pulse_seg as ps
ps.BACKBONE = "facebook/dinov2-base"

# Now safe to import the fewshot eval — it uses ps.PULSESeg() at runtime
sys.path.insert(0, str(Path(__file__).parent))
import eval_camus_fewshot  # noqa: F401 — runs __main__ via exec below

if __name__ == "__main__":
    eval_camus_fewshot.main()
