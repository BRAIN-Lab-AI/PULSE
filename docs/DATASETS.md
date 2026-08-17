# Datasets

All four datasets are **public**; access requires a free registration on each
portal. PULSE is trained only on ACDC; the others are used for zero-shot /
few-shot evaluation.

| Dataset | Modality | Use in PULSE | Link |
|---|---|---|---|
| ACDC | Cine MRI (short-axis) | Train + test | https://acdc.creatis.insa-lyon.fr |
| M&Ms-2 | Cine MRI (multi-vendor) | Zero-shot test | https://www.ub.edu/mnms-2/ |
| Sunnybrook | Cine MRI | Zero-shot LV test | https://www.cardiacatlas.org/sunnybrook-cardiac-data/ |
| CAMUS | 2D echocardiography | Few-shot transfer | https://www.creatis.insa-lyon.fr/Challenge/camus/ |

## ACDC layout (required for training)
```
ACDC/database/
├── training/
│   ├── patient001/
│   │   ├── patient001_frame01.nii.gz         # ED image
│   │   ├── patient001_frame01_gt.nii.gz      # ED label
│   │   ├── patient001_frame12.nii.gz         # ES image
│   │   ├── patient001_frame12_gt.nii.gz      # ES label
│   │   └── Info.cfg
│   └── ... patient100/
└── testing/
    └── patient101/ ... patient150/
```

Tell PULSE where the data is (either works):
```bash
export ACDC_ROOT=/path/to/ACDC/database        # environment variable, OR
python pulse/train.py --data_root /path/to/ACDC/database ...   # CLI flag
```

Labels: `1 = RV`, `2 = Myocardium`, `3 = LV`, `0 = background`.

> Don't have the data yet and just want to test the code? You can still run
> `python pulse/infer.py --input <any_short_axis>.nii.gz` on a single volume
> once you have the weights (see `checkpoints/README.md`).
