# Inference & Evaluation

## A. Segment a single volume (easiest; CPU or GPU)
```bash
python pulse/infer.py \
  --input path/to/patient_frame01.nii.gz \
  --folds_dir checkpoints/folds_vdino \
  --output prediction.nii.gz
```
Force CPU with `--device cpu`, or use one fold with `--weights checkpoints/folds_vdino/fold_0/best_model.pth`.
Output labels: `0=BG, 1=RV, 2=Myo, 3=LV`.

## B. Full ACDC evaluation + biomarker extraction (5-fold ensemble)
```bash
python pulse/ensemble_eval.py \
  --folds_dir checkpoints/folds_vdino \
  --out diagnosis_features_ensemble.json \
  --data_root /path/to/ACDC/database
```
Reports Dice / IoU / HD95 / ASSD and writes per-patient biomarkers.

## C. Cardiomyopathy diagnosis (Random Forest on biomarkers)
```bash
python pulse/diagnosis_classify.py --features diagnosis_features_ensemble.json --seeds 10
```

## D. Structured clinical report
```bash
python pulse/generate_report.py --features diagnosis_features_ensemble.json --out_dir reports/
```

## E. Cross-dataset / few-shot experiments
```bash
python pulse/eval_mnm.py            --folds_dir checkpoints/folds_vdino   # M&Ms-2 zero-shot
python pulse/eval_sunnybrook.py     --folds_dir checkpoints/folds_vdino   # Sunnybrook zero-shot LV
python pulse/eval_camus_fewshot.py  --folds_dir checkpoints/folds_vdino   # CAMUS few-shot
```
(See each script's `--help` for dataset-path flags.)
