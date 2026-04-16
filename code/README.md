# Phase 1 segmentation utilities (greenfield)

This folder is a new, isolated implementation for Paper 1 reproducibility.
Legacy repository files are treated as read-only reference.
`code/` is the source of truth for all new implementation.

## Scope
- Segmentation image/label audit
- Segmentation label extraction from zip
- ISAT-to-COCO conversion (import-safe, explicit call)
- Lightweight CLI scripts and tests

## Run
From the repository root:
```powershell
python .\code\scripts\audit_seg.py --img-dir "TomatoMAP\TomatoMAP_seg\images" --lbl-dir "TomatoMAP\TomatoMAP_seg\labels"
python .\code\scripts\extract_seg_labels.py --zip-path "TomatoMAP\TomatoMAP_seg.zip" --dest-dir "TomatoMAP\TomatoMAP_seg\labels"
python .\code\scripts\convert_isat_to_coco.py --task-dir "TomatoMAP\TomatoMAP_seg\images" --label-dir "TomatoMAP\TomatoMAP_seg\labels" --categories "TomatoMAP\TomatoMAP_seg\labels\isat.yaml" --output-dir "code\outputs\paper1\baseline_v1\coco"
```

Categories note: the categories file must contain ISAT label entries with `name` fields (entries like `- name: tomato`).

## Tests
```powershell
python -m unittest discover -s ".\code\tests" -p "test_*.py"
```

## Phase 2 baseline (Paper 1)
Use one supervised baseline path with a frozen canonical split.

```powershell
python .\code\scripts\prepare_paper1_baseline.py --config ".\code\configs\paper1\baseline_v1.json" --freeze-split
python .\code\scripts\run_paper1_baseline.py --config ".\code\configs\paper1\baseline_v1.json" --stage train
python .\code\scripts\run_paper1_baseline.py --config ".\code\configs\paper1\baseline_v1.json" --stage eval
```

Notes:
- Split is frozen once in `code/configs/paper1/split_manifest_v1.json` and reused on every run.
- Derived COCO outputs are kept under `code/outputs/paper1/baseline_v1/coco/`.
- Dataset-view link/junction is optional; if it fails, the wrapper copies source images into `code/outputs/paper1/baseline_v1/dataset_view/images/` so the legacy engine still sees `images/` and `cocoOut/` under the same runtime root.

## GitHub and Kaggle workflows
Use `code/docs/paper1_github_kaggle_runbook.md` for the exact push, Kaggle dataset, and Kaggle baseline execution steps.
