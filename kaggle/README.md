# Kaggle Workflows (Paper 1)

This folder contains Kaggle preparation assets for the new root `code/` pipeline.

## Separation of concerns
- Code workflow: clone this repository and run commands from repo root.
- Data workflow: publish data as a separate Kaggle dataset using `kaggle/datasets/paper1-data-template/`.

## Kaggle notebook working directory
Use repository root as working directory (for example `/kaggle/working/TomatoMAP`).

## Baseline config for Kaggle
Use `code/configs/paper1/baseline_v1.kaggle.json` when running on Kaggle. It keeps the same baseline settings with Kaggle-friendly paths.

## Data dataset commands (manual metadata template flow)
```powershell
kaggle datasets create -p .\kaggle\datasets\paper1-data-template --dir-mode zip
kaggle datasets version -p .\kaggle\datasets\paper1-data-template -m "Paper1 data update" --dir-mode zip
```

