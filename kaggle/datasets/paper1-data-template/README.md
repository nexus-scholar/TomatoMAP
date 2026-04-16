# Kaggle Data Package Template (Paper 1)

This folder is a data-only Kaggle dataset template for Paper 1 baseline reproducibility.

## Purpose
- Keep code in this Git repository.
- Publish data as a separate Kaggle dataset.
- Attach the data dataset to Kaggle notebooks that run this repo.

## Before create/version
1. Fill `dataset-metadata.json` (`id`, `title`, `subtitle`, `description`).
2. Populate `data/TomatoMAP_seg/images/` and `data/TomatoMAP_seg/labels/`.
3. Keep folder names exactly as documented in `expected-structure.md`.

## Create dataset
```powershell
kaggle datasets create -p .\kaggle\datasets\paper1-data-template --dir-mode zip
```

## Create new version later
```powershell
kaggle datasets version -p .\kaggle\datasets\paper1-data-template -m "Paper1 data update" --dir-mode zip
```

