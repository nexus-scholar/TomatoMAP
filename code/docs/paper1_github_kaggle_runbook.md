# Paper 1 GitHub + Kaggle Runbook

This guide keeps `code/` as the source of truth and treats `TomatoMAP/` legacy content as read-only reference.

## 1) GitHub push workflow
Create `.gitignore` first, then use a safety check before commit:

```powershell
cd C:\Users\mouadh\Desktop\TomatoMAP
git init
git branch -M main
git status
git add .
git status
git commit -m "chore: github and kaggle prep for paper1 baseline"
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

## 2) Kaggle data dataset workflow (manual metadata template)
1. Edit `kaggle/datasets/paper1-data-template/dataset-metadata.json`.
2. Populate `kaggle/datasets/paper1-data-template/data/TomatoMAP_seg/{images,labels}`.

```powershell
kaggle datasets create -p .\kaggle\datasets\paper1-data-template --dir-mode zip
kaggle datasets version -p .\kaggle\datasets\paper1-data-template -m "Paper1 data update" --dir-mode zip
```

## 3) Kaggle code workflow
In a Kaggle notebook terminal:

```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
python -m unittest discover -s ./code/tests -p "test_*.py"
python ./code/scripts/prepare_paper1_baseline.py --config ./code/configs/paper1/baseline_v1.kaggle.json --freeze-split
python ./code/scripts/run_paper1_baseline.py --config ./code/configs/paper1/baseline_v1.kaggle.json --stage train
python ./code/scripts/run_paper1_baseline.py --config ./code/configs/paper1/baseline_v1.kaggle.json --stage eval
```

## 4) Attach data dataset in Kaggle notebook
- Add your uploaded dataset in notebook "Add data".
- Confirm mounted path (usually `/kaggle/input/<dataset-slug>/`).
- Ensure it contains `TomatoMAP_seg/images` and `TomatoMAP_seg/labels`.

