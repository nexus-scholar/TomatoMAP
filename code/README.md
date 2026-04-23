# TomatoMAP Reproducible Research Codebase

This folder (`code/`) is the modern, isolated implementation for **Paper 1 reproducibility** on label-efficient tomato instance segmentation. The legacy repository files outside of this folder are treated as read-only reference data. `code/` is the strict source of truth for all new modeling, data conversion, and evaluation logic.

## 🌟 Key Features
- **Strictly YOLO / Ultralytics**: Lean, efficient, and optimized for Kaggle/Colab execution (Detectron2 legacy removed).
- **Type-Safe Configuration**: Driven by Pydantic schemas (`src/experiments/config.py`).
- **Reproducible Splits**: Freezes canonical train/val/test splits once via a `split_manifest_v1.json` to guarantee zero data leakage between runs.
- **Label-Efficient & Semi-Supervised Ready**: Supports native subset filtering by class name (e.g., training only on `["unripe", "fully-ripe"]`). Orphaned images map automatically to an `unlabeled_images.json` pool for downstream semi-supervised learning workflows.

---

## 📂 Directory Structure

```text
code/
├── configs/               # Pydantic-validated JSON configurations (e.g., baseline_v1_yolov11.json)
├── docs/                  # Runbooks and workflow guides (e.g., GitHub/Kaggle synchronization)
├── scripts/               # Lightweight CLI points for local execution and debugging
├── src/
│   ├── data/              # Dataset parsers, ISAT-to-COCO converters, split generators
│   ├── experiments/       # High-level pipeline logic (training, evaluation, split freezing)
│   └── utils/             # File I/O and path utilities
├── tests/                 # Integrity, split validation, and smoke tests
├── main.py                # Underlying execution engine for YOLO training/eval
└── kaggle_interactive_workflow.ipynb  # Main Jupyter notebook for Kaggle/Colab execution
```

---

## 🚀 Workflows

### 1. Data Preparation Utilities (Local)
If you need to manually audit or extract the ISAT annotations into COCO format:
```powershell
# Audit existing raw data
python .\code\scripts\audit_seg.py --img-dir "TomatoMAP\TomatoMAP_seg\images" --lbl-dir "TomatoMAP\TomatoMAP_seg\labels"

# Extract labels from zip
python .\code\scripts\extract_seg_labels.py --zip-path "TomatoMAP\TomatoMAP_seg.zip" --dest-dir "TomatoMAP\TomatoMAP_seg\labels"

# Direct conversion (rarely needed manually, handled by the experiment pipeline)
python .\code\scripts\convert_isat_to_coco.py --task-dir "..." --label-dir "..." --categories "...\isat.yaml" --output-dir "code\outputs\paper1\baseline_v1\coco"
```
*Note: The categories file must contain ISAT label entries with `name` fields (e.g., `- name: tomato`).*

### 2. Executing Experiments (Local CLI)
Experiments consist of freezing the dataset split, training, and evaluation. Paths and parameters are defined in JSON configurations (e.g., `code/configs/paper1/baseline_v1_yolov11.json`).

```powershell
# 1. Freeze canonical split safely
python .\code\scripts\prepare_paper1_baseline.py --config ".\code\configs\paper1\baseline_v1_yolov11.json" --freeze-split

# 2. Run Training
python .\code\scripts\run_paper1_baseline.py --config ".\code\configs\paper1\baseline_v1_yolov11.json" --stage train

# 3. Run Evaluation
python .\code\scripts\run_paper1_baseline.py --config ".\code\configs\paper1\baseline_v1_yolov11.json" --stage eval
```

### 3. Executing Experiments (Kaggle / Notebooks)
For Kaggle environments, executing CLI scripts via `%%bash` can be unreliable and hard to debug. Instead, use the native Python notebook:

**Open and run `code/kaggle_interactive_workflow.ipynb`**. 
This notebook:
1. Imports the `src.experiments` modules directly.
2. Allows you to inject a `selected_labels` filter manually (e.g., `selected_labels=["unripe"]`) to construct label-efficient splits dynamically.
3. Automatically saves untouched background images to `coco_dir/unlabeled_images.json`.
4. Runs the `train` and `eval` stages cleanly with inline output.

---

## 🔬 Label-Efficient & Semi-Supervised Learning
To evaluate how models perform on subsets of classes, the dataset parser respects class filtering natively.
When a list of target classes is provided (via the notebook or config overrides):
1. **Filtering**: Annotations not matching the target classes are stripped.
2. **Unlabeled Pool**: If an image loses all its annotations due to filtering (or had none to begin with), it is diverted entirely from the supervised `train`/`val` sets.
3. **Export**: These orphaned images are recorded in `unlabeled_images.json` in the active output directory, providing a perfect ingestion point for Pseudo-labeling or Mean Teacher architectures later.

---

## ✅ Testing
To run the lightweight suite of dataset parsing, integrity, and split validation checks:
```powershell
python -m unittest discover -s ".\code\tests" -p "test_*.py"
```
