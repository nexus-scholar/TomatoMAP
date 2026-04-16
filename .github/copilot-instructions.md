# TomatoMAP repository instructions

This repository supports PhD research on tomato detection and instance segmentation using TomatoMAP, with emphasis on reproducible experiments and label-efficient segmentation.

## Main objective
Help maintain a clean, reproducible research codebase for experiments, paper writing, and thesis work.

## Repository understanding
- `TomatoMAP-Cls/`, `TomatoMAP-Det/`, and `TomatoMAP_seg/` are dataset folders and should be treated as data sources.
- `code/` is the target location for reusable Python logic.
- `kaggle/` contains Kaggle-specific training and execution helpers.
- `tests/` contains integrity and smoke tests.
- `research/docs/research-context/` contains research notes, gap statements, plans, and writing support.
- Root scripts such as `check_missing_labels.py`, `extract_labels.py`, `find_missing_images.py`, and `analyze_seg_labels.py` may contain transitional logic that should gradually be moved into `code/`.
- Notebooks are exploratory. Prefer reusable Python modules for final workflows.

## Ground rules
- Do not modify raw dataset files unless explicitly asked.
- Do not assume unavailable segmentation labels exist.
- Do not fabricate annotations, counts, metrics, or results.
- Do not silently alter train/val/test splits.
- Avoid broad refactors without presenting a plan first.
- Prefer small, reversible edits.

## Coding preferences
- Use Python.
- Prefer modular functions and light CLIs.
- Avoid hardcoded absolute machine-specific paths.
- Use clear names and simple folder-local imports when possible.
- Add run instructions for any new script.
- Keep output paths explicit.

## Experiment expectations
Every experiment-related script should make these explicit:
- dataset path
- split source
- class mapping
- model name
- image size
- epochs
- batch size
- output directory
- seed where relevant

## Validation expectations
Before claiming a change is complete, prefer lightweight checks such as:
- import check
- path existence check
- split count check
- image-label matching check
- small dry-run or no-op mode if available

## Collaboration style
Act like a cautious research engineer.
If context is missing, ask a targeted question instead of guessing.
Optimize for reproducibility, correctness, and paper-readiness.
