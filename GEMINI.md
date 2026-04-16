# GEMINI project context

You are working in a PhD research repository for TomatoMAP-based tomato detection and instance segmentation.

## Objective
Help build a reproducible research workflow for a first paper on label-efficient tomato instance segmentation.

## Project facts
- The repo contains dataset folders, scripts, notebooks, and research notes.
- The codebase should gradually move from notebook-heavy work toward reusable Python modules.
- The segmentation work must rely only on verified released labels.
- Reproducibility is more important than aggressive refactoring.

## Folder meanings
- `code/` = reusable implementation
- `tests/` = validation and integrity checks
- `kaggle/` = Kaggle-specific execution helpers
- `research/docs/research-context/` = long-lived research memory and writing support
- dataset folders = source data; do not modify casually

## Rules
- Never fabricate labels, experiment outcomes, or missing metadata.
- Prefer minimal diffs.
- Ask before major restructuring.
- When moving logic, preserve the original behavior.
- When creating scripts, include clear run commands.
- When proposing refactors, explain risk and benefit.

## Default working style
- Start by summarizing the task briefly.
- If the task touches multiple files, propose a short plan.
- Prefer one well-scoped step at a time.
- Keep code simple and reproducible.
