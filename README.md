# TomatoMAP Research Workflow and Benchmark Artifacts

This repository is a downstream research-workflow archive built around the public
TomatoMAP / TomatoMAP-Seg dataset family. It is not the canonical TomatoMAP
dataset repository and it does not claim authorship of the original dataset,
imaging station, annotations, metadata, or upstream validation study.

The purpose of this repo is narrower: keep Mouadh Bekhouche's public
reproducibility work, benchmark preparation code, Kaggle runbooks, and
source-faithful data-processing utilities in one inspectable place.

## Provenance boundary

- Upstream TomatoMAP source work: the TomatoMAP dataset, TomatoMAP-Seg image and
  label resources, multi-angle/multi-pose acquisition design, original metadata,
  and dataset validation belong to the upstream TomatoMAP authors. Start from
  the upstream project page: https://0yj.github.io/tomato_map/ and the upstream
  repository: https://github.com/0YJ/TomatoMAP.
- This repository: contains a public research workflow that audits available
  TomatoMAP-Seg label files, converts segmentation annotations into benchmark
  formats, freezes reproducible splits, and organizes local/Kaggle execution for
  supervised baseline experiments.
- Source labels vs generated artifacts: original TomatoMAP-Seg per-image
  annotation files are treated as source labels. COCO files, split manifests,
  filtered views, and Kaggle packages are generated benchmark artifacts and
  should be cited or described as derived artifacts, not as original labels.
- Claims boundary: this repository is not evidence of a new dataset, production
  model, state-of-the-art result, or completed scientific claim. It is evidence
  of a reproducible benchmark workflow and careful experiment organization.

## What is in this repo

```text
.
+-- code/       # Main reproducible workflow: parsers, conversion, splits, tests
+-- kaggle/     # Kaggle dataset/package templates and execution notes
+-- metadata/   # Upstream TomatoMAP metadata reference files
+-- research/   # Working research notes and context; not the public claim surface
+-- pyproject.toml
+-- uv.lock
```

The current public surface should be `code/`, `kaggle/`, and `metadata/`.
The `research/` folder may contain working notes, planning context, or draft
thinking. Treat it as historical project context rather than reviewed public
documentation.

## Main workflow

The active implementation lives in `code/`.

It provides:

- ISAT / TomatoMAP-Seg label auditing utilities.
- Annotation extraction helpers.
- ISAT-to-COCO conversion code.
- Frozen split management through `code/configs/paper1/split_manifest_v1.json`.
- Experiment configuration through
  `code/configs/paper1/exp01_supervised_yolo_baseline.json`.
- Lightweight tests for split validation, conversion, extraction, and baseline
  runtime behavior.
- Kaggle-oriented runbooks for executing the workflow away from the local
  machine.

Useful entry points:

- `code/README.md` for implementation details.
- `code/docs/paper1_github_kaggle_runbook.md` for GitHub/Kaggle workflow notes.
- `kaggle/README.md` for Kaggle packaging conventions.
- `metadata/README.md` for the upstream metadata reference.

## Expected data layout

Raw TomatoMAP image and label payloads are intentionally not tracked in this
repository. Local or Kaggle runs should provide the dataset separately and keep
the raw data outside Git history.

Typical expected structure:

```text
TomatoMAP/
+-- TomatoMAP_seg/
    +-- images/
    +-- labels/
```

Generated outputs should remain untracked. The `.gitignore` excludes raw dataset
trees, model checkpoints, logs, local Kaggle credentials, and `code/outputs/`.

## Lightweight verification

This repository is designed so that code-level checks can run without launching
heavy training jobs:

```powershell
python -m unittest discover -s ".\code\tests" -p "test_*.py"
```

Training and evaluation commands depend on the dataset being mounted locally or
in Kaggle. Do not treat missing local raw data as a code failure.

## How to cite or describe this repository safely

Safe wording:

> A reproducible research workflow for auditing TomatoMAP-Seg annotations,
> converting them into benchmark-ready formats, freezing train/validation/test
> splits, and running supervised segmentation baselines on local or Kaggle
> infrastructure.

Avoid wording that implies:

- Mouadh authored the original TomatoMAP dataset.
- This repository corrected or replaced the upstream dataset.
- Generated COCO files are original labels.
- Current baseline artifacts establish a state-of-the-art result.
- Unreviewed working notes are final thesis or paper claims.

## License and upstream attribution

Before redistributing derived datasets, figures, benchmark packages, or model
outputs, check the upstream TomatoMAP license/citation requirements and the
license of any model or training framework used in the workflow.

When this repository is mentioned publicly, cite the upstream TomatoMAP project
for the dataset/source material and cite this repository only for Mouadh's
workflow, conversion, split, and benchmark-organization code.
