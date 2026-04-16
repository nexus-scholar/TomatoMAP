# Experiment Plan

## Main principle

Keep the experiment matrix small and directly tied to the main claim.

## Recommended model scope

Primary model:
- YOLOv8n-seg.

Optional secondary model:
- YOLOv8s-seg, only if runtime remains manageable.

No third model family unless required by a supervisor or reviewer.

## Core experiments

### E1. Supervised baseline
Train on the ~700 labeled images only. [cite:54]

Purpose:
- establish the main baseline,
- obtain qualitative failure cases,
- create the first result table.

### E2. Semi-supervised main experiment
Train using:
- labeled subset: ~700 images,
- unlabeled subset: remaining TomatoMAP-Seg images without current labels. [cite:54]

Possible strategy:
- pseudo-labeling,
- teacher-student framework,
- confidence filtering,
- weak/strong augmentation pairing.

### E3. Label fraction ablation
Use:
- 25% of labeled set,
- 50% of labeled set,
- 100% of labeled set. [cite:54]

Purpose:
- show annotation efficiency,
- strengthen the core paper claim.

### E4. Optional future-label experiment
If full labels arrive:
- train a fuller-supervision reference model,
- compare it with the semi-supervised limited-label setup.

Purpose:
- create an upper-bound benchmark,
- strengthen the discussion without changing the paper narrative.

## Metrics

Recommended:
- mask mAP,
- precision,
- recall.

Optional:
- class-wise results if class distribution is stable enough.

## Qualitative analysis

Collect examples of:
- successful masks,
- failure under occlusion,
- failure on small objects,
- cluttered scenes,
- likely annotation-noise cases.

## Compute rule

Given the limited weekly Kaggle GPU budget, use:
- 1 short sanity-check run before any full run,
- 1â€“2 serious training runs per week,
- no large hyperparameter sweeps. [web:77][web:79]

