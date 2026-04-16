# Supervisor Brief

## One-minute version

The proposed first paper is on semi-supervised / label-efficient instance segmentation for TomatoMAP-Seg. This choice is driven by a practical constraint: only around 700 segmentation labels are currently available locally, while 2k+ additional labels may or may not be released later. [cite:54]

## Why this is the right first paper

- It is feasible within 3 months. [cite:55]
- It fits limited Kaggle T4 compute. [cite:55][web:77][web:79]
- It produces a clearer novelty story than another architecture-tweaking tomato segmentation paper. [web:32][web:33][web:35][web:53]

## What will be tested

- Supervised baseline on available labels.
- Semi-supervised version using unlabeled TomatoMAP-Seg images.
- Label-fraction ablation.
- Optional upper-bound experiment if missing labels arrive.

## What is needed from supervision

- Agreement on freezing the first paper scope.
- Agreement on target journal.
- Fast feedback on novelty framing and experiment sufficiency.
- Avoid expanding the first paper into a larger but riskier project.

## Key request to supervisor

Support a narrow, submission-first strategy rather than a broad thesis-perfect strategy.

