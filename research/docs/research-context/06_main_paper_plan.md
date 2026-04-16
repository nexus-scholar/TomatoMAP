# Main Paper Plan

## Working title

**Label-Efficient Instance Segmentation for Greenhouse Tomato Phenotyping Using Semi-Supervised Learning on TomatoMAP-Seg**

## Central idea

Use the available ~700 labeled TomatoMAP-Seg images as the supervised subset and the remaining currently unlabeled TomatoMAP-Seg images as an unlabeled pool for semi-supervised learning. [cite:54]

## Why this paper is the best first paper

- It does not depend on the missing labels arriving.
- It matches the student's 3-month deadline. [cite:55]
- It fits limited Kaggle T4 compute. [cite:55][web:77][web:79]
- It gives a cleaner novelty story than another YOLO architecture modification. [web:32][web:33][web:35][web:53]

## Main research questions

1. Can semi-supervised learning improve tomato instance segmentation over a supervised baseline trained on ~700 masks?
2. How much labeled data is really needed?
3. If extra labels arrive later, how far is the semi-supervised setting from fuller supervision?

## Expected paper contributions

1. A practical label-efficient instance segmentation pipeline for TomatoMAP-Seg.
2. A study of performance under limited annotation.
3. Evidence that unlabeled tomato images can improve segmentation quality.
4. A reproducible training protocol under modest compute resources.

## Paper boundaries

This first paper should NOT try to solve:
- multi-view geometry deeply,
- full weak-supervision from boxes,
- multiple crops,
- large multi-model benchmarking.

Those can belong to follow-up work.

