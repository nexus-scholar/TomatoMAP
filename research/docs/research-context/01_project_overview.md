# Project Overview

## Topic

Tomato instance segmentation for greenhouse phenotyping, with possible links to fruit development monitoring, robotics, and agricultural computer vision.

## Dataset

The project is built around TomatoMAP, a tomato multi-angle multi-pose dataset with classification, detection, and segmentation subsets. The published dataset description positions TomatoMAP as a unified resource for fine-grained phenotyping across multiple computer vision tasks. [web:39][web:40][web:43]

## Immediate problem

The segmentation subset downloaded locally contains more than 3k images, but only around 700 labels are currently available in practice according to the current working setup. The remaining 2k+ labels may be released later by the publisher, but there is no guarantee yet. [cite:54]

## Main strategic choice

The first paper should not depend on the missing labels arriving. Instead, the current lack of labels should be turned into the research problem itself:
- limited pixel-level annotations,
- many unlabeled images,
- need for label-efficient instance segmentation.

## First paper direction

**Semi-supervised / label-efficient instance segmentation on TomatoMAP-Seg**.

This is the safest direction because it remains valid whether the full missing labels arrive or not.

## Bigger thesis direction

A coherent thesis can be built around:
1. Label-efficient segmentation.
2. Weak supervision or cross-task transfer using TomatoMAP-Det.
3. Optional multi-view consistency or annotation-efficiency study.

## Why this direction is strong

Many recent tomato segmentation papers still focus on architecture modifications built on relatively small fully labeled datasets, often using YOLOv8-seg or Mask R-CNN style pipelines. [web:32][web:33][web:35][web:53]

This project is stronger if it claims:
- practical learning under annotation scarcity,
- efficient use of unlabeled data,
- reuse of TomatoMAP structure,
rather than only "a better segmentation backbone".

