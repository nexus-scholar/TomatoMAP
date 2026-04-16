# Research Gap

## What the literature already has

Recent tomato segmentation papers commonly use:
- YOLOv8-seg variants,
- Mask R-CNN style pipelines,
- architecture improvements for ripeness, overlap, or robotic picking scenarios. [web:32][web:33][web:35][web:53]

These papers are useful, but many of them are based on relatively small fully labeled custom datasets and emphasize architectural improvement more than annotation efficiency. [web:32][web:33][web:35][web:53]

## What is missing

The under-explored space is:
- learning with limited tomato mask labels,
- semi-supervised instance segmentation in greenhouse tomato settings,
- practical use of TomatoMAP as a structured multi-task dataset,
- experimental evidence for reducing annotation burden.

## Why TomatoMAP matters

TomatoMAP is more than just another image collection because it is presented as a unified dataset with classification, detection, and segmentation subsets, as well as multi-angle and multi-pose acquisition. [web:39][web:40][web:43]

That creates opportunities beyond simple supervised segmentation:
- unlabeled image use,
- cross-task transfer from detection,
- multi-view consistency,
- annotation-efficient training.

## Main novelty claim for Paper 1

The novelty is not "inventing the largest model".

The novelty is:
1. framing the real annotation scarcity as the core research problem,
2. studying label-efficient tomato instance segmentation on TomatoMAP-Seg,
3. showing a practical method that works with constrained resources.

