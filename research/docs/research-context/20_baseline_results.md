# Baseline Experiment Results: Detectron2

## Overview
This document records the initial baseline results for Tomato Instance Segmentation on the TomatoMAP dataset using Detectron2 (Mask R-CNN). The model is evaluated on 10 fine-grained classes.

## Results
- **Bbox AP:** 2.8%
- **Segm AP:** 8.2%
- **`fast_rcnn/fg_cls_accuracy`:** ~49%

## Analysis
- The baseline performance is very low, mainly because the model struggles to distinguish between the 10 fine-grained tomato classes (nascent, mini, unripe green tomato, semi ripe, fully ripe, and sizes 2mm-12mm).
- The model performs reasonably well on bulk classes (e.g., `unripe green tomato` scored ~27% AP, `fully ripe` scored ~15% AP).
- The model fails completely on nuanced or small size classes (e.g., `nascent`, `2mm`, `4mm`), which pulls the overall mAP down to single digits.

## Next Steps
Given the highly imbalanced and fine-grained nature of the dataset, future research efforts should focus on:
1. **Class Balancing:** Techniques to handle the severe imbalance between bulk classes and rare/small classes.
2. **Advanced Augmentation:** specialized augmentation strategies for small object detection.
3. **Label-Efficient Segmentation:** as planned in the research outline, exploring methods that reduce the reliance on exhaustive fine-grained annotations.

