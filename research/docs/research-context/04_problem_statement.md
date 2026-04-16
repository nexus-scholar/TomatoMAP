# Problem Statement

Tomato instance segmentation is useful for greenhouse phenotyping and potentially for robotics, fruit monitoring, and plant development analysis. Recent tomato vision studies show continued interest in segmentation for fruit detection, ripeness analysis, and practical field or greenhouse use. [web:32][web:35][web:53]

However, pixel-level annotation is expensive and slow, especially for cluttered scenes, overlaps, and small or partially occluded objects. In the current project setting, the practical bottleneck is not lack of images but lack of full mask annotations. [cite:54]

The project problem can therefore be stated as:

> How can we train an effective tomato instance segmentation model on TomatoMAP-Seg when only a limited subset of masks is available, while a much larger pool of related unlabeled images exists?

## Paper-level formulation

The first paper should answer:
- whether unlabeled TomatoMAP-Seg images can improve segmentation,
- how performance changes with reduced label fractions,
- how close a label-efficient method gets to a stronger supervision setting if more labels later become available.

## Thesis-level formulation

The broader thesis asks how to perform reliable tomato perception under annotation scarcity using dataset structure, weak supervision, and efficient training.

