# Dataset Status

## Primary dataset

TomatoMAP is described as a tomato multi-angle multi-pose dataset for fine-grained phenotyping, with subsets for classification, detection, and segmentation. [web:39][web:40][web:43]

## Published segmentation context

The Scientific Data descriptor reports a segmentation subset with 3,616 high-resolution images and semantic/instance annotations as part of the broader dataset release. [web:40][web:43]

## Local working reality

The current local copy contains:
- more than 3k segmentation images,
- only around 700 segmentation labels available in practice,
- a missing 2k+ label portion that may be released later. [cite:54]

## Current interpretation

For research planning, the data should currently be treated as:
- `Labeled set`: ~700 images with usable segmentation labels.
- `Unlabeled pool`: remaining TomatoMAP-Seg images without labels currently available.
- `Possible future extension`: full labels if the publisher responds.

## Why this matters

This situation naturally supports a paper on:
- semi-supervised instance segmentation,
- label-efficient segmentation,
- annotation scarcity in agricultural vision.

## Important note

The project must not be blocked by waiting for the missing labels.

