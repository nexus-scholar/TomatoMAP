# Expected Folder Structure

```text
paper1-data-template/
├── dataset-metadata.json
├── README.md
├── expected-structure.md
└── data/
    └── TomatoMAP_seg/
        ├── images/
        │   ├── <image1>.JPG
        │   └── ...
        └── labels/
            ├── <image1>.json
            └── ...
```

Notes:
- Keep image and label stems aligned.
- Include only verified released segmentation labels.
- Do not include model outputs/checkpoints in this data package.

