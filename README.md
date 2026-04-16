# TomatoMAP: *Solanum lycopersicum* (Tomato) Multi-Angle Multi-Pose Dataset for Fine-Grained Phenotyping

Observer bias and inconsistencies in traditional plant phenotyping methods limit the accuracy and reproducibility of fine-grained plant analysis. To overcome these challenges, we developed TomatoMAP, a comprehensive dataset for *Solanum lycopersicum* using an Internet of Things (IoT) based imaging system with standardized data acquisition protocols. Our dataset contains 64,464 RGB images that capture 12 different plant poses from four camera elevation angles. Each image includes manually annotated bounding boxes for seven regions of interest (ROIs), including leaves, panicle, batch of flowers, batch of fruits, axillary shoot, shoot and whole plant area, along with 50 fine-grained growth stage classifications based on the BBCH scale. Additionally, we provide 3,616 high-resolution image subset with pixel-wise semantic and instance segmentation annotations for fine-grained phenotyping. We validated our dataset using a cascading model deep learning framework combining MobileNetv3 for classification, YOLOv11 for object detection, and MaskRCNN for segmentation. Through AI vs. Human analysis involving five domain experts, we demonstrate that the models trained on our dataset achieve accuracy and speed comparable to the experts. Cohen’s Kappa and inter-rater agreement heatmap confirm the reliability of automated fine-grained phenotyping using our approach. Details can be found in our homepage: https://0yj.github.io/tomato_map/ 

#### Repo Structure

This repository contains two folders:

- `metadata` contains meta data of TomatoMAP, including comprehensive manual and semi-automatic phenotyping data for 101 tomato plant samples. The data covers growth measurements, developmental stages tracking, and imaging meta. Data is generated from a controlled greenhouse environment at Julius Kuehn Institute experiment with automated imaging systems. Details see ***README.md*** under the subfolder. 
- `TomatoMAP` contains the image data as well as the annotations. 

#### Data Generation

- Clone our code repo
```
git clone https://github.com/0YJ/TomatoMAP
cd TomatoMAP/code
```
- Please download the repository, fully unzip in path: TomatoMAP/code/
- Run through our ***TomatoMAP_builder.ipynb***, you will get TomatoMAP-Cls, TomatoMAP-Det, and the downloaded TomatoMAP-Seg.
#### Model Training

TomatoMAP project offers chance to train your own model based on our dataset. 

To train your own model, easliy run through ***TomatoMAP_trainer.ipynb***. 

#### Author Contributions

Y.Z. and S.R. designed the data station. Y.Z. designed the dataset, performed the collecting of the data, train- ing of the models, and validating of the dataset. Y.Z., A.K., and S.R. wrote the manuscript, and handled the submissions of manuscript and dataset. Y.Z, A.K, and S.R designed the experiments. Y.Z, S.ST., A.K., and S.R. conducted the experiments.

#### Competing Interests

The authors have declared no conflicts of interest.

# Acknowledgments 

This project was powered by the de.NBI Cloud within the German Network for Bioinformatics Infrastructure (de.NBI) and ELIXIR-DE (Forschungszentrum J¨ulich and W-de.NBI-001, W-de.NBI-004, W-de.NBI-008, W- de.NBI-010, W-de.NBI-013, W-de.NBI-014, W-de.NBI-016, W-de.NBI-022)

This work was fundend by the Federal Ministry of Agriculture, Food and Regional Identity.

## Reproducible Paper 1 Workflow (Current)
- `code/` is the source of truth for the new Paper 1 baseline implementation.
- `TomatoMAP/` is kept as legacy read-only reference.
- For exact GitHub push, Kaggle dataset, and Kaggle execution steps, use:
  - `code/docs/paper1_github_kaggle_runbook.md`
  - `kaggle/README.md`
