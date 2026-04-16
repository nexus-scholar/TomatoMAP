# Motivation and Research Gap for Label-Efficient Tomato Instance Segmentation on TomatoMAP

## 1. Application context and motivation

Tomato is a major greenhouse crop where automated perception underpins tasks such as precision harvesting, yield estimation, plant health monitoring, and phenotyping. Deep-learning-based detection and segmentation have become central tools in these workflows, enabling robots and phenotyping systems to localize fruits, leaves, flowers, and disease symptoms in complex greenhouse scenes. However, high-quality instance segmentation requires dense pixel-wise annotation, which is especially costly in tomato canopies due to occlusions, overlapping fruits, cluttered leaves, and variable lighting.[^1][^2][^3][^4][^5][^6][^7]

TomatoMAP was recently proposed as a multi-angle, multi-pose dataset for fine-grained tomato phenotyping, with coordinated subsets for classification, detection, and segmentation. This unified design strongly positions TomatoMAP as a benchmark for integrated perception tasks, yet in practical use the segmentation subset can be only partially annotated at a given time (as in the present project, where approximately 700 images are labeled and 2k+ images remain unlabeled), highlighting a real-world scenario in which dataset releases lag behind full annotation. This mismatch between image availability and mask availability motivates research into label-efficient segmentation methods that can exploit unlabeled or weakly labeled TomatoMAP images.[^8][^7][^9]

## 2. Recent tomato detection and segmentation literature (2023–2025)

### 2.1 YOLOv8-based tomato detection

In the last 2–3 years there has been a clear trend toward YOLOv8-based detection models for tomato in greenhouse environments. A 2023 study proposed a lightweight YOLOv8-based tomato detection algorithm that combines feature enhancement and attention mechanisms, achieving 93.4% mAP on a custom dataset of 3,098 images with three classes, collected in real greenhouse scenarios. Another 2023 work introduced MHSA-YOLOv8 for tomato maturity detection and counting, constructing a maturity grading dataset and reporting high accuracy for online and offline grading tasks.[^3][^4]

More recently, a 2024 Frontiers in Plant Science article presented S-YOLO, a lightweight greenhouse tomato detection model based on YOLOv8s, integrating a GSConv_SlimNeck, an improved α-SimSPPF, a β-SIoU loss, and SE attention to improve detection of small and occluded tomatoes. S-YOLO achieved 96.60% accuracy and 92.46% mAP at 74 FPS on a greenhouse dataset, underscoring the community’s focus on accuracy–speed trade-offs in supervised detection. These works collectively show that tomato detection research has rapidly converged on improving YOLOv8-style architectures using fully labeled detection datasets.[^10][^11]

### 2.2 YOLOv8-Seg and Mask R-CNN for tomato instance segmentation

Instance segmentation for tomato has likewise seen several YOLOv8-Seg-based improvements. In 2023, a study in *Agriculture* proposed an improved YOLOv8-Seg network for instance segmentation of healthy and diseased tomato plants during the growth stage. Their model integrated feature enhancement and attention modules and outperformed YOLOv8s-Seg, YOLOv5s-Seg, YOLOv7-Seg, and Mask R-CNN, achieving a segment mAP@0.5 of 92.2% and real-time inference (3.5 ms) on a custom dataset.[^2][^12]

In 2024, another work focused on tomato leaf disease detection via an improved YOLOv8-Seg instance segmentation algorithm, stressing the importance of precise delineation of leaves and symptomatic regions under varying lighting and occlusions. This study highlighted challenges that are directly relevant to TomatoMAP-Seg, such as overlapping leaves, occluded fruits, and complex greenhouse backgrounds, and again relied on a fully labeled segmentation dataset.[^6]

A 2024 paper on detecting tomato leafminer (Tuta absoluta) damage used YOLOv8-Seg to segment damage regions on tomato leaves in greenhouse conditions. The authors built an original dataset of 800 images labeled with bounding boxes and converted these to masks using the Segment Anything Model (SAM), then trained YOLOv8-Seg models at multiple scales and reported strong mAP scores for both boxes and masks at 640×640 resolution. This work neatly illustrates the community’s willingness to use SAM to accelerate labeling, but still ultimately assumes a fully supervised training regime for segmentation.[^1]

In 2025, a Frontiers in Plant Science article introduced ACP-Tomato-Seg, an improved YOLOv8s-Seg model for tomato ripeness detection and fruit segmentation in complex field environments. The model uses an Adaptive and Oriented Feature Refinement module and a Custom Multi-scale Pooling module with residual connections, plus a partial self-attention mechanism, to handle occlusion, fruit overlap, and multi-scale tomatoes across six ripeness categories, using a dataset of 1,061 images. ACP-Tomato-Seg significantly improved both bounding-box and mask mAP over vanilla YOLOv8s-Seg, again under fully supervised training.[^13]

### 2.3 Summary of recent tomato segmentation trends

Across these works, several consistent patterns emerge:

- Most recent tomato segmentation and detection studies (2023–2025) are **architecture-centric**, focusing on refinements to YOLOv8 or related detectors/segmenters: attention modules, lightweight necks, new loss functions, or improved backbones.[^4][^11][^2][^3][^6][^13]
- Datasets are typically **self-collected and fully labeled**, with sizes on the order of several hundred to a few thousand images and task-specific class definitions (ripeness categories, health status, leaf symptoms, pest damage, etc.).[^2][^3][^4][^6][^13][^1]
- Learning paradigms are almost exclusively **fully supervised**; the works treat label scarcity as a practical challenge but do not explicitly design label-efficient or semi-supervised training regimes.[^6][^13][^1][^2]

These trends indicate a methodological gap: while architectures and base detection/segmentation accuracy have improved, the question of how to perform tomato instance segmentation when only a subset of masks is available (e.g., due to delayed annotation on a large dataset like TomatoMAP) has not been directly addressed in the tomato-specific literature.

## 3. Label-efficient and semi/weakly-supervised learning in agricultural vision

Although tomato-specific work rarely focuses on label efficiency, the broader agricultural computer vision community has begun to explore semi-supervised and weakly supervised approaches to reduce annotation costs.

### 3.1 Semi-supervised segmentation for weeds and crops

A 2022 Frontiers in Plant Science paper proposed SemiWeedNet, a semi-supervised weed and crop segmentation network designed to reduce the need for large labeled datasets in field settings. SemiWeedNet integrates a multiscale enhancement module with selective kernel attention and uses online hard example mining, while consistency regularization is applied to unlabeled data to make representations robust to environmental variations. Experiments on a public weed dataset showed that SemiWeedNet outperformed fully supervised baselines while using fewer labeled examples, demonstrating the value of semi-supervised segmentation in agricultural scenarios.[^14]

While slightly older than the 2–3 year focus window, SemiWeedNet is important because it directly addresses the annotation burden in agricultural segmentation and shows that semi-supervised methods can be effective in practice.

### 3.2 Semi-supervised detection for weeds

A 2024 arXiv study evaluated semi-supervised learning frameworks for multi-class weed detection, comparing pseudo-labeling and unsupervised regression loss strategies on weed datasets. The authors emphasize that most perception algorithms in precision weed management are built under supervised learning and require large-scale labeled data, which is time-consuming and labor-intensive. Their experiments show that semi-supervised detectors can close much of the performance gap to fully supervised baselines using significantly fewer weed annotations, reinforcing the general trend toward label-efficient approaches in agriculture.[^15]

### 3.3 Semi- and self-supervised instance segmentation in plant phenotyping

Beyond weeds, there is emerging work on label-efficient instance segmentation for plant phenotyping. A recent semi-self-supervised approach proposes transforming semantic segmentation into instance-level segmentation using a GLMask representation, aiming to reduce manual labeling for densely packed objects typical of agricultural imagery. The method is evaluated on wheat head instance segmentation, achieving mAP@50 of 98.5% and outperforming conventional instance segmentation models, and also shows gains on the COCO dataset, suggesting good generality. The authors explicitly motivate their work by the high cost of creating pixel-level instance annotations for dense plant structures.[^16]

In 2025, a Frontiers in Plant Science study on zero-shot instance segmentation for plant phenotyping in vertical farming combines Grounding DINO and SAM with a vegetation-cover-aware NMS and specialized point-prompt strategies. This framework aims to perform instance segmentation without any target-specific annotations, directly addressing the scarcity of labeled images for diverse plant types in controlled environments. The method achieves better zero-shot segmentation performance than Grounded SAM and supervised YOLOv11 baselines on vertical farming datasets, further underscoring the community trend toward annotation-efficient phenotyping.[^17]

### 3.4 Weakly supervised segmentation for crop mapping

Beyond plant-level imagery, weakly supervised semantic segmentation methods have been proposed for satellite-based crop mapping. The Exact method, presented at CVPR 2025, operates on satellite image time series and uses space–time perceptive clues to train segmentation networks from image-level labels only, achieving around 95% of fully supervised performance on SITS benchmarks. This work tackles the difficulty of obtaining pixel-level labels for crop parcels and demonstrates that weakly supervised methods can deliver near-supervised performance in agricultural mapping tasks.[^18]

Taken together, these recent works in weeds, wheat phenotyping, vertical farming, and satellite crop mapping show a strong **cross-domain recognition** that label scarcity is a central barrier in agricultural vision and that semi-supervised, weakly supervised, and zero-shot methods are viable routes to reducing annotation costs.[^16][^18][^15][^17][^14]

## 4. Tomatomap and the under-explored label-efficiency space for tomatoes

TomatoMAP is a relatively new dataset that consolidates tomato images for multiple tasks—classification, detection, and segmentation—captured from multi-angle, multi-pose views in greenhouse settings. The dataset’s structure is explicitly designed to support fine-grained phenotyping and integrated perception tasks, and the Scientific Data descriptor emphasizes the availability of high-resolution segmentation images and calibration information.[^7][^9][^8]

Despite this, there is currently little or no published work that:

- uses TomatoMAP-Seg as the central benchmark for tomato instance segmentation, and
- explicitly studies **label-efficient or semi-supervised** training regimes on TomatoMAP, particularly under partial annotation scenarios.

Recent tomato instance segmentation papers (e.g., improved YOLOv8-Seg for healthy/diseased plants, YOLOv8-Seg for leaf disease and pest damage, ACP-Tomato-Seg for ripeness segmentation) all rely on fully labeled datasets and do not exploit unlabeled images at scale. Meanwhile, the broader agricultural vision and plant phenotyping communities have produced concrete examples of semi-supervised weed segmentation, semi-supervised weed detection, semi/self-supervised wheat head instance segmentation, and zero-shot instance segmentation for phenotyping, explicitly motivated by the cost of dense annotation.[^13][^15][^17][^14][^1][^2][^6][^16]

This contrast reveals a **specific gap**:

- The **ideas and methods** of label-efficient instance segmentation from the broader agricultural and phenotyping literature have **not yet been systematically applied to tomato instance segmentation**, especially on a structured, multi-task resource like TomatoMAP.
- The practical issue of **partial availability of segmentation labels** (e.g., 700 annotated TomatoMAP-Seg images with thousands more unlabeled) is not directly addressed in tomato-focused work, even though it is highly representative of real dataset release scenarios.[^7]

## 5. Concrete research gap and motivation for the proposed project

### 5.1 Research gap

Based on the recent literature (approximately 2023–2025), the following research gap can be articulated:

1. **Tomato-specific gap**: Recent tomato instance segmentation and detection research focuses on developing improved YOLOv8/YOLOv8-Seg variants and Mask R-CNN-style models trained on fully labeled, relatively small datasets. These studies rarely consider scenarios where segmentation annotations are incomplete or delayed.[^11][^3][^1][^2][^6][^13]

2. **Dataset gap**: TomatoMAP provides a unified, multi-task dataset with segmentation, detection, and classification subsets collected in multi-angle, multi-pose settings, but there is little evidence that this structure has been exploited for label-efficient segmentation or cross-task supervision in tomatoes.[^9][^8][^7]

3. **Methodological gap**: Label-efficient learning (semi-supervised, weakly supervised, zero-shot) has been successfully applied in agricultural vision for weeds, wheat phenotyping, vertical farming, and satellite crop mapping, yet these methodologies have not been extended to tomato instance segmentation despite similar annotation challenges.[^18][^15][^17][^14][^16]

### 5.2 Motivation for label-efficient TomatoMAP-Seg research

Given this gap, the motivation for a project on label-efficient instance segmentation on TomatoMAP-Seg is strong and multi-layered:

- **Practical relevance**: In real deployments, dataset releases and annotation efforts are often asynchronous; practitioners may have thousands of tomato images but only a subset annotated with instance masks, exactly as in the current TomatoMAP-Seg situation. Demonstrating that strong segmentation performance is achievable with limited masks plus unlabeled data would directly reduce annotation burden for greenhouse phenotyping and robotics.[^7]

- **Scientific novelty**: Applying semi-supervised or label-efficient instance segmentation to TomatoMAP-Seg unifies two previously separate lines of work: tomato instance segmentation (currently architecture-centric and fully supervised) and label-efficient agricultural perception (semi-supervised weeds, wheat, vertical farming, and weakly supervised crop mapping).[^15][^17][^14][^1][^2][^6][^16]

- **Dataset leverage**: TomatoMAP’s multi-task and multi-view design means that future extensions can exploit detection labels (TomatoMAP-Det) or multi-angle consistency, offering a natural progression from label-efficient segmentation toward cross-task and multi-view learning without requiring new data collection.[^8][^9][^7]

- **Alignment with current trends**: The broader plant science and agricultural AI literature is moving toward label-efficient methods (semi-supervised, weakly supervised, zero-shot) because annotation is the main bottleneck, not raw imagery. A TomatoMAP-based label-efficiency study would align tomato phenotyping research with this trend while providing a well-defined benchmark.[^17][^14][^16][^18][^15]

## 6. Resulting research questions for the project

The literature reviewed above motivates the following concrete research questions for the proposed project:

1. **Label-efficiency question**: How much tomato segmentation performance can be recovered using semi-supervised learning on TomatoMAP-Seg when only a small subset (e.g., ~700 images) has instance masks and the remainder is treated as unlabeled data?[^7]

2. **Annotation budget question**: How does instance segmentation performance vary as the fraction of labeled TomatoMAP-Seg images changes (e.g., 25%, 50%, 100% of the available masks), and can semi-supervised methods reduce the number of masks required to reach a target mAP compared to fully supervised baselines?[^14][^16][^15]

3. **Comparison to fully supervised upper bound**: If and when the missing segmentation labels become available, how close can a label-efficient TomatoMAP-Seg pipeline get to a fully supervised model trained on all segmentation annotations, in terms of mAP and robustness to occlusion, lighting, and multi-stage fruit development?[^1][^2][^6][^13]

Addressing these questions would fill a clear gap between current tomato instance segmentation practice and the emerging label-efficient paradigms in agricultural computer vision, while exploiting the unique structure of TomatoMAP as a multi-task, multi-view phenotyping resource.[^9][^16][^15][^17][^8][^7]

---

## References

1. [Determination of tomato leafminer: Tuta absoluta (Meyrick) (Lepidoptera: Gelechiidae) damage on tomato using deep learning instance segmentation method](https://link.springer.com/10.1007/s00217-024-04516-w) - Pests significantly negatively affect product yield and quality in agricultural production. Agricult...

2. [Improved YOLOv8-Seg Network for Instance Segmentation of Healthy and Diseased Tomato Plants in the Growth Stage](https://www.mdpi.com/2077-0472/13/8/1643/pdf?version=1692617981) - The spread of infections and rot are crucial factors in the decrease in tomato production. Accuratel...

3. [A Lightweight YOLOv8 Tomato Detection Algorithm Combining Feature Enhancement and Attention](https://www.mdpi.com/2073-4395/13/7/1824/pdf?version=1688896308) - A tomato automatic detection method based on an improved YOLOv8s model is proposed to address the lo...

4. [Tomato Maturity Detection and Counting Model Based on MHSA-YOLOv8](https://www.mdpi.com/1424-8220/23/15/6701) - ... main innovations of this study are summarized as follows: (1) a tomato maturity grading and coun...

5. [Tomato Anomalies Detection in Greenhouse Scenarios Based on YOLO-Dense](https://pmc.ncbi.nlm.nih.gov/articles/PMC8063041/) - Greenhouse cultivation can improve crop yield and quality, and it not only solves people’s daily nee...

6. [ENHANCING REAL-TIME INSTANCE SEGMENTATION FOR PLANT DISEASE DETECTION WITH IMPROVED YOLOV8-SEG ALGORITHM](https://ijits-bg.com/sites/default/files/archive/2024(vol.16)/No2/contents/2024-N2-03.pdf)

7. [Tomato Multi-Angle Multi-Pose Dataset for Fine-Grained Phenotyping](https://www.nature.com/articles/s41597-026-06926-9) - TomatoMAP is validated across three computer vision tasks: fine-grained BBCH-based phenological stag...

8. [0YJ/TomatoMAP - GitHub](https://github.com/0YJ/TomatoMAP) - TomatoMAP includes three subsets, TomatoMAP-Cls, TomatoMAP-Det and TomatoMAP-Seg for 50 BBCH classif...

9. [Scientific Data Descriptor Tomato Multi-Angle Multi-Pose Dataset for ...](https://arxiv.org/html/2507.11279v1)

10. [Enhanced tomato detection in greenhouse environments: a lightweight model based on S-YOLO with high accuracy](https://pmc.ncbi.nlm.nih.gov/articles/PMC11375900/) - ...intricate surroundings is essential for advancing the automation of tomato harvesting. Current ob...

11. [Enhanced tomato detection in greenhouse environments - Frontiers](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2024.1451018/full) - A lightweight greenhouse tomato object detection model named S-YOLO is proposed, based on YOLOv8s wi...

12. [PeerJ Improved greenhouse segmentation via YOLOv8 and ...](https://peerj.com/articles/cs-3665/) - 2023. Improved YOLOv8-SEG network for instance segmentation of healthy and diseased tomato plants in...

13. [Frontiers | Tomato ripeness detection and fruit segmentation based on instance segmentation](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1503256/full)

14. [Semi-supervised Learning for Weed and Crop Segmentation Using ...](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.927368/full) - In this paper, we propose a weed and crop segmentation method, SemiWeedNet, to accurately identify t...

15. [Performance Evaluation of Semi-supervised Learning Frameworks ...](https://arxiv.org/html/2403.03390v1) - In our previous review on label-efficient learning in agriculture (Li et al., 2023) , we presented v...

16. [From Semantic To Instance: A Semi-Self-Supervised Learning ...](https://arxiv.org/html/2506.16563v1)

17. [Zero-shot instance segmentation for plant phenotyping in vertical ...](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1536226/full) - This study addresses the critical issue of scarce annotated data in vertical farming by developing a...

18. [Exploring Space-Time Perceptive Clues for Weakly Supervised ...](https://cvpr.thecvf.com/virtual/2025/poster/32710) - Exact explores space-time perceptive clues to capture the essential patterns of different crop types...

