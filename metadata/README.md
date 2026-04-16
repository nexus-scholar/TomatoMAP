# TomatoMAP Metadata

## Introduction
This folder contains meta data of TomatoMAP, including comprehensive manual and semi-automatic phenotyping data for 101 tomato plant samples. The data covers growth measurements, developmental stages tracking, and imaging meta. Data is generated from a controlled greenhouse environment at Julius Kuehn Institute experiment with automated imaging systems. 

## Meta Descriptor

### 1. cabin.csv
**Description**: Greenhouse cabin configuration and environmental control settings  

| Column | Type | Unit/Values | Description |
|--------|------|-------------|-------------|
| cabinid | Integer | ID | Unique cabin identifier |
| cabinname | String | - | Cabin name |
| size | Integer | m² | Cabin capacity |
| humiditymax | Integer | % | Maximum humidity setting |
| humiditymin | Integer | % | Minimum humidity setting |
| tempmax | Integer | °C | Maximum temperature setting |
| tempmin | Integer | °C | Minimum temperature setting |
| lightmax | Integer | lx | Maximum light intensity |
| lightmin | Integer | lx | Minimum light intensity |
| autowatering | Boolean | True/False | Automatic watering system status |
| additionlight | Boolean | True/False | Supplemental lighting status |
| shadingscreen | Boolean | True/False | Shading screen availability |
| heatingmax | Integer | Level (1-4) | Maximum heating level |
| coolingmax | Integer | Level (1-4) | Maximum cooling level |

### 2. camera.csv
**Description**: Camera configuration for Phenotyper v1.0

| Column | Type | Unit/Values | Description |
|--------|------|-------------|-------------|
| cameraid | Integer | ID | Unique camera identifier |
| cameraname | String | - | Camera model |
| cameraang | Integer | 0-4 | Camera angle identifier |
| camerapos | String | - | Camera position degree |
| calibrated | Boolean | True/False | Calibration status |

### 3. fine_grained_pheno.csv
**Description**: Fine-grained phenotyping data 

| Column | Type | Unit/Values | Description |
|--------|------|-------------|-------------|
| fine_grained_phenotyping_id | Integer | ID | Unique record identifier |
| plant_phenotyping_id | Integer | ID | Foreignkey link to phenotyping.csv |
| node_sideshoot_location | String | ltree Code | Plant node location (e.g., "9", "11.1" for sub-nodes) |
| panicle_location | String | ltree Code | Panicle identifier (a, A, b, A1, A2, etc., big case = main panicle, else side panicle) |
| bud_small_num | Integer | Count | Number of small buds (<4mm) |
| bud_big_num | Integer | Count | Number of large buds (4-8mm) |
| bud_opened_num | Integer | Count | Number of opened buds |
| flower_blooming_num | Integer | Count | Number of blooming flowers |
| flower_wilted_num | Integer | Count | Number of wilted flowers |
| fruit_nascent_num | Integer | Count | Number of nascent fruits |
| fruit_small_num | Integer | Count | Number of small fruits |
| fruit_bigger_num | Integer | Count | Number of bigger fruits |
| fruit_ripe_num | Integer | Count | Number of ripe fruits |
| panicle_leaf_num | Integer | Count | Number of leaves |
| pheno_date | Date | YYYY-MM-DD | Observation date |

### 4. phenotyping.csv
**Description**: General plant phenotyping 

| Column | Type | Unit/Values | Description |
|--------|------|-------------|-------------|
| phenotypingid | Integer | ID | Unique phenotyping record |
| plant_id | Integer | ID | Link to plants.csv |
| plant_height | Integer | cm | Plant height measurement |
| nodes_number | Integer | Count | Number of nodes on main stem |
| side_shoots_number | Integer | Count | Number of side shoots |
| leaf_number | Integer | Count | Total number of leaves |
| observe_date | Date | YYYY-MM-DD | Observation date |
| trimmed | Boolean | True/False | Whether plant was trimmed |

### 5. pi.csv
**Description**: Raspberry Pi device configuration  

| Column | Type | Unit/Values | Description |
|--------|------|-------------|-------------|
| piid | Integer | ID | Unique Pi identifier |
| cabin | String | - | Associated cabin name |
| camera | Integer | ID | Associated camera ID |
| storage | Integer | ID | Associated storage ID |
| cable | Integer | ID | Associated power cable ID |
| pi_name | String | - | Device model |

### 6. plants.csv
**Description**: Plant sample metadata

| Column | Type | Unit/Values | Description |
|--------|------|-------------|-------------|
| plant_id | Integer | ID | Unique plant identifier |
| transgene | Boolean | TRUE/FALSE | Transgenic status |
| date_in | Date | MM/DD/YYYY | Experiment start date |
| date_out | Date | MM/DD/YYYY | Experiment end date |
| cabin | String | - | Cabin assignment |
| art | String | - | Species |
| accession | String | - | Accession name |
| seed | String | - | Seed source |

### 7. pose.csv
**Description**: Camera elevation angles and 12 pose id

| Column | Type | Unit/Values | Description |
|--------|------|-------------|-------------|
| poseid | Integer | ID | Unique pose identifier |
| angle | Integer | Degrees (0-330) | Rotation angle in 30° increments |

**Note**: Angles range from 0° to 330° in 30° increments, providing 12 different poses.

### 8. powercable.csv
**Description**: Power cable specifications  


| Column | Type | Unit/Values | Description |
|--------|------|-------------|-------------|
| cableid | Integer | ID | Unique cable identifier |
| cablename | String | - | Cable type/brand |
| length | Decimal | meters | Cable length |

### 9. raw_pheno.csv
**Description**: Raw image capture metadata  

| Column | Type | Unit/Values | Description |
|--------|------|-------------|-------------|
| raw_phenotypic_data_id | Integer | ID | Unique record identifier |
| plant_id | Integer | ID | Foreignkey link to plants.csv |
| position | Integer | ID | Camera position identifier |
| pose | Integer | ID | Link to pose.csv (camera angle) |
| image_set_id | Integer | ID | Image set identifier |
| image_size | String | (width, height) | Image dimensions in pixels |
| image_meta | String | Filename | Original image filename with meta |

### 10. storages.csv
**Description**: Storage device specifications  

| Column | Type | Unit/Values | Description |
|--------|------|-------------|-------------|
| storageid | Integer | ID | Unique storage identifier |
| storagename | String | - | Storage brand/model |
| size | Integer | GB | Storage capacity |

### 11. TomatoMAPSeg_meta.csv
**Description**: TomatoMAP-Seg image meta

| Column | Type | Unit/Values | Description |
|--------|------|-------------|-------------|
| Filename | String | - | Image filename |
| DateTimeOriginal | DateTime | YYYY:MM:DD HH:MM:SS | Capture timestamp |
| ImageWidth | Integer | pixels | Image width |
| ImageHeight | Integer | pixels | Image height |
| FNumber | Decimal | f-stop | Camera aperture |
| ExposureTime | String | seconds | Exposure time |
| ISO | Integer | ISO value | Camera sensitivity |
| FocalLength | String | mm | Lens focal length |
| Model | String | - | Camera model |

## Related PDF Documents

- **cabin_protocol.pdf**: Shows cabin layout with plant positions arrangment
- **label_info.pdf**: Provides detailed classification labels for TomatoMAP-Cls, TomatoMAP-Det, and TomatoMAP-Seg tasks
- **TomatoMAP-Cls_tree.txt**: Directory structure for TomatoMAP-Cls
- **TomatoMAP-Det_tree.txt**: Directory structure for TomatoMAP-Det