# SEED-MAP Data Framework
## Description
This repository contains a collection of scripts and tools designed to facilitate the generation, transformation, and utilization of the SEED-MAP dataset.



## Dataset Download
To train and evaluate the models in this repository, you need to download the required datasets. Follow the instructions below to download and organize the datasets.

### Available Datasets

| Dataset Name       | Description                                   | Download Link                                                                 | File Format | Size    |
|--------------------|-----------------------------------------------|-------------------------------------------------------------------------------|-------------|---------|
| SEED-MAP          | Original dataset consisting of image-label pairs | [Download here](https://1drv.ms/u/s!ApIuZ8oQlFPzkd0IVO8x4OeIsFGSKQ?e=VP0KdX)                                | `.zip`      | 14.1 GB   |
| COCO form          | SEED-MAP converted to COCO format | [Download here](https://1drv.ms/u/s!ApIuZ8oQlFPzkd0HS_xT2T0GtYQJ0w?e=miEozW)                                | `.zip`   | 13.6 GB   |
| ADE20K form          | SEED-MAP converted to ADE20K format | [Download here](https://1drv.ms/u/s!ApIuZ8oQlFPzkd0Gp6jZu6r0T_2_Pg?e=OPz8rq)                                | `.zip`      | 10.9 GB   |

### Directory Structure

```plaintext
datasets
├── satellite_dataset_241125
│   ├── image
│   └── label
├── satellite_coco_241213
│   ├── annotations
│   ├── test2017
│   ├── train2017
│   └── val2017
└── satellite_ade20k_241213
    ├── annotations
    │   ├── training
    │   └── validation
    └── images
        ├── training
        └── validation
```
## Usage Guide

### 1. Prepare the Dataset
Download the HD map dataset from the National Geographic Information Institute (NGII).  
[Link to NGII HD Map Data](https://map.ngii.go.kr/ms/pblictn/preciseRoadMap.do)

#### 1.1. Dataset Structure
Organize the downloaded dataset into a structured format.  
For example:
```plaintext
NGII_HD_Map
├── (B110)_Precision_Road_Map_Seoul_Northern_Ganseon_Road
│   └── SEC001_Hawolgok_Junction_To_Wangjagung_Village_Entrance
├── (B110)_Precision_Road_Map_Seoul_Junggu
│   ├── SEC001_Manrijae_Road_To_Cheongpa_Road
│   └── . . .
├── . . .
└── (B110)_Precision_Road_Map_Seoul_Route_94(Gangnam_Circulation_Road)
    └── SEC001_Geumcheon_Tollgate_To_Yangjae_IC
```


### 2. Convert HD Map Data to JSON
Use the script shape_to_json_main.py to convert the required shapefiles into JSON format for easier data manipulation.

```bash
python src/shape_to_json_main.py
```

#### Purpose:
Convert HD map shapefile data into JSON format for improved accessibility and ease of transformation.


### 3. Generate Coordinate Lists for Seoul and Incheon
Run the script generate_coord_list.py to create a list of coordinates within the Seoul and Incheon regions.

```bash
python src/generate_coord_list.py
```
#### Purpose:
Identify coordinates in the target areas for further processing.


### 4. Generate Lane and Road Marking Labels
Use the JSON-based HD map data and coordinate list to generate lane and road marking labels for the specified coordinate regions by running generate_labels.py.

```bash
python src/generate_labels.py
```
#### Purpose:
Extract labels for lanes and road markings based on the specific coordinate areas.


### 5. Align Labels with Images
Run build_dataset.py to align the generated labels with corresponding images, producing aligned labels for training.

```bash
python src/build_dataset.py
```
#### Purpose:
Create a training-ready dataset by aligning road marking labels with their respective images.

### 6. Convert to COCO Format
Run the following command to convert the dataset to COCO format:

```bash
python src/converter/convert_src_to_coco.py
```

### 7. Convert to ADE20K Format
Run the following command to convert the dataset to ADE20K format:

```bash
python src/converter/convert_src_to_ade20k.py
```
---
## Sample images by category and type
![all categories](https://github.com/user-attachments/assets/e337833e-c0c2-4368-bed3-9e320e8eaabf)





## Inference Sample Images
### Object Detection
![127 07206000000001,37 616668](https://github.com/user-attachments/assets/b3bc484d-3b39-4150-a8cd-da358b67801f)
![127 06378000000001,37 4929](https://github.com/user-attachments/assets/5f752628-5ec8-4175-992d-fd865cccd411)
![127 04906000000001,37 489079999999994](https://github.com/user-attachments/assets/fe751f86-d152-43a7-8201-ad44f9aa67b0)
![126 721585,37 5165856](https://github.com/user-attachments/assets/ec32fefc-71f4-4d66-8126-bdf58372d1fd)

### Semantic Segmentation
![127 04998,37 50436](https://github.com/user-attachments/assets/5e719f25-41dc-4a9a-935c-ec3b7d6fcce3)
![127 05274,37 48144](https://github.com/user-attachments/assets/1f7586f9-f461-43a6-8624-c2dd4fbce518)
![127 05182,37 573119999999996](https://github.com/user-attachments/assets/360e5ec2-b132-4f74-b1ce-7174c9f5ba06)
![127 05458,37 534155999999996](https://github.com/user-attachments/assets/e51854bf-4b2d-4369-a60a-c07312d66e00)

