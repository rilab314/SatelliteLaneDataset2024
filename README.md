# SEED-MAP data framework
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
Comprehensive usage instructions and sample scripts are currently under preparation and will be made available in a future update.


## Sample Image Data
Sample image data for testing and demonstration purposes will be included in a subsequent release.
