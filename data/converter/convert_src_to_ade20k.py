import os
import cv2
import json
import shutil
import numpy as np
from glob import glob
from tqdm import tqdm

import data.converter.utils.generate_train_val_test_coords as gen_train_val_test_coords
import src.config.config as cfg
from src.utils.labels_to_segmentation_labels import generate_segmentation_labels

def convert_src_to_ade20k(src_dir):
    os.makedirs(src_dir.replace('segmentation_label', 'ade20k_label'), exist_ok=True)
    src_label_paths = sorted(glob(os.path.join(src_dir, '*.json')))
    for src_label_path in tqdm(src_label_paths):
        semantic_image = generate_semantic_image(src_label_path)
        cv2.imwrite(src_label_path.replace('segmentation_label', 'ade20k_label').replace('.json', '.png'), semantic_image)

def generate_semantic_image(src_label_path):
    semantic_image = np.ones((768, 768), dtype=np.uint8)
    with open(src_label_path, 'r') as f:
        data = json.load(f)
    for road_object in data[1:]:
        points = road_object['image_points']
        polygon_points = np.array(points, dtype=np.int32)
        cv2.fillPoly(semantic_image, [polygon_points], 2)
    return semantic_image

def generate_train_val_coords(drive, json_save_path, lon_range, lat_range):
    label_paths = glob(os.path.join(drive, "*.json"))

    dataset = {'train': [],
               'validation': []}
    for label_path in label_paths:
        label_lon, label_lat = os.path.basename(label_path)[:-5].split(',')
        if lon_range[0] < float(label_lon) < lon_range[1] and lat_range[0] < float(label_lat) < lat_range[1]:
            dataset['validation'].append(os.path.basename(label_path)[:-5])
        else:
            dataset['train'].append(os.path.basename(label_path)[:-5])
    with open(json_save_path, 'w') as f:
        json.dump(dataset, f)

def divide_train_val(src_path, dst_path, coords_json_path):
    with open(coords_json_path, 'r') as f:
        coords = json.load(f)
    os.makedirs(os.path.join(dst_path, 'images', 'training'), exist_ok=True)
    os.makedirs(os.path.join(dst_path, 'images', 'validation'), exist_ok=True)
    os.makedirs(os.path.join(dst_path, 'annotations', 'training'), exist_ok=True)
    os.makedirs(os.path.join(dst_path, 'annotations', 'validation'), exist_ok=True)

    for coord in tqdm(coords['train'], desc='Copy training dataset'):
        label_path = os.path.join(src_path, 'ade20k_label', str(coord)+'.png')
        moved_label_path = os.path.join(dst_path, 'annotations', 'training', str(coord)+'.png')
        shutil.copy(label_path, moved_label_path)
        image_path = os.path.join(src_path, 'image', str(coord)+'.png')
        moved_image_path = os.path.join(dst_path, 'images', 'training', str(coord)+'.png')
        shutil.copy(image_path, moved_image_path)

    for coord in tqdm(coords['validation'], desc='Copy validation dataset'):
        label_path = os.path.join(src_path, 'ade20k_label', str(coord)+'.png')
        moved_label_path = os.path.join(dst_path, 'annotations', 'validation', str(coord)+'.png')
        shutil.copy(label_path, moved_label_path)
        image_path = os.path.join(src_path, 'image', str(coord)+'.png')
        moved_image_path = os.path.join(dst_path, 'images', 'validation', str(coord)+'.png')
        shutil.copy(image_path, moved_image_path)

if __name__ == '__main__':
    coords_save_path = os.path.join(cfg.DATASET_PATH, 'dataset.json')
    generate_segmentation_labels()
    convert_src_to_ade20k(cfg.SEGMENTATION_LABEL_PATH)
    gen_train_val_test_coords.generate_train_val_test_coords(cfg.LABEL_PATH, coords_save_path, cfg.DATASET_RATIO, [cfg.SEOUL_CONFIG, cfg.INCHEON_CONFIG])
    divide_train_val(cfg.DATASET_PATH, cfg.CUSTOM_ADE20K_PATH, coords_save_path)