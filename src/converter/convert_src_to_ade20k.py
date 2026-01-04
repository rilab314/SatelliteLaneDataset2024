import os
import cv2
import json
import shutil
import numpy as np
from typing import Tuple
from glob import glob
from tqdm import tqdm
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon

import settings
import src.config.config as cfg
import src.config.config_converter as cfg_converter
import src.converter.utils.generate_train_val_test_coords as gen_train_val_test_coords
import src.config.color_map_config as color_map_config


def generate_ade20k_dataset(src_label_dir: str, dst_dir: str, coords_json_path: str, draw_color_label: bool):
    """
    Generate ADE20K dataset directly from source labels.

    Args:
        src_label_dir (str): Path to the source label JSON files.
        dst_dir (str): Destination directory to save ADE20K dataset.
        coords_json_path (str): Path to JSON file containing train/validation split.
        draw_color_label (bool): Whether to draw color-label annotations.
    """
    with open(coords_json_path, 'r') as f:
        coords = json.load(f)

    # Create necessary directories for ADE20K dataset
    os.makedirs(os.path.join(dst_dir, 'images', 'training'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'images', 'validation'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'annotations', 'training'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'annotations', 'validation'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'annotations', 'test'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'color_annotations', 'training'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'color_annotations', 'validation'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'color_annotations', 'test'), exist_ok=True)

    # Process training data
    for coord in tqdm(coords['train'], desc="Processing training dataset"):
        _process_label_and_image(coord, src_label_dir, dst_dir, 'training', draw_color_label)

    # Process validation data
    for coord in tqdm(coords['validation'], desc="Processing validation dataset"):
        _process_label_and_image(coord, src_label_dir, dst_dir, 'validation', draw_color_label)

    for coord in tqdm(coords['validation'], desc="Processing validation dataset"):
        _process_label_and_image(coord, src_label_dir, dst_dir, 'test', draw_color_label)

def _process_label_and_image(coord: str, src_label_dir: str, dst_dir: str, dataset_type: str, draw_color_label: bool):
    """
    Process a single label and corresponding image for the specified dataset type.

    Args:
        coord (str): Coordinate identifier (filename without extension).
        src_label_dir (str): Source directory containing the label JSON files.
        dst_dir (str): Destination directory for ADE20K dataset.
        dataset_type (str): "training" or "validation".
    """
    label_path = os.path.join(src_label_dir, f"{coord}.json")
    image_path = os.path.join(cfg.DATASET_PATH, 'image', f"{coord}.png")

    if not os.path.exists(label_path) or not os.path.exists(image_path):
        return  # Skip if the corresponding label or image doesn't exist

    semantic_image = generate_semantic_image(label_path)

    label_dst = os.path.join(dst_dir, 'annotations', dataset_type, f"{coord}.png")
    cv2.imwrite(label_dst, semantic_image)

    image_dst = os.path.join(dst_dir, 'images', dataset_type, f"{coord}.png")
    shutil.copy(image_path, image_dst)

    if draw_color_label:
        color_label_dst = os.path.join(dst_dir, 'color_annotations', dataset_type, f"{coord}.png")
        color_map = [meta['color'] for meta in sorted(color_map_config.KindDictColors_lanedet, key=lambda x: x['id'])]
        color_map.insert(0, (0, 0, 0))
        colormap = np.array(color_map, dtype=np.uint8)

        color_image = colormap[semantic_image]
        cv2.imwrite(color_label_dst, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))


def generate_semantic_image(label_path: str) -> np.ndarray:
    """
    Generate a semantic segmentation image directly from a JSON label file.

    Args:
        label_path (str): Path to the JSON label file.

    Returns:
        np.ndarray: Semantic segmentation image.
    """
    semantic_image = np.ones((768, 768), dtype=np.uint8) # label 1 is ignore
    category_priority = {
        key:val['priority']  for key, val in cfg_converter.ADE20K_LANE_CATEGORIES.items()
    }
    category_to_label_id = {
        val['category']:val['id']  for key, val in cfg_converter.ADE20K_LANE_CATEGORIES.items()
    }
    with open(label_path, 'r') as f:
        data = json.load(f)

    road_objects = sorted(
        data[1:],  # Exclude metadata
        # Sort road objects in reverse priority order (higher priority objects are drawn last),
        # ensuring that objects with lower priority do not obscure higher priority objects.
        key=lambda obj: -category_priority.get(obj['category_id'], float('-inf'))
    )
    for road_object in road_objects:
        if road_object['category'] in category_to_label_id:
            label_id = category_to_label_id[road_object['category']] + 1 # label 1 is ignore
            geometry_type = road_object['geometry_type']
            image_points = road_object['image_points']

            # Convert LINE_STRING to POLYGON if necessary
            if geometry_type in ['LINE_STRING', 'MULTILINE_STRING']:
                image_points = expand_line_to_polygon(image_points, buffer_size=1.5)
            else:
                continue

            # Draw polygons on the semantic image
            if isinstance(image_points[0][0], list):  # MultiPolygon case
                for polygon in image_points:
                    polygon_points = np.array(polygon, dtype=np.int32)
                    cv2.fillPoly(semantic_image, [polygon_points], label_id)
            else:  # Single Polygon case
                polygon_points = np.array(image_points, dtype=np.int32)
                cv2.fillPoly(semantic_image, [polygon_points], label_id)

    return semantic_image


def expand_line_to_polygon(image_points, buffer_size=5.):
    """
    Expand a line to a polygon using a buffer.

    Args:
        image_points (list): List of line points.
        buffer_size (float): Buffer size for expanding lines.

    Returns:
        list: Polygon points.
    """
    if isinstance(image_points[0][0], list):  # MultiLineString case
        geometry = MultiLineString(image_points)
    else:  # LineString case
        geometry = LineString(image_points)

    polygon = geometry.buffer(buffer_size, cap_style='round')
    if isinstance(polygon, Polygon):
        return [list(map(int, map(round, coord))) for coord in polygon.exterior.coords]
    elif isinstance(polygon, MultiPolygon):
        return [[list(map(int, map(round, coord))) for coord in poly.exterior.coords] for poly in polygon.geoms]


if __name__ == '__main__':
    coords_save_path = os.path.join(cfg.DATASET_PATH, 'dataset.json')
    ade20k_output_path = cfg.CUSTOM_ADE20K_PATH

    gen_train_val_test_coords.generate_train_val_test_coords(
        cfg.LABEL_PATH, coords_save_path, cfg.DATASET_RATIO, [cfg.SEOUL_CONFIG, cfg.INCHEON_CONFIG]
    )

    generate_ade20k_dataset(cfg.LABEL_PATH, ade20k_output_path, coords_save_path, draw_color_label=True)
