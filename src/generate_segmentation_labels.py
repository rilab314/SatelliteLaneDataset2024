import json
import os.path
import numpy as np
from glob import glob
from typing import List, Tuple
from tqdm import tqdm
from shapely.geometry import LineString, Polygon, MultiLineString, MultiPolygon

import src.config.config as cfg
from src.utils.json_file_io import JsonFileReader, write_to_json
from src.dto import GeometryType ,RoadObject


def generate_segmentation_labels():
    os.makedirs(cfg.SEGMENTATION_LABEL_PATH, exist_ok=True)

    reader = JsonFileReader(cfg.LABEL_PATH)
    label_list = reader.list_files()
    for label_path in tqdm(label_list, desc='Generating Segmentation Labels...'):
        origin_geometries = reader.read(label_path)
        metadata = [origin_geometries[0]]
        obj_data_list = origin_geometries[1:]
        filtered_segmentation_data = filter_and_polygonization_data(obj_data_list)
        segmentation_label_path = label_path.replace(cfg.LABEL_PATH, cfg.SEGMENTATION_LABEL_PATH)
        write_to_json(segmentation_label_path, metadata + filtered_segmentation_data)


def filter_and_polygonization_data(obj_data_list: List[RoadObject]) -> List[RoadObject]:
    filtered_obj_data = []
    for obj in obj_data_list:
        if obj.category == 'center_line':
            if obj.geometry_type == 'LINE_STRING':
                obj.geometry_type = 'POLYGON'
            elif obj.geometry_type == 'MULTILINE_STRING':
                obj.geometry_type = 'MULTIPOLYGON'

            if len(obj.image_points):
                obj.image_points = expand_line_to_polygon(obj.image_points, 2.5)
                filtered_obj_data.append(obj)

    return filtered_obj_data


def expand_line_to_polygon(image_points, buffer_size=5.):
    if isinstance(image_points[0][0], list):  # Check if it's MultiLineString
        geometry = MultiLineString(image_points)
    else:  # It's LineString
        geometry = LineString(image_points)

    polygon = geometry.buffer(buffer_size, cap_style='round')
    if isinstance(polygon, Polygon):
        points = [list(map(int, map(round, coord))) for coord in polygon.exterior.coords]
        return points

    elif isinstance(polygon, MultiPolygon):
        # return [[list(map(int, map(round, coord))) for coord in poly.exterior.coords] for poly in polygon.geoms]
        multi_points = []
        for poly in polygon.geoms:
            points = [list(map(int, map(round, coord))) for coord in poly.exterior.coords]
            multi_points.append(points)
        return multi_points

if __name__ == '__main__':
    generate_segmentation_labels()