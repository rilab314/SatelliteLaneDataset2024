import numpy as np
import json
from typing import List, Tuple

import src.config.config as cfg
from src.utils.json_file_io import JsonFileReader
from src.utils.shape_file_reader import ShapeFileReader
from src.dto import GeometryObject


def generate_coord_list():
    reader = JsonFileReader(cfg.JSON_PATH)
    # reader = ShapeFileReader(cfg.SHAPE_PATH)  # 이걸로도 똑같이 할 수 있게
    json_list = reader.list_files()
    map_cols = int((cfg.LATITUDE_RANGE[1] - cfg.LATITUDE_RANGE[0]) // cfg.LATITUDE_STRIDE)
    map_rows = int((cfg.LONGITUDE_RANGE[1] - cfg.LONGITUDE_RANGE[0]) // cfg.LONGITUDE_STRIDE)
    touch_map = np.zeros((map_rows, map_cols), dtype=np.int32)
    for file_path in json_list:
        geometries = reader.read(file_path)
        for geometry in geometries:
            update_touch_map(touch_map, geometry)

    coordinates = touch_map_to_coordinates(touch_map)
    write_coordinates_to_file(coordinates)


def update_touch_map(touch_map: np.array, geometry: GeometryObject):
    lane = np.array(geometry.coordinates)
    in_range_mask = (
            (cfg.LONGITUDE_RANGE[0] < lane[:, 0]) & (lane[:, 0] < cfg.LONGITUDE_RANGE[1]) &
            (cfg.LATITUDE_RANGE[0] < lane[:, 1]) & (lane[:, 1] < cfg.LATITUDE_RANGE[1])
    )
    lane_in_range = lane[in_range_mask]

    lon_indices = ((lane_in_range[:, 0] - cfg.LONGITUDE_RANGE[0]) // cfg.LONGITUDE_STRIDE).astype(int)
    lat_indices = ((lane_in_range[:, 1] - cfg.LATITUDE_RANGE[0]) // cfg.LATITUDE_STRIDE).astype(int)
    lon_diffs = np.abs((lon_indices * cfg.LONGITUDE_STRIDE + cfg.LONGITUDE_RANGE[0]) - lane_in_range[:, 0])
    lat_diffs = np.abs((lat_indices * cfg.LATITUDE_STRIDE + cfg.LATITUDE_RANGE[0]) - lane_in_range[:, 1])

    valid_points_mask = (lon_diffs < 0.000575) & (lat_diffs < 0.0004775)
    valid_lon_indices = lon_indices[valid_points_mask]
    valid_lat_indices = lat_indices[valid_points_mask]

    np.add.at(touch_map, (valid_lon_indices, valid_lat_indices), 1)


def touch_map_to_coordinates(touch_map: np.array) -> List[Tuple[str, str]]:
    valid_map = touch_map[touch_map > 0]
    return [(str(x), str(y)) for x, y in valid_map]


def write_coordinates_to_file(coordinates: List[Tuple[str, str]]):
    save_path = 'coord_list'
    with open(save_path, 'w') as f:
        json.dump(coordinates, f)
