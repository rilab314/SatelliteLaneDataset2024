import numpy as np
import json
from typing import List, Tuple
from pyproj import Transformer
from tqdm import tqdm

import src.config.config as cfg
from src.utils.json_file_io import JsonFileReader
from src.utils.shape_file_reader import ShapeFileReader
from src.dto import GeometryObject


def generate_coord_list():
    reader = JsonFileReader(cfg.JSON_PATH)
    # reader = ShapeFileReader(cfg.SHAPE_PATH)  # 이걸로도 똑같이 할 수 있게
    json_list = reader.list_files()
    region_configs = [cfg.SEOUL_CONFIG, cfg.INCHEON_CONFIG]
    coordinates = []
    for region_config in region_configs:
        map_cols = int((region_config['LATITUDE_RANGE'][1] - region_config['LATITUDE_RANGE'][0]) // region_config['LATITUDE_STRIDE'] + 1)
        map_rows = int((region_config['LONGITUDE_RANGE'][1] - region_config['LONGITUDE_RANGE'][0]) // region_config['LONGITUDE_STRIDE'] + 1)
        touch_map = np.zeros((map_rows, map_cols), dtype=np.int32)
        for file_path in tqdm(json_list, desc='Generating coordinate list'):
            geometries = reader.read(file_path)
            for geometry in geometries:
                update_touch_map(touch_map, geometry, region_config)

        coordinates += touch_map_to_coordinates(touch_map, region_config)
        
    write_coordinates_to_file(coordinates)


def update_touch_map(touch_map: np.array, geometry: GeometryObject, region_config: dict):
    if geometry.geometry_type in ['MULTIPOLYGON', 'MULTILINE_STRING']:
        flattened_list = [item for sublist in geometry.coordinates for item in sublist]
        lane_webmercator = np.array(flattened_list)
    else:
        lane_webmercator = np.array(geometry.coordinates)

    web_to_lonlat = Transformer.from_crs('EPSG:3857', 'EPSG:4326', always_xy=True)
    lane = lane_transform_for_numpy(lane_webmercator, web_to_lonlat)

    in_range_mask = (
            (region_config['LONGITUDE_RANGE'][0] < lane[:, 0]) & (lane[:, 0] < region_config['LONGITUDE_RANGE'][1]) &
            (region_config['LATITUDE_RANGE'][0] < lane[:, 1]) & (lane[:, 1] < region_config['LATITUDE_RANGE'][1])
    )
    lane_in_range = lane[in_range_mask]

    lon_indices = ((lane_in_range[:, 0] - region_config['LONGITUDE_RANGE'][0]) // region_config['LONGITUDE_STRIDE']).astype(int)
    lat_indices = ((lane_in_range[:, 1] - region_config['LATITUDE_RANGE'][0]) // region_config['LATITUDE_STRIDE']).astype(int)
    lon_diffs = np.abs((lon_indices * region_config['LONGITUDE_STRIDE'] + region_config['LONGITUDE_RANGE'][0]) - lane_in_range[:, 0])
    lat_diffs = np.abs((lat_indices * region_config['LATITUDE_STRIDE'] + region_config['LATITUDE_RANGE'][0]) - lane_in_range[:, 1])

    valid_points_mask = (lon_diffs < 0.000575) & (lat_diffs < 0.0004775)
    valid_lon_indices = lon_indices[valid_points_mask]
    valid_lat_indices = lat_indices[valid_points_mask]
    np.add.at(touch_map, (valid_lon_indices, valid_lat_indices), 1)


def lane_transform_for_numpy(lane: np.array, transformer: Transformer) -> np.array:
    x_coords, y_coords = lane[:, 0], lane[:, 1]
    transformed_x, transformed_y = transformer.transform(x_coords, y_coords)
    lane = np.vstack((transformed_x, transformed_y)).T
    return lane


def touch_map_to_coordinates(touch_map: np.array, region_config: dict) -> List[Tuple[str, str]]:
    coordinates = []
    for x, y in zip(*np.where(touch_map > 0)):
        coordinates.append((str(x* region_config['LONGITUDE_STRIDE'] + region_config['LONGITUDE_RANGE'][0]),
                            str(y*region_config['LATITUDE_STRIDE'] + region_config['LATITUDE_RANGE'][0])))
    return coordinates


def write_coordinates_to_file(coordinates: List[Tuple[str, str]]):
    save_path = cfg.COORD_LIST_PATH
    with open(save_path, 'w') as f:
        json.dump(coordinates, f)


if __name__ == '__main__':
    generate_coord_list()
