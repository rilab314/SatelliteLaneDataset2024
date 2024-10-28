import numpy as np
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
    pass


def touch_map_to_coordinates(touch_map: np.array) -> List[Tuple[str, str]]:
    pass


def write_coordinates_to_file(coordinates: List[Tuple[str, str]]):
    pass
