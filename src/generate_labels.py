import os.path

import numpy as np
from typing import List, Tuple

import src.config.config as cfg
from src.utils.json_file_io import JsonFileReader, write_to_json
from src.utils.shape_file_reader import ShapeFileReader
from src.dto import GeometryObject


def generate_labels():
    img_center_coords = read_coordinates(filename='')
    img_tlbr_coords = convert_to_tlbr(img_center_coords)
    reader = JsonFileReader(cfg.JSON_PATH)
    # reader = ShapeFileReader(cfg.SHAPE_PATH)  # 이걸로도 똑같이 할 수 있게
    json_list = reader.list_files()
    for file_path in json_list:
        geometries = reader.read(file_path)
        for geometry in geometries:
            update_labels(geometry, img_tlbr_coords, img_center_coords)


def read_coordinates(filename: str) -> List[Tuple[str, str]]:
    '''
    :param filename: 
    :return: [(x1, y1), (x2, y2), (x3, y3), ...]
    좌표값은 모든 자릿수가 반드시 문자열로 보존이 되어야 해당 파일을 열 수 있다. float이나 np.array로 바꿔선 안됨
    '''
    pass


def convert_to_tlbr(center_coords: List[Tuple[str, str]]) -> np.ndarray:
    pass


def update_labels(geometry: GeometryObject, tlbr_coords: np.ndarray, center_coords: List[Tuple[str, str]]):
    """
    :param geometry: 
    :param tlbr_coords: (N, 4) [(x1,y1,x2,y2), ...] 
    :param center_coords: (N, 2) [(x1, y1), (x2, y2), (x3, y3), ...]
    :return: 
    """
    updated_files = []
    coordinates = np.array(geometry.coordinates)[np.newaxis]  # (1, M, 2)
    tlbr_coords = tlbr_coords[:, np.newaxis]  # (N, 1, 4)
    # mask: (N, M) N = # images, M = geometries in shape file
    mask = (tlbr_coords[..., 0] < coordinates[..., 0]) & (tlbr_coords[..., 1] < coordinates[..., 1])
    mask &= (tlbr_coords[..., 2] > coordinates[..., 0]) & (tlbr_coords[..., 3] > coordinates[..., 1])
    # mask: (N,)
    image_mask = np.sum(mask, axis=1) > 0
    tlbr_coords_new = tlbr_coords[image_mask]
    center_coords_new = center_coords[image_mask]
    
    for tlbr, center in zip(tlbr_coords_new, center_coords_new):
        filename = to_label_filename(center)
        update_file(filename, geometry, tlbr)
        

def to_label_filename(center_coords: List[str]) -> str:
    pass


def update_file(filename: str, geometry: GeometryObject, tlbr: np.ndarray):
    reader = JsonFileReader()
    old_road_objects = reader.read(filename)
    new_road_objects = convert_to_road_object(geometry, tlbr)
    road_objects = old_road_objects + new_road_objects
    write_to_json(filename, road_objects)


def convert_to_road_object(geometry, tlbr):
    pass
