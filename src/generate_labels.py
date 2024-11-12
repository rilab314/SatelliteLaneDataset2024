import json
import os.path
import numpy as np
from glob import glob
from typing import List, Tuple
from pyproj import Transformer
from tqdm import tqdm

import src.config.config as cfg
from src.utils.json_file_io import JsonFileReader, write_to_json
from src.utils.shape_file_reader import ShapeFileReader
from src.dto import GeometryObject, RoadObject, MetaData


def generate_labels():
    os.makedirs(cfg.LABEL_PATH, exist_ok=True)

    img_center_coords = read_coordinates(filename=cfg.COORD_LIST_PATH)
    img_tlbr_coords = convert_to_tlbr(img_center_coords)
    reader = JsonFileReader(cfg.JSON_PATH)
    # reader = ShapeFileReader(cfg.SHAPE_PATH)  # 이걸로도 똑같이 할 수 있게
    json_list = reader.list_files()
    for file_path in tqdm(json_list, desc='Generating Labels'):
        geometries = reader.read(file_path)
        global_touch_map = np.zeros((len(img_center_coords), len(geometries)), dtype=np.int32)
        for geometry_num, geometry in enumerate(geometries):
            image_mask = generate_image_mask(geometry, img_tlbr_coords, img_center_coords)
            update_global_touch_map(global_touch_map, image_mask, geometry_num)
        write_label(global_touch_map, geometries, img_tlbr_coords, img_center_coords)

def read_coordinates(filename: str) -> List[Tuple[str, str]]:
    '''
    :param filename: 
    :return: [(x1, y1), (x2, y2), (x3, y3), ...]
    좌표값은 모든 자릿수가 반드시 문자열로 보존이 되어야 해당 파일을 열 수 있다. float이나 np.array로 바꿔선 안됨
    '''
    with open(filename, 'r') as f:
        return json.load(f)


def convert_to_tlbr(center_coords: List[Tuple[str, str]]) -> np.ndarray:
    half_width_lon = cfg.ONE_PIXEL_LONGITUDE * cfg.IMAGE_SIZE_w
    half_height_lat = cfg.ONE_PIXEL_LATITUDE * cfg.IMAGE_SIZE_h
    center_lonlat = np.array([(float(x), float(y)) for x, y in center_coords])
    lonlat_to_web = Transformer.from_crs('EPSG:4326', 'EPSG:3857')
    x_min, y_min = lonlat_to_web.transform(center_lonlat[:, 1] - half_height_lat, center_lonlat[:, 0] - half_width_lon)
    x_max, y_max = lonlat_to_web.transform(center_lonlat[:, 1] + half_height_lat, center_lonlat[:, 0] + half_width_lon)


    return np.vstack([x_min, y_min, x_max, y_max]).T


def generate_image_mask(geometry: GeometryObject, tlbr_coords: np.ndarray, center_coords: List[Tuple[str, str]]):
    """
    :param geometry:
    :param tlbr_coords: (N, 4) [(x1,y1,x2,y2), ...] 
    :param center_coords: (N, 2) [(x1, y1), (x2, y2), (x3, y3), ...]
    :return: 
    """
    tlbr_coords = tlbr_coords[:, np.newaxis]  # (N, 1, 4)

    image_mask = np.zeros(tlbr_coords.shape[0], dtype=bool)  # (N,)

    if geometry.geometry_type in ['MULTILINE_STRING', 'MULTIPOLYGON']:
        for coords in geometry.coordinates:
            mask = calculate_mask(coords, tlbr_coords)
            # mask: (N,)
            image_mask |= np.any(mask, axis=1)
    else:
        mask = calculate_mask(geometry.coordinates, tlbr_coords)
        # mask: (N,)
        image_mask = np.any(mask, axis=1)
    return image_mask

def calculate_mask(coords, tlbr_coords):
    # tlbr_coords (N, 1, 4)
    coordinates = np.array(coords)[np.newaxis]  # (1, M, 2)
    # mask: (N, M)
    mask = (tlbr_coords[..., 0] < coordinates[..., 0]) & (tlbr_coords[..., 1] < coordinates[..., 1])
    mask &= (tlbr_coords[..., 2] > coordinates[..., 0]) & (tlbr_coords[..., 3] > coordinates[..., 1])
    return mask




def update_global_touch_map(global_touch_map, image_mask, geometry_num):
    global_touch_map[image_mask, geometry_num] += 1

def write_label(global_touch_map, geometries, img_tlbr_coords, img_center_coords):
    for geo_touch_map, tlbr, center in zip(global_touch_map, img_tlbr_coords, img_center_coords):
        if not np.all(geo_touch_map == 0):
            valid_geometries = np.array(geometries)[geo_touch_map==True]
            filepath = to_label_filepath(center)
            update_file(filepath, valid_geometries.tolist(), tlbr)


def to_label_filepath(center_coord: Tuple[str, str]) -> str:
    file_coord = center_coord[0]+","+center_coord[1]
    file_path = os.path.join(cfg.LABEL_PATH, f"{file_coord}.json")
    return file_path


def update_file(filename: str, geometries: List[GeometryObject], tlbr: np.ndarray):
    reader = JsonFileReader()
    road_objects = reader.read(filename)
    if not road_objects:
        road_objects = [MetaData(type='matadata',
                                     image_x1y1x2y2=tlbr.tolist(),
                                     coordinate_format='webmercator',
                                     format_code='EPSG:3857',
                                     region=cfg.REGION)]

    id_list = search_objects_ids(road_objects)

    for geometry in geometries:
        if geometry.id not in id_list:
            new_road_objects = convert_to_road_object(geometry, tlbr, filename)
            road_objects += [new_road_objects]
    write_to_json(filename, road_objects)


def convert_to_road_object(geometry, tlbr, filename):
    image_points = convert_geometry_to_image_points(geometry, tlbr)
    category_name = cfg.KindDict[geometry.kind]
    # if geometry.type == '101':
    #     geometry.type = '111'
    type_name = cfg.TypeDict[geometry.type]
    image_id = os.path.basename(filename).split('_')[0]

    return RoadObject(id=geometry.id,
                      category_id=geometry.kind,
                      type_id=geometry.type,
                      category=category_name,
                      type=type_name,
                      geometry_type=geometry.geometry_type,
                      image_points=image_points,
                      global_points=geometry.coordinates,
                      image_id=image_id)


def convert_geometry_to_image_points(geom, tlbr):
    image_points = []
    if geom.geometry_type in ['MULTILINE_STRING', 'MULTIPOLYGON']:
        for ge in geom.coordinates:
            image_points.append(coords_to_image_points(ge, tlbr).tolist())
    else:
        image_points = coords_to_image_points(geom.coordinates, tlbr).tolist()
    return image_points

def coords_to_image_points(coords, tlbr):
    np_coords = np.array(coords)
    x = np_coords[:, 0]
    y = np_coords[:, 1]
    x_pixels = ((cfg.IMAGE_SIZE_w * (x - tlbr[0]) / (tlbr[2] - tlbr[0])).astype(np.int32))
    y_pixels = ((cfg.IMAGE_SIZE_h * (tlbr[3] - y) / (tlbr[3] - tlbr[1])).astype(np.int32))
    image_points = np.stack((x_pixels, y_pixels), axis=-1)
    return image_points

def search_objects_ids(road_objects: List):
    return [road_object.id for road_object in road_objects if isinstance(road_object, RoadObject)]

if __name__ == '__main__':
    generate_labels()
