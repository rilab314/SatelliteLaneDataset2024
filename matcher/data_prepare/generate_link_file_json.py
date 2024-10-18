import numpy as np
import cv2
import math
import geopandas as gpd
import pandas as pd
import os
import json
from glob import glob
from PIL import Image
from pyproj import Transformer
from shapely.ops import transform
from shapely.geometry import LineString, Polygon
from tqdm import tqdm


from matcher.config import config
from matcher.dataclasses_roadobject import RoadObject
from matcher.file_io import serialize_dataclass, deserialize_dataclass, save_json_with_custom_indent

# 파일 이름에서 WebMercator 좌표 추출
def extract_coords(filename):
    parts = filename.split('_')[1].replace('.png', '').split(',')
    return list(map(float, parts))


def transform_geometry(geometry, transformer):
    if geometry is None:
        return None

    def transform_coords(coords):
        # Convert to numpy array for vectorized transformation
        coords_np = np.array(coords)
        transformed = transformer.transform(coords_np[:, 0], coords_np[:, 1])
        # Reshape back into the original coordinate form
        return np.column_stack(transformed)

    if geometry.geom_type == "LineString":
        return LineString(transform_coords(geometry.coords))

    elif geometry.geom_type == "Polygon":
        # Transform exterior and interiors in a vectorized way
        transformed_exterior = LineString(transform_coords(geometry.exterior.coords))

        transformed_interiors = [
            LineString(transform_coords(interior.coords)) for interior in geometry.interiors
        ]

        return Polygon(transformed_exterior, transformed_interiors)

    else:
        # For other types of geometries, use the default approach
        return transform(lambda x, y, z=0: transformer.transform(x, y), geometry)


# # 도로 데이터를 픽셀 좌표로 변환
# def convert_geometry_to_pixels(geom):
#     return [coords_to_pixels(x, y, x_min, y_max, x_max, y_min, width, height) for x, y, *_ in np.array(geom.bounds)]
def convert_geometry_to_pixels(geom):
    # geom이 Polygon 또는 LineString과 같은 단일 지오메트리 객체라고 가정
    if isinstance(geom, Polygon):
        exterior_coords = [coords_to_pixels(x, y, x_min, y_max, x_max, y_min, width, height) for x, y in
                           np.array(geom.exterior.coords)]
        return exterior_coords
    elif isinstance(geom, LineString):
        return [coords_to_pixels(x, y, x_min, y_max, x_max, y_min, width, height) for x, y in np.array(geom.coords)]
    return []

def clip_pixels(pixel_coords, width, height):
    np_pixel_coords = np.array(pixel_coords)
    valid_coords = ((np_pixel_coords[:, 0] < width) & (np_pixel_coords[:, 0] >= 0) &
                    (np_pixel_coords[:, 1] < height) & (np_pixel_coords[:, 1] >= 0))
    if np.any(valid_coords):
        return pixel_coords
    else:
        return []

# 좌표를 이미지 픽셀로 변환
def coords_to_pixels(x, y, x_min, y_max, x_max, y_min, width, height):
    x_pixel = int((width * (x - x_min) / (x_max - x_min)) * ratio + x_ad)
    y_pixel = int((height * (y - y_max) / (y_min - y_max)) * ratio + y_ad)
    return x_pixel, y_pixel


def draw_roads(img, geometries, type):
    if type in ["1", "5"]:
        if len(geometries) > 1:
            cv2.polylines(img, [np.array(geometries, dtype=np.int32)], True, (255, 255, 0), 1)
    else:
        prev_point = None
        if np.all(np.isnan(geometries)) or np.all(np.isnan(geometries)):
            return img
        for point in geometries:
            if prev_point is not None:
                cv2.line(img, prev_point, point, (0, 255, 255), 1)
            prev_point = point

    return img



def find_files(root_folder, file_name_to_find):
    found_files = []
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if os.path.join(subdir, file).endswith(file_name_to_find):
                found_files.append(os.path.join(subdir, file))
    return found_files





def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def serialize_geometry(geometry):
    if isinstance(geometry, Polygon):
        exterior = [(x, y) for x, y in geometry.exterior.coords]
        return exterior
    elif isinstance(geometry, LineString):
        return [(x, y) for x, y in geometry.coords]
    return None

def prepare_data_for_json(total_road_links):
    output_data = {
        'geometry': [],
        'ID': [sublist.tolist() for sublist in total_road_links['ID']],
        'kind': [sublist.tolist() for sublist in total_road_links['kind']],
        'type': [sublist.tolist() for sublist in total_road_links['type']]
    }
    for geometries in total_road_links['geometry']:
        serialized_geometries = []
        for geom in geometries:
            serialized_geom = serialize_geometry(geom)
            serialized_geometries.append(serialized_geom)
        output_data['geometry'].append(serialized_geometries)
    return output_data


if __name__ == '__main__':
    root_folder = '/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/국토정보플랫폼/국토지리/unzip'
    path_list = glob(os.path.join(root_folder, "*"))
    for path in path_list:
        path_name = os.path.basename(path)
        surface_links_paths = find_files(path, '/HDMap_UTM52N_타원체고/B3_SURFACEMARK.shp')
        surface_links_paths.sort()

        lane_links_paths = find_files(path, '/HDMap_UTM52N_타원체고/B2_SURFACELINEMARK.shp')
        lane_links_paths.sort()

        total_surface_links = {"geometry": [], "ID": [], "kind": [], "type": []}
        error_files = []
        UTM_to_webmer_transformer = Transformer.from_crs('EPSG:32652', 'EPSG:3857', always_xy=True)
        for r_l_path in tqdm(surface_links_paths, desc="Transforming Road Datas"):
            try:
                road_links = gpd.read_file(r_l_path)
                road_links['transformed_geometry'] = road_links['geometry'].apply(transform_geometry, transformer=UTM_to_webmer_transformer)
                if len(road_links["transformed_geometry"]) > 1:
                    total_surface_links["geometry"].append(road_links["transformed_geometry"])
                    total_surface_links["ID"].append(road_links['ID'])
                    total_surface_links["kind"].append(road_links['Kind'])
                    total_surface_links["type"].append(road_links['Type'])
            except Exception as e:
                print("Error: ", e)

            # if len(total_surface_links["geometry"]) == 4:
            #     break

        total_lane_links = {"geometry": [], "ID": [], "kind": [], "type": []}
        for r_l_path in tqdm(lane_links_paths, desc="Transforming Road Datas"):
            try:
                road_links = gpd.read_file(r_l_path)
                road_links['transformed_geometry'] = road_links['geometry'].apply(transform_geometry, transformer=UTM_to_webmer_transformer)
                if len(road_links["transformed_geometry"]) > 1:
                    total_lane_links["geometry"].append(road_links["transformed_geometry"])
                    total_lane_links["ID"].append(road_links['ID'])
                    total_lane_links["kind"].append(road_links['Kind'])
                    total_lane_links["type"].append(road_links['Type'])
            except Exception as e:
                print("Error: ", e)
            #
            # if len(total_lane_links["geometry"]) == 4:
            #     break

        # total_road_links = {"geometry": total_surface_links["geometry"]+total_lane_links["geometry"],
        #                     "ID": total_surface_links["ID"] + total_lane_links["ID"],
        #                     "kind": total_surface_links["kind"] + total_lane_links["kind"],
        #                     "type": total_surface_links["type"] + total_lane_links["type"]}

        total_road_links = {"geometry": total_lane_links["geometry"] + total_surface_links["geometry"],
                            "ID": total_lane_links["ID"] + total_surface_links["ID"],
                            "kind": total_lane_links["kind"] + total_surface_links["kind"],
                            "type": total_lane_links["type"] + total_surface_links["type"]}

        prepared_data = prepare_data_for_json(total_road_links)
        save_to_json(prepared_data, os.path.join(config.SaveFolder, f"{path_name}.json"))





