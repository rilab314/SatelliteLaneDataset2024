import numpy as np
import cv2
import math
import geopandas as gpd
import pandas as pd
import os
import json
from glob import glob
from tqdm import tqdm

from matcher.config import config, ID_name_mapping
from matcher.dataclasses_roadobject import RoadObject, RoadMetaData
from matcher.file_io import serialize_dataclass, deserialize_dataclass, save_json_with_custom_indent


def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def road_links_npy(json_path):
    road_links = load_json(json_path)
    links2numpy(road_links, json_path)





def links2numpy(road_links, json_path):
    total_npy_road_list = []
    total_id = []
    total_kind = []
    total_type = []

    max_length = 0

    for i, geometries in enumerate(tqdm(road_links["geometry"], desc="Calculating max length")):
        for geometry in geometries:
            if geometry != None:
                max_length = max(max_length, len(geometry))


    for i, (geometries, IDs, kinds, types) in enumerate(tqdm(zip(road_links["geometry"], road_links["ID"], road_links["kind"], road_links["type"]), desc="Processing road data")):
        for geometry, ID, kind_id, type_id in zip(geometries, IDs, kinds, types):
            if geometry == None:
                continue
            npy_road = np.array(geometry)

            if npy_road.shape[0] < max_length:
                padding = np.full((max_length - npy_road.shape[0], 2), np.nan)
                npy_road = np.vstack([npy_road, padding])

            total_npy_road_list.append(npy_road.tolist())
            total_id.append(ID)
            total_kind.append(kind_id)
            total_type.append(type_id)

    if len(total_npy_road_list) > 0:
        print("Saving...")
        save_path = json_path.replace("shape2webmer_json", "webmer_json2npyjson")

        # save_json_with_custom_indent(
        #     {"points": total_npy_road_list, "id": total_id, "kind": total_kind, "type": total_type}, save_path)

        with open(save_path, "w") as f:
            json.dump({"points": total_npy_road_list, "id": total_id, "kind": total_kind, "type": total_type}, f)

def generate_valid_label(numpy_json_list, image_path):
    total_valid_label = []
    for json_path in tqdm(numpy_json_list, desc="Loading road links"):
        valid_data = valid_label(image_path, json_path)
        print(valid_data)
        total_valid_label += valid_data
    p = 1

def extract_coords(image_name):
    parts = image_name.split("_")[1].replace(".png", "").split(",")
    return list(map(float, parts))

def convert_geometry_to_pixels(geom, x_min, y_min, x_max, y_max, width, height):
    coords = np.array(geom)
    x = coords[:, 0]
    y = coords[:, 1]
    x_pixels = ((width * (x - x_min) / (x_max - x_min)).astype(np.int32))
    y_pixels = ((height * (y_max - y) / (y_max - y_min)).astype(np.int32))
    pixel_coords = np.stack((x_pixels, y_pixels), axis=-1)
    return pixel_coords.tolist()

def valid_label(image_path, json_path):
    image_name = image_path.split("/")[-1]

    road_data = load_json(json_path)
    geometry = np.array(road_data["points"])
    x_min, y_min, x_max, y_max = extract_coords(image_path.split("/")[-1])
    x_coords = geometry[:, :, 0]
    y_coords = geometry[:, :, 1]

    x_in_bounds = (x_min <= x_coords) & (x_coords <= x_max)
    y_in_bounds = (y_min <= y_coords) & (y_coords <= y_max)

    points_in_bounds = x_in_bounds & y_in_bounds

    valid_indices = np.where(np.any(points_in_bounds, axis=1))[0]
    valid_data_points = geometry[valid_indices]
    valid_data_points = [point[~np.isnan(point).any(axis=1)] for point in valid_data_points]
    total_valid_data = []

    for index in valid_indices:
        ID = road_data["ID"][index]
        kind_id = road_data["kind"][index]
        type_id = road_data["type"][index]
        pixel_coords = convert_geometry_to_pixels(valid_data_points, x_min, y_min, x_max, y_max, 768, 768)
        ko_kind_name = ID_name_mapping.KindDict.get(kind_id, 'Unknown Category')
        ko_type_name = ID_name_mapping.TypeDict.get(type_id, 'Unknown Type')
        en_kind_name = ID_name_mapping.KindDict_English.get(ko_kind_name, 'Unknown Category')
        en_type_name = ID_name_mapping.TypeDict_English.get(ko_type_name, 'Unknown Type')
        data_dict = RoadObject(id=ID, category_id=kind_id, type_id=type_id, category=en_kind_name,
                                                  type=en_type_name, pixel_points=pixel_coords,
                                                  web_mercator_points=valid_data_points, image_id=int(image_name.split("_")[0]))
        total_valid_data.append(data_dict)

    return total_valid_data





if __name__ == '__main__':
    # file_path = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/국토정보플랫폼/dataset/shape2webmer_json"
    # path_list = glob(os.path.join(file_path, "*.json"))
    # path_list.sort()
    # for path in path_list:
    #     print(os.path.basename(path))
    #     road_links_npy(path)
    image_folder = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/국토정보플랫폼/dataset/768x768/origin_image"
    image_list = glob(os.path.join(image_folder, "*.png"))
    image_list.sort()
    npy_json_folder = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/국토정보플랫폼/dataset/webmer_json2npyjson"
    numpy_json_list = glob(os.path.join(npy_json_folder, "*.json"))
    numpy_json_list.sort()
    for image_path in image_list:
        generate_valid_label(numpy_json_list, image_path)