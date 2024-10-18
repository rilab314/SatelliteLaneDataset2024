import numpy as np
import cv2
import math
import geopandas as gpd
import pandas as pd
import os
import json
from glob import glob
from PIL import Image
from pandas.core.groupby.base import groupby_other_methods
from pyproj import Transformer
from shapely.ops import transform
from shapely.geometry import LineString, Polygon
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from matcher.config import config, ID_name_mapping
from matcher.dataclasses_roadobject import RoadObject, RoadMetaData
from matcher.file_io import serialize_dataclass, deserialize_dataclass, save_json_with_custom_indent


def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def road_links_npy(json_path):
    road_links = load_json(json_path)
    links2numpy(road_links, json_path)
    npy_json_paths = glob(json_path.replace(json_path.split("/")[-1], f"roads_possible_numpy*"))
    x1y1x2y2 = [14118026.832884334,4507375.457635951,14118273.3263203,4507623.071205212]
    for path in tqdm(npy_json_paths, desc="Loading road links"):
        valid_data = valid_label(x1y1x2y2, path)
        print(valid_data)
    p=1

def links2numpy(road_links, json_path):
    total_npy_road_list = []
    total_id = []
    total_kind = []
    total_type = []

    max_length = 0
    max_length_list = []

    for i, geometries in enumerate(tqdm(road_links["geometry"], desc="Calculating max length")):
        for geometry in geometries:
            if geometry != None:
                max_length = max(max_length, len(geometry))
        if (i + 1) % 100 == 0:
            max_length_list.append(max_length)
            max_length = 0
    if max_length > 0:
        max_length_list.append(max_length)


    for i, (geometries, IDs, kinds, types) in enumerate(tqdm(zip(road_links["geometry"], road_links["ID"], road_links["kind"], road_links["type"]), desc="Processing road data")):
        for geometry, ID, kind_id, type_id in zip(geometries, IDs, kinds, types):
            if geometry == None:
                continue
            npy_road = np.array(geometry)

            if npy_road.shape[0] < max_length_list[i // 100]:
                padding = np.full((max_length_list[i // 100] - npy_road.shape[0], 2), np.nan)
                npy_road = np.vstack([npy_road, padding])

            total_npy_road_list.append(npy_road.tolist())
            total_id.append(ID)
            total_kind.append(kind_id)
            total_type.append(type_id)
        if (i + 1) % 100 == 0:
            save_path = json_path.replace(json_path.split("/")[-1], f"roads_possible_numpy_{(i + 1) % 100}.json")
            save_json_with_custom_indent(
                {"points": total_npy_road_list, "id": total_id, "kind": total_kind, "type": total_type}, save_path)
            total_npy_road_list = []
            total_id = []
            total_kind = []
            total_type = []

        if len(total_npy_road_list) > 0:
            save_path = json_path.replace(json_path.split("/")[-1], f"roads_possible_numpy_{(i + 1) % 100}.json")
            save_json_with_custom_indent(
                {"points": total_npy_road_list, "id": total_id, "kind": total_kind, "type": total_type}, save_path)

def valid_label(x1y1x2y2, json_path):
    road_data = load_json(json_path)
    geometry = np.array(road_data["points"])
    x1, y1, x2, y2 = x1y1x2y2
    x_coords = geometry[:, :, 0]
    y_coords = geometry[:, :, 1]

    x_in_bounds = (np.minimum(x1, x2) <= x_coords) & (x_coords <= np.maximum(x1, x2))
    y_in_bounds = (np.minimum(y1, y2) <= y_coords) & (y_coords <= np.maximum(y1, y2))

    points_in_bounds = x_in_bounds & y_in_bounds

    valid_indices = np.where(np.any(points_in_bounds, axis=1))[0]

    return valid_indices





if __name__ == '__main__':
    file_path = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/국토정보플랫폼/total_road_links.json"
    road_links_npy(file_path)