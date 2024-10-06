import numpy as np
import cv2
import math
import geopandas as gpd
import pandas as pd
import os
import numpy as np
import json
from glob import glob
from PIL import Image
from pyproj import Transformer
from shapely.ops import transform
from shapely.geometry import LineString, Polygon
from tqdm import tqdm

from matcher.road_objects import RoadObjectsProcessor
from matcher.config import config

def lonlat_clip(lon_range, lat_range, steplon, steplat):
    lon_range_diff = lon_range[1]-lon_range[0]
    lat_range_diff = lat_range[1]-lat_range[0]
    lon_index_num = int(lon_range_diff//steplon)
    lat_index_num = int(lat_range_diff//steplat)
    range_grid = np.zeros((lon_index_num, lat_index_num), dtype=int)
    road_links = road_objects_processor_links()
    for road_link in tqdm(road_links, desc="LonLatClip"):
        for lane in road_link:
            try:
                lane_np = np.array(lane.coords)
                for dot in lane_np:
                    if lon_range[0] < dot[0] < lon_range[1] and lat_range[0] < dot[1] < lat_range[1]:
                        a = int((dot[0]-lon_range[0])//steplon)
                        b = int((dot[1]-lat_range[0])//steplat)
                        range_grid[a, b] += 1
            except Exception as e:
                print(e)

    positive_indices = np.argwhere(range_grid >= 1).astype(float)
    positive_indices[:, 0] = positive_indices[:, 0] * steplon + lon_range[0]
    positive_indices[:, 1] = positive_indices[:, 1] * steplat + lat_range[0]
    positive_lonlat = positive_indices.tolist()


    with open('/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/국토정보플랫폼/'
              'array_data.json', 'w') as json_file:
        json.dump(positive_lonlat, json_file)




def road_objects_processor_links():
    road_objects_processor = RoadObjectsProcessor()
    UTM_to_lonlat = Transformer.from_crs("EPSG:32652", "EPSG:4326", always_xy=True)
    lane_shp_files = road_objects_processor.find_files(config.root_folder, config.LaneShapeFile)
    lane_links = road_objects_processor.shape_load_and_transform(lane_shp_files, UTM_to_lonlat)
    return lane_links

lonlat_clip([126.8255, 127.1516],[37.4738, 37.6202], 0.00092, 0.000764)




