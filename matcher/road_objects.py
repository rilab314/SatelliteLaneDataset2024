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
from concurrent.futures import ProcessPoolExecutor, as_completed

from matcher.config import config, ID_name_mapping
from matcher.dataclasses_roadobject import RoadObject, RoadMetaData
from matcher.file_io import serialize_dataclass, deserialize_dataclass, save_json_with_custom_indent


class RoadObjectsProcessor:
    def __init__(self):
        self.image_paths = glob(config.ImagesFolder + "/origin_image/*.png")
        self.image_paths.sort()
        self.UTM_to_web_transformer = Transformer.from_crs("EPSG:32652", "EPSG:3857", always_xy=True)

    def process(self):
        print("Loading Road Links...")
        total_road_links = self.load_from_json(config.TotalRoadLinksJsonFIle)
        print("Load Completed Road Links")
        self.process_road_objects_and_save_image(total_road_links)
        # with ProcessPoolExecutor() as executor:
        #     BATCH_SIZE = 3
        #
        #     for i in range(0, len(self.image_paths), BATCH_SIZE):
        #         batch = self.image_paths[i:i + BATCH_SIZE]
        #         futures = [executor.submit(self.process_single_image, image_path, total_road_links) for image_path in batch]
        #         for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Images"):
        #             future.result()

    def find_files(self, root_folder, file_pattern):
        found_files = []
        for subdir, dirs, files in os.walk(root_folder):
            for file in files:
                if os.path.join(subdir, file).endswith(file_pattern):
                    found_files.append(os.path.join(subdir, file))
        return found_files

    def shape_load_and_transform(self, files, transformer):
        total_road_links = {"geometry": [], "ID": [], "kind": [], "type": []}
        for r_l_path in tqdm(files, desc="Transforming Road Datas"):
            try:
                road_links = gpd.read_file(r_l_path)
                road_links["transformed_geometry"] = road_links["geometry"].apply(self.transform_geometry, transformer=transformer)
                # road_links["transformed_geometry"] = self.transform_geometry(road_links["geometry"], transformer)

                if len(road_links["transformed_geometry"]) > 1:
                    total_road_links["geometry"].append(road_links["transformed_geometry"])
                    total_road_links["ID"].append(road_links['ID'])
                    total_road_links["kind"].append(road_links['Kind'])
                    total_road_links["type"].append(road_links['Type'])
            except Exception as e:
                print(e)



        return total_road_links

    def transform_geometry(self, geometry, transformer):
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


    def process_road_objects_and_save_image(self, total_road_links):

        for image_path in self.image_paths:
            road_objects = []
            image_name = image_path.split("/")[-1]
            x_min, y_min, x_max, y_max = self.extract_coords(image_name)
            width, height = Image.open(image_path).size
            metadata = RoadMetaData(type="metadata", image_x1y1x2y2=[x_min,y_max,x_max,y_min],
                                    coordinate_format="web mercator", region="Seoul, Korea")
            road_objects.append(metadata)

            # image = cv2.imread(image_path)

            for geometries, IDs, kinds, types \
                    in tqdm(zip(total_road_links["geometry"], total_road_links["ID"], total_road_links["kind"], total_road_links["type"]), desc="Drawing Road Datas"):
                for geometry, ID, kind_id, type_id in zip(geometries, IDs, kinds, types):
                    try:
                        pixel_coords = self.convert_geometry_to_pixels(geometry, type_id, x_min, y_min, x_max, y_max, width, height)
                        #
                        clipped_pixel_coords = self.clip_pixels(pixel_coords, width, height)
                        ko_kind_name = ID_name_mapping.KindDict.get(kind_id, 'Unknown Category')
                        ko_type_name = ID_name_mapping.TypeDict.get(type_id, 'Unknown Type')
                        en_kind_name = ID_name_mapping.KindDict_English.get(ko_kind_name, 'Unknown Category')
                        en_type_name = ID_name_mapping.TypeDict_English.get(ko_type_name, 'Unknown Type')
                        if len(clipped_pixel_coords) > 0:
                            road_obj = RoadObject(id=ID, category_id=kind_id, type_id=type_id, category=en_kind_name,
                                                  type=en_type_name, pixel_points=clipped_pixel_coords, web_mercator_points=geometry)
                            road_objects.append(road_obj)

                        # image = self.draw_roads(image, clipped_pixel_coords, type_id)
                    except Exception as e:
                        pass
            s_r = serialize_dataclass(road_objects)
            label_path = os.path.join(config.ImagesFolder, "label", image_name.replace(".png", ".json"))
            save_json_with_custom_indent(s_r, label_path)
            # save_root = os.path.join(config.ImagesFolder, "image", image_name)
            # cv2.imwrite(save_root, image)
            print(f"save {label_path}")

    def process_single_image(self, image_path, total_road_links):
        road_objects = []
        image_name = os.path.basename(image_path)
        x_min, y_min, x_max, y_max = self.extract_coords(image_name)
        width, height = Image.open(image_path).size
        metadata = RoadMetaData(type="metadata", image_x1y1x2y2=[x_min, y_max, x_max, y_min],
                                coordinate_format="web mercator", region="Seoul, Korea")
        road_objects.append(metadata)

        for geometries, IDs, kinds, types in zip(total_road_links["geometry"], total_road_links["ID"], total_road_links["kind"], total_road_links["type"]):
            for geometry, ID, kind_id, type_id in zip(geometries, IDs, kinds, types):
                try:
                    pixel_coords = self.convert_geometry_to_pixels(geometry, type_id, x_min, y_min, x_max, y_max, width, height)
                    clipped_pixel_coords = self.clip_pixels(pixel_coords, width, height)
                    ko_kind_name = ID_name_mapping.KindDict.get(kind_id, 'Unknown Category')
                    ko_type_name = ID_name_mapping.TypeDict.get(type_id, 'Unknown Type')
                    en_kind_name = ID_name_mapping.KindDict_English.get(ko_kind_name, 'Unknown Category')
                    en_type_name = ID_name_mapping.TypeDict_English.get(ko_type_name, 'Unknown Type')
                    if len(clipped_pixel_coords) > 0:
                        road_obj = RoadObject(
                            id=ID, category_id=kind_id, type_id=type_id,
                            category=en_kind_name,
                            type=en_type_name,
                            pixel_points=clipped_pixel_coords, web_mercator_points=geometry
                        )
                        road_objects.append(road_obj)
                except Exception:
                    pass
        label_path = os.path.join(config.ImagesFolder, "label", image_name.replace(".png", ".json"))
        s_r = serialize_dataclass(road_objects)
        save_json_with_custom_indent(s_r, label_path)
        print(f"save {label_path}")

    def extract_coords(self, image_name):
        parts = image_name.split("_")[1].replace(".png", "").split(",")
        return list(map(float, parts))

    def convert_geometry_to_pixels(self, geom, type, x_min, y_min, x_max, y_max, width, height):
        coords = np.array(geom)
        x = coords[:, 0]
        y = coords[:, 1]
        x_pixels = ((width * (x - x_min) / (x_max - x_min)).astype(np.int32))
        y_pixels = ((height * (y_max - y) / (y_max - y_min)).astype(np.int32))
        pixel_coords = np.stack((x_pixels, y_pixels), axis=-1)
        if type in ["1", "5"]:
            return pixel_coords.tolist()
        else:
            return pixel_coords.tolist()

    def clip_pixels(self, pixel_coords, width, height):
        np_pixel_coords = np.array(pixel_coords)
        valid_coords = ((np_pixel_coords[:, 0] < width) & (np_pixel_coords[:, 0] >= 0) &
                        (np_pixel_coords[:, 1] < height) & (np_pixel_coords[:, 1] >= 0))
        if np.any(valid_coords):
            return pixel_coords
        else:
            return []

    def coords_to_pixels(self, x, y, x_min, y_max, x_max, y_min, width, height):
        x_pixel = int(width * (x - x_min) / (x_max - x_min))
        y_pixel = int(height * (y - y_max) / (y_min - y_max))
        return x_pixel, y_pixel

    def draw_roads(self, img, geometries, type_id):
        if type_id in ["1", "5"]:
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

    def load_from_json(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data




if __name__ == '__main__':
    road_drawer = RoadObjectsProcessor()
    road_drawer.process()
