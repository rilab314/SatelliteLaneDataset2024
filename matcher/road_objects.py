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

from matcher.config import config, ID_name_mapping
from matcher.dataclasses_roadobject import RoadObject
from matcher.file_io import serialize_dataclass, deserialize_dataclass, save_json_with_custom_indent


class RoadObjectsProcessor:
    def __init__(self):
        self.image_paths = glob(config.ImagesFolder + "/origin_image/*.png")
        self.image_paths.sort()
        self.UTM_to_web_transformer = Transformer.from_crs("EPSG:32652", "EPSG:3857", always_xy=True)

    def process(self):
        # lane_shp_files = self.find_files(config.root_folder, config.LaneShapeFile)
        # lane_links = self.shape_load_and_transform(lane_shp_files)
        #
        # ##
        # self.lane_process(lane_links)

        # lane_shp_files = self.find_files(config.root_folder, config.LaneShapeFile)
        # lane_links = self.shape_load_and_transform(lane_shp_files, self.UTM_to_web_transformer)
        # surface_shp_files = self.find_files(config.root_folder, config.SurfaceMarkShapeFile)
        # surface_links = self.shape_load_and_transform(surface_shp_files, self.UTM_to_web_transformer)
        # #
        # total_road_links = {"geometry": lane_links["geometry"]+surface_links["geometry"],
        #                     "ID": lane_links["ID"] + surface_links["ID"],
        #                     "kind": lane_links["kind"] + surface_links["kind"],
        #                     "type": lane_links["type"] + surface_links["type"]}

        total_road_links = self.load_from_json(config.TotalRoadLinksJsonFIle)

        self.process_road_objects_and_save_image(total_road_links)

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
        if geometry is not None:
            if geometry.geom_type == "LineString":
                return LineString([transformer.transform(x, y) for x, y, *_ in geometry.coords])
            elif geometry.geom_type == "Polygon":
                exterior = geometry.exterior
                transformed_exterior = LineString([transformer.transform(x, y) for x, y, *_ in exterior.coords])

                interiors = []
                for interior in geometry.interiors:
                    transformed_interior = LineString([transformer.transform(x, y) for x, y, *_ in interior.coords])
                    interiors.append(transformed_interior)

                return Polygon(transformed_exterior, interiors)
            else:
                return transform(lambda x, y, z=0: transformer.transform(x, y), geometry)
        else:
            return None

    def process_road_objects_and_save_image(self, total_road_links):

        for image_path in self.image_paths:
            road_objects = []
            image_name = image_path.split("/")[-1]
            x_min, y_min, x_max, y_max = self.extract_coords(image_name)
            width, height = Image.open(image_path).size

            image = cv2.imread(image_path)
            for geometries, IDs, kinds, types \
                    in tqdm(zip(total_road_links["geometry"], total_road_links["ID"], total_road_links["kind"], total_road_links["type"]), desc="Drawing Road Datas"):
                for geometry, ID, kind_id, type_id in zip(geometries, IDs, kinds, types):
                    try:
                        pixel_coords = self.convert_geometry_to_pixels(geometry, type, x_min, y_min, x_max, y_max, width, height)
                        #
                        clipped_pixel_coords = self.clip_pixels(pixel_coords, width, height)
                        kind_name = ID_name_mapping.KindDict.get(kind_id, 'Unknown Category')
                        type_name = ID_name_mapping.TypeDict.get(type_id, 'Unknown Type')
                        if len(clipped_pixel_coords) > 0:
                            road_obj = RoadObject(id=ID, category_id=kind_id, type_id=type_id, category=kind_name, type=type_name, points=clipped_pixel_coords)
                            road_objects.append(road_obj)

                        image = self.draw_roads(image, clipped_pixel_coords, type)
                    except Exception as e:
                        pass
            s_r = serialize_dataclass(road_objects)
            label_path = os.path.join(config.SaveFolder, "label", image_name.replace(".png", ".json"))
            save_json_with_custom_indent(s_r, label_path)
            save_root = os.path.join(config.SaveFolder, "drawn", image_name)
            cv2.imwrite(save_root, image)
            print(f"save {save_root}")

    def extract_coords(self, image_name):
        parts = image_name.split("_")[1].replace(".png", "").split(",")
        return list(map(float, parts))

    def convert_geometry_to_pixels(self, geom, type, x_min, y_min, x_max, y_max, width, height):
        if type in ["1", "5"]:
            exterior_coords = [self.coords_to_pixels(x, y, x_min, y_max, x_max, y_min, width, height) for x, y in
                               np.array(geom)]
            return exterior_coords
        else:
            lane = [self.coords_to_pixels(x, y, x_min, y_max, x_max, y_min, width, height) for x, y in np.array(geom)]
            return lane

    def clip_pixels(self, pixel_coords, width, height):
        np_pixel_coords = np.array(pixel_coords)
        valid_coords = ((np_pixel_coords[:, 0] < width) & (np_pixel_coords[:, 0] >= 0) &
                        (np_pixel_coords[:, 1] < height) & (np_pixel_coords[:, 1] >= 0))
        if np.any(valid_coords):
            return pixel_coords
        else:
            return []


    def conv(self, geom, x_min, y_min, x_max, y_max, width, height):

        lane = [self.coords_to_pixels(x, y, x_min, y_max, x_max, y_min, width, height) for x, y in np.array(geom.coords)]
        return {"lane": lane}

    def coords_to_pixels(self, x, y, x_min, y_max, x_max, y_min, width, height):
        x_pixel = int(width * (x - x_min) / (x_max - x_min))
        y_pixel = int(height * (y - y_max) / (y_min - y_max))
        return x_pixel, y_pixel

    def draw_roads(self, img, geometries, type):
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

        if "exterior" in geometries:
            exterior_coords = geometries["exterior"]
            if len(exterior_coords) > 1:
                cv2.polylines(img, [np.array(exterior_coords, dtype=np.int32)], True, (255, 255, 0), 1)

        if "interiors" in geometries:
            for interior_coords in geometries["interiors"]:
                if len(interior_coords) > 1:
                    cv2.polylines(img, [np.array(interior_coords, dtype=np.int32)], True, (255, 0, 255), 1)
        return img

    def load_from_json(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def lane_convert_geometry_to_pixels(self, geom, x_min, y_min, x_max, y_max, width, height):
        return [self.coords_to_pixels(x, y, x_min, y_max, x_max, y_min, width, height) for x, y, *_ in np.array(geom.coords)]

    def lane_draw_roads(self, img, pixel_coords):
        for line_pixels in pixel_coords:
            prev_point = None
            if np.all(np.isnan(line_pixels)) or np.all(np.isnan(line_pixels)):
                continue
            for point in line_pixels:
                if prev_point is not None:
                    cv2.line(img, prev_point, point, (0, 255, 255), 1)
                prev_point = point
        return img

    def lane_process(self, total_road_links):
        for image_path in self.image_paths:
            image_name = image_path.split("/")[-1]
            x_min, y_min, x_max, y_max = self.extract_coords(image_name)
            width, height = Image.open(image_path).size

            image = cv2.imread(image_path)
            for geom_collection in tqdm(total_road_links, desc="Drawing Road Datas"):
                for geometry in geom_collection.geometry:
                    try:
                        pixel_coords = self.lane_convert_geometry_to_pixels(geometry, x_min, y_min, x_max, y_max, width, height)
                        image = self.lane_draw_roads(image, pixel_coords)
                    except Exception as e:
                        pass

            save_root = image_path.replace("box", "box_drew")
            cv2.imwrite(save_root, image)
            print(f"save {save_root}")



if __name__ == '__main__':
    road_drawer = RoadObjectsProcessor()
    road_drawer.process()
