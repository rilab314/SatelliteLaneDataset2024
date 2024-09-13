import numpy as np
import cv2
import math
import geopandas as gpd
import pandas as pd
import os
from glob import glob
from PIL import Image
from pyproj import Transformer
from shapely.ops import transform
from shapely.geometry import LineString, Polygon
from tqdm import tqdm

from config import config


class RoadObjectsProcessor:
    def __init__(self):
        self.image_paths = glob(config.ImagesFolder + "/*.png")
        self.image_paths.sort()
        self.lonlat_to_web_transformer = Transformer.from_crs("EPSG:32652", "EPSG:3857", always_xy=True)

    def process(self):
        lane_shp_files = self.find_files(config.root_folder, config.LaneShapeFile)
        total_lane_links = self.shape_load_and_transform(lane_shp_files)
        self.process_road_objects_and_save_image(total_lane_links)

        surface_shp_files = self.find_files(config.root_folder, config.SurfaceMarkShapeFile)
        total_surface_links = self.shape_load_and_transform(surface_shp_files)
        self.process_road_objects_and_save_image(total_surface_links)

    def find_files(self, root_folder, file_pattern):
        found_files = []
        for subdir, dirs, files in os.walk(root_folder):
            for file in files:
                if os.path.join(subdir, file).endswith(file_pattern):
                    found_files.append(os.path.join(subdir, file))
        return found_files

    def shape_load_and_transform(self, files):
        total_road_links = []
        for r_l_path in tqdm(files, desc="Transforming Road Datas"):
            try:
                road_links = gpd.read_file(r_l_path)
                road_links["transformed_geometry"] = road_links["geometry"].apply(self.transform_geometry)
                if len(road_links["transformed_geometry"]) > 1:
                    total_road_links.append(road_links["transformed_geometry"])
            except Exception as e:
                print(e)
        return total_road_links

    def transform_geometry(self, geometry):
        if geometry is not None:
            if geometry.geom_type == "LineString":
                return LineString([self.lonlat_to_web_transformer.transform(x, y) for x, y, *_ in geometry.coords])
            elif geometry.geom_type == "Polygon":
                exterior = geometry.exterior
                transformed_exterior = LineString([self.lonlat_to_web_transformer.transform(x, y) for x, y, *_ in exterior.coords])

                interiors = []
                for interior in geometry.interiors:
                    transformed_interior = LineString([self.lonlat_to_web_transformer.transform(x, y) for x, y, *_ in interior.coords])
                    interiors.append(transformed_interior)

                return Polygon(transformed_exterior, interiors)
            else:
                return transform(lambda x, y, z=0: self.lonlat_to_web_transformer.transform(x, y), geometry)
        else:
            return None

    def process_road_objects_and_save_image(self, total_road_links):
        for image_path in self.image_paths:
            image_name = image_path.split("/")[-1]
            x_min, y_min, x_max, y_max = self.extract_coords(image_name)
            width, height = Image.open(image_path).size

            image = cv2.imread(image_path)
            for geom_collection in tqdm(total_road_links, desc="Drawing Road Datas"):
                for geometry in geom_collection.geometry:
                    try:
                        pixel_coords = self.convert_geometry_to_pixels(geometry, x_min, y_min, x_max, y_max, width, height)
                        image = self.draw_roads(image, pixel_coords)
                    except Exception as e:
                        pass

            save_root = image_path.replace("box", "box_drew")
            cv2.imwrite(save_root, image)
            print(f"save {save_root}")

    def extract_coords(self, image_name):
        parts = image_name.split("_")[1].replace(".png", "").split(",")
        return list(map(float, parts))

    def convert_geometry_to_pixels(self, geom, x_min, y_min, x_max, y_max, width, height):
        if isinstance(geom, Polygon):
            exterior_coords = [self.coords_to_pixels(x, y, x_min, y_max, x_max, y_min, width, height) for x, y in
                               np.array(geom.exterior.coords)]
            interiors = [[self.coords_to_pixels(x, y, x_min, y_max, x_max, y_min, width, height) for x, y in
                          np.array(interior.coords)] for interior in geom.interiors]
            return {"exterior": exterior_coords, "interiors": interiors}
        elif isinstance(geom, LineString):
            lane = [self.coords_to_pixels(x, y, x_min, y_max, x_max, y_min, width, height) for x, y in np.array(geom.coords)]
            return {"lane": lane}
        return {}

    def coords_to_pixels(self, x, y, x_min, y_max, x_max, y_min, width, height):
        x_pixel = int(width * (x - x_min) / (x_max - x_min))
        y_pixel = int(height * (y - y_max) / (y_min - y_max))
        return x_pixel, y_pixel

    def draw_roads(self, img, geometries):
        if "lane" in geometries:
            pixel_coords = geometries["lane"]
            for line_pixels in pixel_coords:
                prev_point = None
                if np.all(np.isnan(line_pixels)) or np.all(np.isnan(line_pixels)):
                    continue
                for point in line_pixels:
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





