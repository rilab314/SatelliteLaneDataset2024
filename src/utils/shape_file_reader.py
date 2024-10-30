import os
import numpy as np
import geopandas as gpd
from typing import List
from pyproj import Transformer
from shapely.ops import transform
from shapely.geometry import LineString, Polygon

import src.config.config as cfg
from src.dto import GeometryObject, GeometryType


class ShapeFileReader:
    def __init__(self, root_path):
        self.root_path = root_path
        self.shape_list = self.list_files()
        self.UTM_to_webmercator_transformer = Transformer.from_crs('EPSG:32652', 'EPSG:3857', always_xy=True)

    def list_files(self) -> List[str]:
        surface_links_paths = sorted(self.find_files(self.root_path, cfg.SURFACE_SHAPE_endswith_NAME))
        lane_links_paths = sorted(self.find_files(self.root_path, cfg.LANE_SHAPE_endswith_NAME))
        return surface_links_paths + lane_links_paths

    def find_files(self, root_folder:str, file_name_to_find:str) -> List[str]:
        found_files = []
        for subdir, dirs, files in os.walk(root_folder):
            for file in files:
                if os.path.join(subdir, file).endswith(file_name_to_find):
                    found_files.append(os.path.join(subdir, file))
        return found_files

    def get_file_list(self):
        return self.shape_list

    def read(self, shape_path: str) -> List[GeometryObject]:
        geo_df = gpd.read_file(shape_path)
        coordinates = self.transform_geometry(geo_df['geometry'], self.UTM_to_webmercator_transformer)
        geometries = []
        for coord, geo_id, geo_kind, geo_type in zip(coordinates, geo_df['ID'], geo_df['kind'], geo_df['type']):
            geom = self.convert_to_geometry_object(coord, geo_id, geo_kind, geo_type, shape_path)
            geometries.append(geom)
        return geometries

    def convert_to_geometry_object(self, coord, geo_id, geo_kind, geo_type, shape_path) -> GeometryObject:
        if coord.__class__.__name__ == 'LineString':
            geometry_type = GeometryType.LINE_STRING
        elif coord.__class__.__name__ == 'Polygon':
            geometry_type = GeometryType.POLYGON
        return GeometryObject(id=geo_id,
                              kind=geo_kind,
                              type=geo_type,
                              src_file=shape_path.split(cfg.SHAPE_PATH)[1],
                              coordinates=self.serialize_geometry(coord),
                              geometry_type=geometry_type)

    def transform_geometry(self, geometry, transformer):
        if geometry is None:
            return None
        def transform_coords(coords):
            coords_np = np.array(coords)
            transformed = transformer.transform(coords_np[:, 0], coords_np[:, 1])
            return np.column_stack(transformed)

        if geometry.geom_type == "LineString":
            return LineString(transform_coords(geometry.coords))

        elif geometry.geom_type == "Polygon":
            transformed_exterior = LineString(transform_coords(geometry.exterior.coords))
            return Polygon(transformed_exterior)

        else:
            return transform(lambda x, y, z=0: transformer.transform(x, y), geometry)

    def serialize_geometry(self, coord):
        if isinstance(coord, Polygon):
            exterior = [(x, y) for x, y in coord.exterior.coords]
            return exterior
        elif isinstance(coord, LineString):
            return [(x, y) for x, y in coord.coords]
        return None