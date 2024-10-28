from typing import List

from src.dto import GeometryObject


class ShapeFileReader:
    def __init__(self, root_path):
        self.root_path = root_path
        self.shape_list = self.list_files(root_path)

    def list_files(self, root_path) -> List[str]:
        pass

    def get_file_list(self):
        return self.shape_list

    def read(self, shape_path: str) -> List[GeometryObject]:
        geo_df = gpd.read_file(shape_path)
        geometries = []
        for record in geo_df.xxx():
            geom = self.convert_to_geometry_object(record, shape_path)
            geometries.append(geom)
        return geometries

    def convert_to_geometry_object(self, record, shape_path) -> GeometryObject:
        pass
