import dataclasses as dc
from typing import List, Tuple
import cv2
from enum import Enum



class GeometryType(Enum):
    POLYGON = 'POLYGON',
    LINE_STRING = 'LINE_STRING',
    MULTIPOLYGON = 'MULTIPOLYGON',
    MULTILINE_STRING = 'MULTILINE_STRING',


# TODO 필드마다 설명 입력
@dc.dataclass(unsafe_hash=True)
class GeometryObject:
    id: str  # unique id for each geometric object
    kind: str  # type or category of the object (e.g., stop_line, u_turn)
    type: str  # specific subtype of the object (e.g., dashed line, arrow)
    src_file: str  # source file path from root_path
    coordinates: List[Tuple[float,float]]  # coordinates
    geometry_type: GeometryType  # polygon or linestring


@dc.dataclass(unsafe_hash=True)
class RoadObject:
    id: str  # unique id for each geometric object
    category_id: str  # kind id
    category: str  # kind name
    type_id: str  # type id
    type: str  # type name
    geometry_type: GeometryType  # Type of geometry (e.g., POLYGON or LINE_STRING)
    image_points: List[Tuple[int,int]]  # List of (x, y) points in image coordinates
    global_points: List[Tuple[float,float]]  # List of (longitude, latitude) points in global coordinates


@dc.dataclass(unsafe_hash=True)
class MetaData:
    image_x1y1x2y2: List  # Coordinates representing the outermost edges of the image [x1, y1, x2, y2]
    coordinate_format: str  # Format of the coordinates (e.g., web mercator)
    format_code: str  # Code representing the specific coordinate format (e.g., EPSG code)
    region: str  # Geographical region represented by this metadata (e.g., Seoul, Korea)

CLASS_MAPPING = {
    "GeometryObject": GeometryObject,
    "RoadObject": RoadObject,
    "MetaData": MetaData,
}




if __name__ == '__main__':
    a = CLASS_MAPPING["GeometryObject"](id="0", kind="0", type="0", src_file="0", coordinates="0", geometry_type="0")
    b = GeometryObject(id="0", kind="0", type="0", src_file="0", coordinates="0", geometry_type="0")

    path = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/satellite_coordinate_241015/raw_label"
    from glob import glob
    import os
    p = glob(os.path.join(path, "*14118026.832884334,4510805.9017791115,14118273.3263203,4511053.596438353.json"))
    a =  {
            'class': 'GeometryObject',
            'id': 'qwr',
            'kind': 'ad',
            'type': 'd',
            'src_file': 'v',
            'coordinates': 'd',
        'geometry_type': 'q',
        }
    class_name = a.pop('class', None)
    asd = CLASS_MAPPING[class_name](**a)
    cvx = GeometryType.POLYGON
    from shapely.geometry import LineString, Polygon
    import numpy as np

    vcxv = glob(os.path.join(path, f"*00004_14118026.832884334,4509733.767318699,14118273.3263203,4509981.436626772.json"))
    poly = Polygon()
    zxc = GeometryType(poly.__class__)
    p=1

