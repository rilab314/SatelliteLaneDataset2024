import dataclasses as dc
from typing import List, Tuple
import cv2
from enum import Enum


class GeometryType(Enum):
    POLYGON = 'POLYGON',
    LINE_STRING = 'LINE_STRING',


# TODO 필드마다 설명 입력
@dc.dataclass(unsafe_hash=True)
class GeometryObject:
    id: str  # unique id for each geometric object
    kind: str  # ??
    type: str  # ??
    src_file: str  # source file path from root_path
    coordinates: List[Tuple[float,float]]  # coordinates
    geometry_type: GeometryType  # polygon or linestring


@dc.dataclass(unsafe_hash=True)
class RoadObject:
    id: str  # ??
    category_id: str  # ??
    type_id: str
    category: str
    type: str
    geometry_type: GeometryType
    image_points: List[Tuple[int,int]]
    global_points: List[Tuple[float,float]]
    image_id: int


@dc.dataclass(unsafe_hash=True)
class MetaData:
    type: str
    image_x1y1x2y2: List
    coordinate_format: str
    format_code: str
    region: str
