import json
import os

import numpy as np
import dataclasses as dc
from typing import List, Type
import cv2

@dc.dataclass(unsafe_hash=True)
class RoadObject:
	id: str
	category_id: str
	type_id: str
	category: str
	type: str
	pixel_points: List
	web_mercator_points: List
	image_id: int

@dc.dataclass(unsafe_hash=True)
class RoadMetaData:
	type: str
	image_x1y1x2y2: List
	coordinate_format: str
	region: str



