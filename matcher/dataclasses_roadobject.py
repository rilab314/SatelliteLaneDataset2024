import json
import os

import numpy as np
import dataclasses as dc
from typing import List, Type
import cv2

@dc.dataclass(unsafe_hash=True)
class RoadObject:
	id: int
	category: str
	type: str
	points: List
