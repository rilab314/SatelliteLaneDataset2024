import os
import json
import cv2
import numpy as np
import dataclasses as dc

from glob import glob
from typing import List
from enum import Enum


from src.dto import CLASS_MAPPING


def write_to_json(save_path: str, geometries: List):
    '''
    :param save_path: save path
    :param geometries: list of any dataclass objects
    :return:
    list of dicts로 serialize 하고
    json 파일에 쓰기
    [
        {
            'class': 'GeometryObject',
            'id': '',
            'kind': '',
            'coordinates': [[1, 2], [2, 3], ...]  # 한줄에 나오게
        }
    ]
    '''
    serialized_geometries = serialize_dataclass(geometries)
    save_json_with_custom_indent(serialized_geometries, save_path)

def serialize_dataclass(obj):
    if dc.is_dataclass(obj):
        result = {'class': obj.__class__.__name__}
        for field in dc.fields(obj):
            value = getattr(obj, field.name)
            if isinstance(value, cv2.KeyPoint):
                result[field.name] = serialize_dataclass(value)
            elif isinstance(value, Enum):  # Enum 타입을 문자열로 변환
                result[field.name] = value.name
            else:
                result[field.name] = serialize_dataclass(value)
        return result
    elif isinstance(obj, list):
        return [serialize_dataclass(item) for item in obj]
    else:
        return obj

def save_json_with_custom_indent(data, filename, indent=4):
    def _dump(data, level=0):
        if isinstance(data, dict):
            items = []
            for key, value in data.items():
                if key in {'coordinates', 'image_points', 'global_points'} and isinstance(value, list):
                    items.append(f'{" " * (level * indent)}"{key}": {json.dumps(value, ensure_ascii=False)}')
                else:
                    items.append(f'{" " * (level * indent)}"{key}": {_dump(value, level + 1)}')
            return "{\n" + ",\n".join(items) + f'\n{" " * ((level - 1) * indent)}}}'
        elif isinstance(data, list) and all(isinstance(i, (int, float, str)) for i in data):
            return "[" + ", ".join(json.dumps(i, ensure_ascii=False) for i in data) + "]"
        elif isinstance(data, list):
            items = []
            for value in data:
                items.append(f'{" " * (level * indent)}{_dump(value, level + 1)}')
            return "[\n" + ",\n".join(items) + f'\n{" " * ((level - 1) * indent)}]'
        else:
            return json.dumps(data, ensure_ascii=False)

    json_str = _dump(data)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(json_str)


class JsonFileReader:
    def __init__(self, root_path=None):
        self.root_path = root_path
        self.shape_list = self.list_files()

    def list_files(self) -> List[str]:
        if self.root_path is None:
            return []
        return glob(os.path.join(self.root_path, '*.json'))

    def get_file_list(self):
        return self.shape_list

    def read(self, json_path: str) -> List:
        if not os.path.exists(json_path):
            return []
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        data = []
        for road_obj in json_data:
            class_name = road_obj.pop('class', None)
            data.append(CLASS_MAPPING[class_name](**road_obj))
        return data
