import json
import os.path

import numpy as np
import dataclasses as dc
from typing import List, Type
import cv2



def serialize_dataclass(obj):
    if dc.is_dataclass(obj):
        result = {}
        for field in dc.fields(obj):
            value = getattr(obj, field.name)
            if isinstance(value, np.ndarray):
                result[field.name] = {
                    "data": value.tolist(),
                    "dtype": str(value.dtype)
                }
            elif isinstance(value, cv2.KeyPoint):
                result[field.name] = {
                    "pt": value.pt,
                    "size": value.size,
                    "angle": value.angle,
                    "response": value.response,
                    "octave": value.octave,
                    "class_id": value.class_id
                }
            else:
                result[field.name] = serialize_dataclass(value)
        return result
    elif isinstance(obj, list):
        return [serialize_dataclass(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: serialize_dataclass(value) for key, value in obj.items()}
    else:
        return obj


def deserialize_dataclass(data, cls):
    if isinstance(data, dict):
        kwargs = {}
        for field in dc.fields(cls):
            field_data = data[field.name]
            if isinstance(field_data, dict) and "dtype" in field_data:
                kwargs[field.name] = np.array(field_data["data"]).astype(field_data["dtype"])
            elif isinstance(field_data, dict) and "pt" in field_data:
                kwargs[field.name] = cv2.KeyPoint(
                    x=field_data["pt"][0],
                    y=field_data["pt"][1],
                    size=field_data["size"],
                    angle=field_data["angle"],
                    response=field_data["response"],
                    octave=field_data["octave"],
                    class_id=field_data["class_id"]
                )
            else:
                field_cls = field.type if dc.is_dataclass(field.type) else type(field_data)
                kwargs[field.name] = deserialize_dataclass(field_data, field_cls)
        return cls(**kwargs)
    elif isinstance(data, list):
        return [deserialize_dataclass(item, cls.__args__[0]) for item in data]
    else:
        return data


def save_json_with_custom_indent(data, filename, indent=4):
    def _dump(data, level=0):
        if isinstance(data, dict):
            items = []
            for key, value in data.items():
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


def save_map_points(filename: str, map_points: List):
    print(f'[save_map_points] write {len(map_points)} points to {filename}')
    data = serialize_dataclass(map_points)
    save_json_with_custom_indent(data, filename, indent=2)


def load_map_points(filename: str, cls: Type[dc.dataclass]):
    with open(filename, 'r') as f:
        data = json.load(f)
    map_points = deserialize_dataclass(data, List[cls])
    print(f'loaded {len(map_points)} points from {filename}')
    return map_points


# if __name__ == '__main__':
#     from map_build.object_mapping.map_visualizer import MapVisualizer
#     map_points_folder_path = os.path.join(cfg.DATA_PATH, 'point_map')
#     map_points_path = os.path.join(map_points_folder_path, 'map_points.json')
#     map_points = load_map_points(map_points_path, MapPoint)
#     vis = MapVisualizer()
#     pose = Pose(posi=np.array([0,0,0], dtype=np.float32), quat=np.array([0,0,0,1], dtype=np.float32))
#     vis.draw_floor_map(map_points, pose)
#     cv2.waitKey(0)

