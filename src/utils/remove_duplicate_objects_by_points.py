import json
import os
from tqdm import tqdm

from src.utils.json_file_io import JsonFileReader, write_to_json

def remove_duplicate_objects(json_folder_path):
    for file_name in tqdm(os.listdir(json_folder_path), desc="Removing duplicate objects"):
        if file_name.endswith(".json"):
            file_path = os.path.join(json_folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            unique_objects = []
            seen_image_points = set()

            for obj in data:
                if obj["class"] == "RoadObject":
                    points_tuple = tuple(tuple(p) for p in obj["image_points"])
                    if points_tuple not in seen_image_points:
                        unique_objects.append(obj)
                        seen_image_points.add(points_tuple)
                else:
                    unique_objects.append(obj)
            write_to_json(file_path, unique_objects)

if __name__ == '__main__':
    label_path = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2024_SATELLITE/datasets/satellite_good_matching_241122/label'
    remove_duplicate_objects(label_path)
