import json
import os
from glob import glob
from tqdm import tqdm

import matcher.file_io as file_io
import matcher.config.ID_name_mapping as ID_name_mapping
from matcher.dataclasses_roadobject import RoadObject
from matcher.file_io import save_json_with_custom_indent

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def add_image_id(label_path):
    data = load_json(label_path)
    image_id = int(label_path.split("/")[-1].split("_")[0])
    if data[0]["type"] == "metadata":
        data[0]["image_id"] = image_id
    else:
        print(f"Error label: {label_path}")

    save_json_with_custom_indent(data, label_path)

if __name__ == '__main__':
    label_folder = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/국토정보플랫폼/dataset/label"
    label_list = glob(os.path.join(label_folder, "*.json"))
    for label_path in tqdm(label_list, desc="Add Image ID..."):
        add_image_id(label_path)

