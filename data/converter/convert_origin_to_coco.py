import os
import json
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

from matcher.config.config import TotalDataset
from matcher.file_io import serialize_dataclass, deserialize_dataclass, save_json_with_custom_indent


class ConvertOriginToCOCO:
    def __init__(self):
        self.origin_path = TotalDataset
        self.origin_image_list = glob(os.path.join(self.origin_path, "image", "*.png"))
        self.origin_label_list = glob(os.path.join(self.origin_path, "label", "*.json"))

    def convert_process(self):
        coco_format = {"info": {},
                       "licenses": [],
                       "images": [],
                       "annotations": [],
                       "categories": []}

        for path in tqdm(self.origin_label_list, desc="Label contents converting..."):
            origin_data = self.load_json_data(path)
            origin2coco_annotation = self.generate_annotation_coco_format(origin_data)
            coco_format["annotations"] += origin2coco_annotation

        for image_path in tqdm(self.origin_image_list, desc="Image contents converting..."):
            origin2coco_images = self.generate_images_coco_format(image_path)
            coco_format["images"].append(origin2coco_images)


        save_path = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/국토정보플랫폼/dataset/coco_format.json"
        save_json_with_custom_indent(coco_format, save_path)

    def generate_annotation_coco_format(self, origin_data):
        annotations = []
        image_id = None
        for data in origin_data:
            if data["type"] == "metadata":
                image_id = data["image_id"]
                continue
            annotation_dict = self.gen_annotation_dict(data, image_id)
            annotations.append(annotation_dict)

        return annotations


    def gen_annotation_dict(self, data, image_id):
        annotation_dict = {"segmentation": [],
                           "area": float,
                           "iscrowd": int,
                           "image_id": int,
                           "bbox": [],
                           "category_id": int,
                           "id": str,
                           "type_id": int}
        area = self.get_area_of_polygon(data["pixel_points"])
        bbox = self.get_bbox_of_polygon(data["pixel_points"], [768, 768, 3])

        annotation_dict["segmentation"] = data["pixel_points"]
        annotation_dict["area"] = area
        annotation_dict["iscrowd"] = 0
        annotation_dict["image_id"] = image_id
        annotation_dict["bbox"] = bbox
        annotation_dict["category_id"] = int(data["category_id"])
        annotation_dict["id"] = data["id"]
        annotation_dict["type_id"] = int(data["type_id"])

        return annotation_dict

    def get_area_of_polygon(self, polygon):
        points = np.array(polygon)
        x = points[:, 0]
        y = points[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area

    def get_bbox_of_polygon(self, points, image_shape):
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        x_min = np.min(points[:, 0])
        y_min = np.min(points[:, 1])
        x_max = np.max(points[:, 0])
        y_max = np.max(points[:, 1])

        img_height, img_width = image_shape[:2]

        x_min = int(max(0, min(x_min, img_width - 1)))
        y_min = int(max(0, min(y_min, img_height - 1)))
        x_max = int(max(0, min(x_max, img_width - 1)))
        y_max = int(max(0, min(y_max, img_height - 1)))
        return [x_min, y_min, x_max, y_max]

    def generate_images_coco_format(self, image_path):
        image_dict = {"license": 1,
                      "file_name": image_path.split("/")[-1],
                      "coco_url": "",
                      "height": int,
                      "width": int,
                      "date_captured": str,
                      "flickr_url": str,
                      "id": int}

        image = cv2.imread(image_path)
        image_dict["file_name"] = image_path.split("/")[-1]
        image_dict["width"] = image.shape[0]
        image_dict["height"] = image.shape[1]
        image_dict["date_captured"] = ""
        image_dict["flickr_url"] = ""
        image_dict["id"] = int(image_path.split("/")[-1].split("_")[0])
        return image_dict

    def load_json_data(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        return data

if __name__ == '__main__':
    converter = ConvertOriginToCOCO()
    converter.convert_process()
