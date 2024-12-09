import os
import json
import cv2
import shutil
import numpy as np
from glob import glob
from tqdm import tqdm

import src.config.config as cfg
import src.converter.utils.generate_train_val_test_coords as gen_train_val_test_coords
from src.utils.json_file_io import save_json_with_custom_indent
from src.config.ID_name_mapping import *



class ConvertOriginToCOCO:
    def __init__(self, path, save_path='', divide_json=''):
        self.origin_path = path
        self.save_path = save_path
        self.divide_json = divide_json
        self.origin_image_list = sorted(glob(os.path.join(self.origin_path, 'image', '*.png')))
        self.origin_label_list = sorted(glob(os.path.join(self.origin_path, 'label', '*.json')))
        self.annotation_id_num = 1

    def train_val_divide_process(self):
        os.makedirs(os.path.join(self.save_path, 'annotations'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'train2017'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'val2017'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'test2017'), exist_ok=True)
        gen_train_val_test_coords.generate_train_val_test_coords(os.path.join(self.origin_path, 'label'), self.divide_json, cfg.DATASET_RATIO, [cfg.SEOUL_CONFIG, cfg.INCHEON_CONFIG])
        with open(self.divide_json, 'r') as f:
            coords = json.load(f)
        train_image_list = [os.path.join(self.origin_path, 'image', coord+'.png') for coord in coords['train']]
        train_label_list = [os.path.join(self.origin_path, 'label', coord+'.json') for coord in coords['train']]
        val_image_list = [os.path.join(self.origin_path, 'image', coord+'.png') for coord in coords['validation']]
        val_label_list = [os.path.join(self.origin_path, 'label', coord+'.json') for coord in coords['validation']]
        test_image_list = [os.path.join(self.origin_path, 'image', coord+'.png') for coord in coords['test']]

        self.convert_annotation(train_image_list, train_label_list, 'instances_train2017.json')
        self.convert_annotation(val_image_list, val_label_list, 'instances_val2017.json')
        self.copy_divide_image(train_image_list, val_image_list, test_image_list)

    def convert_annotation(self, image_list, label_list, save_filename):
        coco_format = {'info': {},
                       'licenses': [],
                       'images': [],
                       'annotations': [],
                       'categories': []}

        for path in tqdm(label_list, desc='Label contents converting...'):
            origin_data = self.load_json_data(path)
            origin2coco_annotation = self.generate_annotation_coco_format(origin_data, os.path.basename(path).split('.json')[0])
            coco_format['annotations'] += origin2coco_annotation

        for image_path in tqdm(image_list, desc='Image contents converting...'):
            origin2coco_images = self.generate_images_coco_format(image_path)
            coco_format['images'].append(origin2coco_images)

        coco_format['info'] = {'contributor': '', 'date_created': '2024/11/14', 'description': '', 'url': '', 'version': '1.0', 'year': 2024}
        coco_format['categories'] = self.generate_categories_coco_format()
        save_path = os.path.join(self.save_path, 'annotations', save_filename)
        save_json_with_custom_indent(coco_format, save_path)

    def generate_annotation_coco_format(self, origin_data, image_id):
        annotations = []
        safety_zone_objects = []
        for data in origin_data:
            if data['class'] == 'MetaData':
                continue
            #
            if data['category_id'] in ['530'] or data['type_id'] in ['1', '5']:
                annotation_dict = self.gen_annotation_dict(data, image_id)
                annotations.append(annotation_dict)
            elif data['category_id'] == '531': # safety_zone
                if 'bbox' not in data:
                    data['image_points'] = self.filter_array(data['image_points'])
                    if not data['image_points']:
                        continue
                    data['bbox'] = self.get_bbox_of_polygon(data['image_points'], [cfg.IMAGE_SIZE_h, cfg.IMAGE_SIZE_w])
                safety_zone_objects.append(data)

        merged_safety_zones = self.merge_safety_zone_objects(safety_zone_objects, threshold=15)
        # merged_safety_zones = self.merge_safety_zone_objects_by_id(safety_zone_objects, threshold=20)

        for merged_object in merged_safety_zones:
            annotation_dict = self.gen_annotation_dict(merged_object, image_id)
            annotations.append(annotation_dict)
            #
        return annotations

    def gen_annotation_dict(self, data, image_id):
        annotation_dict = {'segmentation': [],
                           'area': float,
                           'iscrowd': int,
                           'image_id': int,
                           'bbox': [],
                           'category_id': int,
                           'id': int,
                           'type_id': int,
                           'road_id': str}
        area = self.get_area_of_polygon(data['image_points'])
        bbox = self.get_bbox_of_polygon(data['image_points'], [cfg.IMAGE_SIZE_h, cfg.IMAGE_SIZE_w])

        annotation_dict['segmentation'] = data['image_points']
        annotation_dict['area'] = area
        annotation_dict['iscrowd'] = 0
        annotation_dict['image_id'] = image_id
        annotation_dict['bbox'] = bbox
        annotation_dict['category_id'] = int(data['category_id'])
        # annotation_dict['id'] = data['id']
        annotation_dict['id'] = self.annotation_id_num
        self.annotation_id_num += 1
        annotation_dict['road_id'] = data['id']
        annotation_dict['type_id'] = int(data['type_id'])

        return annotation_dict

    def get_area_of_polygon(self, polygon):
        points = np.array(polygon)
        x = points[:, 0]
        y = points[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area

    def get_bbox_of_polygon(self, points, image_size):
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        x_min = np.min(points[:, 0])
        y_min = np.min(points[:, 1])
        x_max = np.max(points[:, 0])
        y_max = np.max(points[:, 1])

        img_height, img_width = image_size

        x_min = int(max(0, min(x_min, img_width - 1)))
        y_min = int(max(0, min(y_min, img_height - 1)))
        x_max = int(max(0, min(x_max, img_width - 1)))
        y_max = int(max(0, min(y_max, img_height - 1)))

        w = x_max - x_min
        h = y_max - y_min

        return [x_min, y_min, w, h]

    def filter_array(self, data, height=cfg.IMAGE_SIZE_h, width=cfg.IMAGE_SIZE_w):
        array = np.array(data)
        mask = (array[:, 0] <= height) & (array[:, 1] <= width) & (array >= 0).all(axis=1)
        return array[mask].tolist()

    def merge_safety_zone_objects_by_id(self, objects, threshold: int = 20):
        """
        Merge objects based on their `id` values. Objects are merged if their `id` values are within the threshold.

        Args:
            objects (list): List of objects containing `id`, `bbox`, and `image_points`.
            threshold (int): Maximum difference in `id` values to allow merging.

        Returns:
            list: List of merged objects.
        """
        # Sort objects by 'id'
        objects.sort(key=lambda obj: int(obj['id'][6:]))

        merged_objects = []
        current_merged_object = None

        for obj in objects:
            obj_id = int(obj['id'][6:])

            if current_merged_object is None:
                # Initialize the first object to be merged
                current_merged_object = obj
            else:
                # Calculate the difference in `id` values
                current_id = int(current_merged_object['id'][6:])
                if abs(obj_id - current_id) <= threshold:
                    # Merge the current object with the new object
                    current_merged_object['bbox'] = [
                        min(current_merged_object['bbox'][0], obj['bbox'][0]),
                        min(current_merged_object['bbox'][1], obj['bbox'][1]),
                        max(current_merged_object['bbox'][0] + current_merged_object['bbox'][2],
                            obj['bbox'][0] + obj['bbox'][2]) - min(current_merged_object['bbox'][0], obj['bbox'][0]),
                        max(current_merged_object['bbox'][1] + current_merged_object['bbox'][3],
                            obj['bbox'][1] + obj['bbox'][3]) - min(current_merged_object['bbox'][1], obj['bbox'][1])
                    ]
                    current_merged_object['image_points'] += obj['image_points']
                else:
                    # Append the current merged object and start a new one
                    merged_objects.append(current_merged_object)
                    current_merged_object = obj

        # Append the last merged object
        if current_merged_object:
            merged_objects.append(current_merged_object)
        if merged_objects:
            print(merged_objects[0]['image_id'])
            self.visualize_bboxes(objects, merged_objects, merged_objects[0]['image_id'])

        return merged_objects

    def merge_safety_zone_objects(self, objects, threshold: int = 10):
        merged_objects = []
        used = [False] * len(objects)

        def is_bbox_close(bbox1, bbox2, threshold):
            x1_min, y1_min, w1, h1 = bbox1
            x2_min, y2_min, w2, h2 = bbox2
            x1_max, y1_max = x1_min + w1, y1_min + h1
            x2_max, y2_max = x2_min + w2, y2_min + h2

            return (
                    x1_min - threshold <= x2_max and x1_max + threshold >= x2_min and
                    y1_min - threshold <= y2_max and y1_max + threshold >= y2_min
            )

        for i, obj1 in enumerate(objects):
            if used[i]:
                continue
            merged_object = obj1
            merged_bbox = obj1['bbox']
            merged_points = obj1['image_points']
            used[i] = True

            for j, obj2 in enumerate(objects):
                if i != j and not used[j] and is_bbox_close(merged_bbox, obj2['bbox'], threshold):
                    used[j] = True
                    merged_bbox = [
                        min(merged_bbox[0], obj2['bbox'][0]),
                        min(merged_bbox[1], obj2['bbox'][1]),
                        max(merged_bbox[0] + merged_bbox[2], obj2['bbox'][0] + obj2['bbox'][2]) - min(merged_bbox[0],
                                                                                                      obj2['bbox'][0]),
                        max(merged_bbox[1] + merged_bbox[3], obj2['bbox'][1] + obj2['bbox'][3]) - min(merged_bbox[1],
                                                                                                      obj2['bbox'][1])
                    ]
                    merged_points += obj2['image_points']
            merged_object['image_points'] = merged_points
            merged_object['bbox'] = merged_bbox
            merged_objects.append(merged_object)
        # if merged_objects:
            # print(merged_objects[0]['image_id'])
            # self.visualize_bboxes(objects, merged_objects, merged_objects[0]['image_id'])
        return merged_objects

    def visualize_bboxes(self, objects, merged_objects, image_id='', image_size=(768, 768)):
        """
        Visualize original and merged bounding boxes.

        Args:
            objects (list): List of original objects with 'bbox'.
            merged_objects (list): List of merged objects with 'bbox'.
            image_size (tuple): Size of the visualization canvas (height, width).
        """
        # Create a blank canvas
        original_image = cv2.imread(os.path.join('/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/satellite_good_matching_241125/image', image_id.replace('.json', '.png')))

        original_canvas = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        merged_canvas = original_image.copy()
        combined_canvas = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

        # Random colors for visualizing individual boxes
        def random_color():
            return tuple(random.randint(0, 255) for _ in range(3))

        # Draw original bounding boxes in red
        for obj in objects:
            bbox = obj['bbox']
            x_min, y_min, w, h = bbox
            x_max, y_max = x_min + w, y_min + h
            cv2.rectangle(original_canvas, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        # Draw merged bounding boxes in green
        for obj in merged_objects:
            bbox = obj['bbox']
            x_min, y_min, w, h = bbox
            x_max, y_max = x_min + w, y_min + h
            cv2.rectangle(merged_canvas, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Combine original and merged visualizations
        combined_canvas = cv2.addWeighted(original_canvas, 0.5, merged_canvas, 0.5, 0)

        # Display the results
        cv2.imshow("Original Bounding Boxes (Red)", original_canvas)
        cv2.imshow("Merged Bounding Boxes (Green)", merged_canvas)
        cv2.imshow("Combined View", combined_canvas)
        cv2.imshow('image', original_image)
        cv2.waitKey(0)
    def generate_images_coco_format(self, image_path):
        image_dict = {'license': 1,
                      'file_name': image_path.split('/')[-1],
                      'coco_url': '',
                      'height': int,
                      'width': int,
                      'date_captured': str,
                      'flickr_url': str,
                      'id': int}

        image = cv2.imread(image_path)
        image_dict['file_name'] = image_path.split('/')[-1]
        image_dict['width'] = image.shape[0]
        image_dict['height'] = image.shape[1]
        image_dict['date_captured'] = ''
        image_dict['flickr_url'] = ''
        image_dict['id'] = image_path.split('/')[-1].split('.png')[0]
        return image_dict

    def generate_categories_coco_format(self):
        categories = []
        for kind_id, kind_name in KindDict.items():
            type_key = next((key for key, value in TypeDict.items() if value == kind_name), None)
            if type_key:
                supercategory = TypeDict_English[TypeDict[type_key]]
            else:
                supercategory = 'unknown'

            categories.append({
                'id': int(kind_id),
                'name': KindDict_English[kind_name],
                'supercategory': supercategory
            })
        return categories

    def load_json_data(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def copy_divide_image(self, train_image_list, val_image_list, test_image_list):
        for image_path in tqdm(train_image_list, desc='Copy training dataset'):
            moved_image_path = os.path.join(self.save_path, 'train2017', os.path.basename(image_path))
            shutil.copy(image_path, moved_image_path)
        for image_path in tqdm(val_image_list, desc='Copy validation dataset'):
            moved_image_path = os.path.join(self.save_path, 'val2017', os.path.basename(image_path))
            shutil.copy(image_path, moved_image_path)
        for image_path in tqdm(test_image_list, desc='Copy test dataset'):
            moved_image_path = os.path.join(self.save_path, 'test2017', os.path.basename(image_path))
            shutil.copy(image_path, moved_image_path)

if __name__ == '__main__':
    path = '/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/satellite_good_matching_241125'
    save_path = '/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/satellite_coco_241209'
    divide_json = path+'/dataset.json'
    converter = ConvertOriginToCOCO(path, save_path, divide_json)
    converter.train_val_divide_process()
