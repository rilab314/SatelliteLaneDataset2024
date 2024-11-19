import os
import json
import cv2
import shutil
import numpy as np
from glob import glob
from tqdm import tqdm

from matcher.config.config import TotalDataset
from matcher.file_io import serialize_dataclass, deserialize_dataclass, save_json_with_custom_indent
from matcher.config.ID_name_mapping import *


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
        with open(self.divide_json, 'r') as f:
            coords = json.load(f)
        train_image_list = [os.path.join(self.origin_path, 'image', coord+'.png') for coord in coords['train']]
        train_label_list = [os.path.join(self.origin_path, 'label', coord+'.json') for coord in coords['train']]
        val_image_list = [os.path.join(self.origin_path, 'image', coord+'.png') for coord in coords['validation']]
        val_label_list = [os.path.join(self.origin_path, 'label', coord+'.json') for coord in coords['validation']]
        self.convert_annotation(train_image_list, train_label_list, 'instances_train2017.json')
        self.convert_annotation(val_image_list, val_label_list, 'instances_val2017.json')
        self.copy_divide_image(train_image_list, val_image_list)


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
        # image_id = int(image_id)
        for data in origin_data:
            if data['class'] == 'MetaData':
                continue
            #
            if data['category_id'] in ['530'] or data['type_id'] in ['1', '5']:
            #
                annotation_dict = self.gen_annotation_dict(data, image_id)
                annotations.append(annotation_dict)

        return annotations

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
        bbox = self.get_bbox_of_polygon(data['image_points'], [768, 768, 3])

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

        w = x_max - x_min
        h = y_max - y_min

        return [x_min, y_min, w, h]

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

    def load_json_data(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def copy_divide_image(self, train_image_list, val_image_list):
        for image_path in tqdm(train_image_list, desc='Copy training dataset'):
            moved_image_path = os.path.join(self.save_path, 'train2017', os.path.basename(image_path))
            shutil.copy(image_path, moved_image_path)

        for image_path in tqdm(val_image_list, desc='Copy validation dataset'):
            moved_image_path = os.path.join(self.save_path, 'val2017', os.path.basename(image_path))
            shutil.copy(image_path, moved_image_path)
            moved_image_path = os.path.join(self.save_path, 'test2017', os.path.basename(image_path))
            shutil.copy(image_path, moved_image_path)

if __name__ == '__main__':
    path = '/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/satellite_good_matching_241119'
    save_path = '/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/satellite_coco_241119'
    divide_json = path+'/dataset.json'
    converter = ConvertOriginToCOCO(path, save_path, divide_json)
    converter.train_val_divide_process()
