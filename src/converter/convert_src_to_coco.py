import os
import json
import cv2
import shutil
import numpy as np
from glob import glob
from tqdm import tqdm
from typing import Optional

import src.config.config as cfg
import src.config.config_converter as cfg_converter
from src.utils.json_file_io import save_json_with_custom_indent
from src.config.ID_name_mapping import *


class ConvertOriginToCOCO:
    def __init__(self, src_path, save_path, target):
        self.src_path = src_path
        self.save_path = save_path
        self.target = target
        self.categories = self.get_categories(target)
        self.split_json_file = os.path.join(self.src_path, 'dataset.json')
        self.image_shape = (768, 768)
        self.annotation_id_num = 1

    def get_categories(self, target):
        if target == 'detection':
            return cfg_converter.COCO_OD_CATEGORIES
        elif target == 'segmentation':
            categories = {cat_id: info['category'] for cat_id, info in cfg_converter.ADE20K_LANE_CATEGORIES.items()}
            return categories
        raise Exception('Invalid target')

    def train_val_divide_process(self):
        os.makedirs(os.path.join(self.save_path, 'annotations'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'train2017'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'val2017'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'test2017'), exist_ok=True)
        with open(self.split_json_file, 'r') as f:
            split_list = json.load(f)

        for split, file_list in split_list.items():
            image_list = [os.path.join(self.src_path, 'image', coord + '.png') for coord in file_list]
            label_list = [os.path.join(self.src_path, 'label', coord + '.json') for coord in file_list]
            for image_path in tqdm(image_list, desc='copy image files'):
                coco_split = split if split != 'validation' else 'val'
                shutil.copyfile(image_path, image_path.replace(self.src_path, self.save_path).replace('/image/', f'/{coco_split}2017/'))
            self.convert_and_save_annotation(image_list, label_list, split)

    def convert_and_save_annotation(self, image_list, label_list, split):
        coco_format = {'info': {},
                       'licenses': [],
                       'images': [],
                       'annotations': [],
                       'categories': []}

        for json_path in tqdm(label_list, desc='Label contents converting...'):
            origin_data = self.load_json_data(json_path)
            image_id = os.path.basename(json_path).split('.json')[0]
            annotations = []
            for instance_anno in origin_data:
                if instance_anno['class'] == 'MetaData':
                    continue
                if instance_anno['category_id'] in self.categories:
                    annotation_dict = self.create_coco_annotation_from_linestring(instance_anno, image_id, self.image_shape)
                    if annotation_dict is not None:
                        annotations.append(annotation_dict)
            coco_format['annotations'] += annotations

        for image_path in tqdm(image_list, desc='Image contents converting...'):
            image_anno = self.generate_images_coco_format(image_path)
            coco_format['images'].append(image_anno)

        coco_format['info'] = {'contributor': '', 'date_created': '2024/12/13', 'description': '', 'url': '', 'version': '1.0', 'year': 2024}
        coco_format['categories'] = self.generate_categories_coco_format()
        print(f'[annotation prepared] annotation: {len(coco_format["annotations"])}, image: {len(coco_format["images"])}, '
              f'category: {len(coco_format["categories"])}')
        save_path = os.path.join(self.save_path, 'annotations', f'instances_{split}2017.json')
        print(f'saving annotations to {save_path}')
        save_json_with_custom_indent(coco_format, save_path)
        print(f'conversion done for {split}')

    def create_coco_annotation_from_linestring(self, source_data: dict, image_id: int,
                                               image_shape: tuple) -> Optional[dict]:
        if self.target == 'segmentation':
            segmentation, bbox, area = self.geometric_annotation_from_mask(source_data['image_points'], image_shape)
        else:
            segmentation, bbox, area = self.geometric_annotation_from_points(source_data['image_points'], image_shape)
        if bbox is None:
            return None
        src_id = source_data['category_id']
        class_id = list(self.categories.keys()).index(src_id)
        coco_annotation = {
            'id': self.annotation_id_num,  # 어노테이션의 고유 ID (정수)
            'image_id': image_id,  # 이미의 고유 ID (정수)
            'category_id': class_id,  # 카테고리 ID (정수)
            'segmentation': segmentation,  # [[x1, y1, x2, y2, ...]] 형태
            'area': float(area),  # 분할 영역의 면적
            'bbox': bbox,  # [x, y, width, height]
            'iscrowd': 0  # 단일 객체이므로 0
        }
        self.annotation_id_num += 1
        return coco_annotation

    def geometric_annotation_from_mask(self, image_points, image_shape):
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        points = np.array(image_points, dtype=np.int32)
        cv2.polylines(mask, [points], isClosed=False, color=255, thickness=2)
        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        for contour in contours:
            if contour.size >= 6:
                segmentation.append(contour.flatten().tolist())

        if not segmentation:
            print("Warning: 유효한 폴리곤(segmentation)을 생성하지 못했습니다.")
            return None, None, None

        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        bbox = [x, y, w, h]
        area = cv2.contourArea(all_points)
        return segmentation, bbox, area

    def geometric_annotation_from_points(self, image_points, image_shape):
        image_points = np.array(image_points)
        x, y, w, h = cv2.boundingRect(image_points)
        bbox = [x, y, w, h]
        area = cv2.contourArea(image_points)
        return [], bbox, area

    def generate_images_coco_format(self, image_path):
        image_dict = {'license': 1,
                      'file_name': image_path.split('/')[-1],
                      'coco_url': '',
                      'height': self.image_shape[0],
                      'width': self.image_shape[1],
                      'date_captured': '',
                      'flickr_url': '',
                      'id': image_path.split('/')[-1].split('.png')[0]}
        return image_dict

    def generate_categories_coco_format(self):
        categories = []
        for categ_id, (kind_id, kind_name) in enumerate(self.categories.items()):
            categories.append({'id': categ_id, 'name': kind_name, 'supercategory': self.target})
        return categories

    def load_json_data(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data


if __name__ == '__main__':
    src_path = cfg.DATASET_PATH
    save_path = cfg.CUSTOM_COCO_PATH
    target = ['detection', 'segmentation'][1]
    converter = ConvertOriginToCOCO(src_path, save_path, target)
    converter.train_val_divide_process()
