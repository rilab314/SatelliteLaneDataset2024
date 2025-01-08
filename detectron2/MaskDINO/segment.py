# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import os
from shapely.errors import ShapelyDeprecationWarning
import warnings
warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import torch

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine.defaults import DefaultPredictor

import cv2
import numpy as np

import sys
from maskdino import add_maskdino_config


class SegmentNet():
    def __init__(self, config_file, checkpoint_path, category):
        self.predictor = self.load_model(config_file, checkpoint_path)
        self.device = torch.device("cpu")
        self.category = category

    def load_model(self, config_file, checkpoint_path):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskdino_config(cfg)
        if config_file is None:
            raise FileNotFoundError('SegmentNet config file not found')
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(['MODEL.WEIGHTS', checkpoint_path])
        cfg.freeze()
        return DefaultPredictor(cfg)

    # resize : 이미지 리사이즈, gray_only : 컬러 라벨 생성 여부 결정
    def segment(self, image, resize=None, gray_only=False):
        if resize:
            image = cv2.resize(image, resize)
        pred = self.predictor(image)
        sem_seg_mask = pred['sem_seg'].argmax(dim=0).to(self.device)
        sem_seg_mask = sem_seg_mask.type(torch.uint8).numpy()

        if gray_only: # gray scale 라벨만 생성
            return sem_seg_mask
        return self.gray_to_color_mask(sem_seg_mask), sem_seg_mask

    def gray_to_color_mask(self, mask):
        return mask
        # mask_cats = np.unique(mask)
        # color_mask_shape = list(mask.shape) + [3]
        # color_mask = np.zeros(color_mask_shape, dtype=np.uint8)
        # for cat in mask_cats:
        #     # if cat > 14:
        #     #     continue
        #     color = np.array(self.category[cat]['color'], dtype=np.uint8)
        #     color_0 = color[0]
        #     color[0] = color[2]
        #     color[2] = color_0
        #     color_mask[mask == cat] = color
        #
        # return color_mask

    def pred_origin_merge(self, image, color_mask):
        color_mask_origin_image_mix = (image / 2 + color_mask / 2).astype(np.uint8)
        merged_image = cv2.hconcat([image, color_mask_origin_image_mix])
        return merged_image

    def __call__(self, image, resize=None, gray_only=False):
        return self.segment(image, resize=resize, gray_only=gray_only)


def main():
    cfg_file_path = config.SEGMENTATION_CONFIG_PATH # mask2former의 config 파일 경로 설정
    cpt_file_path = config.SEGMENTATION_WEIGHT_PATH # 체크포인트 파일 경로 설정
    category_def = config.SEGLABELCOLOR # 클래스 이름 및 컬러 설정
    model = SegmentNet(cfg_file_path, cpt_file_path, category_def) # 모델 생성

    img_dir = ''
    img_list = sorted(os.listdir(img_dir))

    for img_name in tqdm(img_list):
        img = cv2.imread(os.path.join(img_dir, img_name))
        color_mask, gray_mask = model(img) # 컬러 예측, 흑백 예측 출력, gray_only에 True 입력시 흑백 예측만 출력
        merged_image = model.pred_origin_merge(img, color_mask) # 컬러 라벨과 원본 이미지를 1:1 비율로 섞은 이미지와 원본 이미지를 묶어서 출력
        cv2.imshow('merged_image', merged_image)
        key = cv2.waitKey()
        if key == 120: # x 키 입력시 종료
            exit()


if __name__ == "__main__":
    from tqdm import tqdm # 진행 정도 확인
    sys.path.append(os.path.abspath('../../map_build')) # map_build를 path에 추가하여 import 준비

    import config # map_build의 config import

    main()
