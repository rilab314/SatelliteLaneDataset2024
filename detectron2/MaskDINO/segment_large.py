# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import os, sys
import cv2

from shapely.errors import ShapelyDeprecationWarning
import warnings
warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


from segment import SegmentNet

def main():
    cfg_file_path = './configs/ade20k/semantic-segmentation/maskdino_R50_bs16_160k_steplr_satellite_2024.yaml' # mask2former의 config 파일 경로 설정
    cpt_file_path = './output/best_model_0074999.pth' # 체크포인트 파일 경로 설정
    # 클래스 이름 및 컬러 설정
    category_def = [{'id': 0, 'name': 'None', 'color':(0, 0, 0)},
                    {'id': 1, 'name': 'ignore_label', 'color':(0, 0, 0)},
                    {'id': 2, 'name': 'center_line', 'color': (77, 77, 255)},
                    {'id': 3, 'name': 'u_turn_zone_line', 'color':(77, 178, 255)},
                    {'id': 4, 'name': 'lane_line', 'color':(77, 255, 77)},
                    {'id': 5, 'name': 'bus_only_lane', 'color':(255, 153, 77)},
                    {'id': 6, 'name': 'edge_line', 'color':(255, 77, 77)},
                    {'id': 7, 'name': 'path_change_restriction_line', 'color':(178, 77, 255)},
                    {'id': 8, 'name': 'no_parking_stopping_line', 'color':(77, 255, 178)},
                    {'id': 9, 'name': 'guiding_line', 'color':(255, 178, 77)},
                    {'id': 10, 'name': 'stop_line', 'color':(77, 102, 255)},
                    {'id': 11, 'name': 'safety_zone', 'color':(255, 77, 128)},
                    {'id': 12, 'name': 'bicycle_lane', 'color':(128, 255, 77)},
                    ]

    model = SegmentNet(cfg_file_path, cpt_file_path, category_def) # 모델 생성

    img_dir = './datasets/satellite_ade20k_241213/images/validation'
    img_list = sorted(os.listdir(img_dir))
    save_dir = './inference_result/241202'
    os.makedirs(save_dir, exist_ok=True)
    for img_name in tqdm(img_list):
        img = cv2.imread(os.path.join(img_dir, img_name))
        color_mask, gray_mask = model(img)  # 컬러 예측, 흑백 예측 출력, gray_only에 True 입력시 흑백 예측만 출력
        # merged_image = model.pred_origin_merge(img, color_mask)  # 컬러 라벨과 원본 이미지를 1:1 비율로 섞은 이미지와 원본 이미지를 묶어서 출력
        cv2.imwrite(os.path.join(save_dir, img_name), color_mask)
        # cv2.imshow('merged_image', merged_image)
        # key = cv2.waitKey()
        # if key == 120: # x 키 입력시 종료
        #     exit()


if __name__ == "__main__":
    from tqdm import tqdm # 진행 정도 확인
    sys.path.append(os.path.abspath('../../map_build')) # map_build를 path에 추가하여 import 준비

    # import config # map_build의 config import

    main()
