import numpy as np
import cv2
import math
import geopandas as gpd
import pandas as pd
import os
from glob import glob
from PIL import Image
from pyproj import Transformer
from shapely.ops import transform
from shapely.geometry import LineString
from tqdm import tqdm


from matcher.config import config


# 파일 이름에서 WebMercator 좌표 추출
def extract_coords(filename):
    parts = filename.split('_')[1].replace('.png', '').split(',')
    return list(map(float, parts))


def transform_geometry(geometry):
    if geometry is not None:
        # transform 함수를 수정하여 3D 좌표에서 z 값을 무시합니다.
        if geometry.geom_type == 'LineString':
            return LineString([transformer.transform(x, y) for x, y, *_ in geometry.coords])
        else:
            # 기타 지오메트리 유형에 대해서는 모든 좌표에서 z 값을 무시합니다.
            return transform(lambda x, y, z=0: transformer.transform(x, y), geometry)
    else:
        return None


# 도로 데이터를 픽셀 좌표로 변환
def convert_geometry_to_pixels(geom):
    return [coords_to_pixels(x, y, x_min, y_max, x_max, y_min, width, height) for x, y, *_ in np.array(geom.coords)]


# 좌표를 이미지 픽셀로 변환
def coords_to_pixels(x, y, x_min, y_max, x_max, y_min, width, height):
    x_pixel = int((width * (x - x_min) / (x_max - x_min)) * ratio + x_ad)
    y_pixel = int((height * (y - y_max) / (y_min - y_max)) * ratio + y_ad)
    return x_pixel, y_pixel


# 이미지에 도로 그리기 함수
def draw_roads(img, pixel_coords):
    for line_pixels in pixel_coords:
        prev_point = None
        if np.all(np.isnan(line_pixels)) or np.all(np.isnan(line_pixels)):
            continue
        for point in line_pixels:
            if prev_point is not None:
                cv2.line(img, prev_point, point, (0, 255, 255), 1)
            prev_point = point
    return img


def find_files(root_folder, file_name_to_find):
    found_files = []
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if os.path.join(subdir, file).endswith(file_name_to_find):
                found_files.append(os.path.join(subdir, file))
    return found_files

def crop_center_cv(img, crop_width, crop_height):
    if img is None:
        print("Image not found")
        return

    # 이미지의 크기 계산
    height, width = img.shape[:2]
    center_x, center_y = width // 2, height // 2
    left = center_x - crop_width // 2
    top = center_y - crop_height // 2
    right = center_x + crop_width // 2
    bottom = center_y + crop_height // 2

    # 중앙을 기준으로 이미지 자르기
    cropped_img = img[top:bottom, left:right]

    return cropped_img

root_folder = '/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/국토정보플랫폼/국토지리/unzip'
road_links_paths = find_files(root_folder, '/HDMap_UTM52N_타원체고/B2_SURFACELINEMARK.shp')
road_links_paths.sort()


total_road_links = []
total_ID = []
total_kind = []
total_type = []
error_files = []
transformer = Transformer.from_crs('EPSG:32652', 'EPSG:3857', always_xy=True)
for r_l_path in tqdm(road_links_paths, desc="Transforming Road Datas"):
    try:
        road_links = gpd.read_file(r_l_path)
        road_links['transformed_geometry'] = road_links['geometry'].apply(transform_geometry)
        if len(road_links['transformed_geometry']) > 1:
            total_road_links.append(road_links['transformed_geometry'])

            #
            total_ID.append(road_links['ID'])
            total_kind.append(road_links['Kind'])
            total_type.append(road_links['Type'])
            #

    except Exception as e:
        print(e)
        error_files.append(r_l_path)
    if len(total_road_links) == 10:
        break

print(error_files)





# 조정 변수 초기화
x_ad, y_ad = 0, 0
ratio = 1
image_num = 0


# image_list = glob("/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/국토정보플랫폼/gookto_image/copy_00001_2/nobox/*.png")
image_list = glob(config.ImagesFolder + "/*.png")
# image_list = glob("/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/국토정보플랫폼/해상도테스트/origin_webmercator" + "/*.png")
image_list.sort()

# 메인 루프
while True:
    ##### 작은 이미지들
    image_path = image_list[image_num]
    print(image_path)
    image_name = image_path.split('/')[-1]
    x_min, y_min, x_max, y_max = extract_coords(image_name)
    width, height = Image.open(image_path).size

    image = cv2.imread(image_path)
    for i in tqdm(total_road_links, desc="Drawing Road Datas"):
        try:
            road_links['pixel_coords'] = i.apply(convert_geometry_to_pixels)
            image = draw_roads(image, road_links['pixel_coords'])
        except:
            print(i)
            continue
    # cv2.imshow('Road Map', image)
    # cv2.imshow('original', cv2.imread(image_path))

    # save_root = os.path.join(config.SaveFolder, image_name)
    save_root = image_path.replace("origin_webmercator", "drawn")
    # cv2.imwrite(save_root, image)
    print(f"save {save_root}")
    image_num += 1




