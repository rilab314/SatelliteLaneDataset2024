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
from shapely.geometry import LineString, Polygon
from tqdm import tqdm


# 파일 이름에서 WebMercator 좌표 추출
def extract_coords(filename):
    parts = filename.split('_')[1].replace('.png', '').split(',')
    return list(map(float, parts))


def transform_geometry(geometry):
    if geometry is not None:
        if geometry.geom_type == 'LineString':
            return LineString([transformer.transform(x, y) for x, y, *_ in geometry.coords])
        elif geometry.geom_type == 'Polygon':
            # Polygon의 외곽선과 내부 링을 변환합니다.
            exterior = geometry.exterior
            transformed_exterior = LineString([transformer.transform(x, y) for x, y, *_ in exterior.coords])

            interiors = []
            for interior in geometry.interiors:
                transformed_interior = LineString([transformer.transform(x, y) for x, y, *_ in interior.coords])
                interiors.append(transformed_interior)

            return Polygon(transformed_exterior, interiors)
        else:
            # 기타 지오메트리 유형에 대해서는 모든 좌표에서 z 값을 무시합니다.
            return transform(lambda x, y, z=0: transformer.transform(x, y), geometry)
    else:
        return None


# # 도로 데이터를 픽셀 좌표로 변환
# def convert_geometry_to_pixels(geom):
#     return [coords_to_pixels(x, y, x_min, y_max, x_max, y_min, width, height) for x, y, *_ in np.array(geom.bounds)]
def convert_geometry_to_pixels(geom):
    # geom이 Polygon 또는 LineString과 같은 단일 지오메트리 객체라고 가정
    if isinstance(geom, Polygon):
        exterior_coords = [coords_to_pixels(x, y, x_min, y_max, x_max, y_min, width, height) for x, y in np.array(geom.exterior.coords)]
        interiors = [[coords_to_pixels(x, y, x_min, y_max, x_max, y_min, width, height) for x, y in np.array(interior.coords)] for interior in geom.interiors]
        return {'exterior': exterior_coords, 'interiors': interiors}
    elif isinstance(geom, LineString):
        return [coords_to_pixels(x, y, x_min, y_max, x_max, y_min, width, height) for x, y in np.array(geom.coords)]
    return []


# 좌표를 이미지 픽셀로 변환
def coords_to_pixels(x, y, x_min, y_max, x_max, y_min, width, height):
    x_pixel = int((width * (x - x_min) / (x_max - x_min)) * ratio + x_ad)
    y_pixel = int((height * (y - y_max) / (y_min - y_max)) * ratio + y_ad)
    return x_pixel, y_pixel


# 이미지에 도로 그리기 함수
def draw_roads(img, geometries):
    for geom_dict in geometries:
        if 'exterior' in geom_dict:
            # 외곽선 그리기
            exterior_coords = geom_dict['exterior']
            if len(exterior_coords) > 1:
                cv2.polylines(img, [np.array(exterior_coords, dtype=np.int32)], True, (255, 255, 0), 1)

        if 'interiors' in geom_dict:
            # 내부 링(구멍) 그리기
            for interior_coords in geom_dict['interiors']:
                if len(interior_coords) > 1:
                    cv2.polylines(img, [np.array(interior_coords, dtype=np.int32)], True, (255, 0, 255), 1)

    return img



def find_files(root_folder, file_name_to_find):
    found_files = []
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if os.path.join(subdir, file).endswith(file_name_to_find):
                found_files.append(os.path.join(subdir, file))
    return found_files



root_folder = '/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/국토정보플랫폼/국토지리/unzip'
road_links_paths = find_files(root_folder, '/HDMap_UTM52N_타원체고/B3_SURFACEMARK.shp')

road_links_paths.sort()


total_road_links = []
error_files = []
transformer = Transformer.from_crs('EPSG:32652', 'EPSG:3857', always_xy=True)
for r_l_path in tqdm(road_links_paths, desc="Transforming Road Datas"):
    try:
        road_links = gpd.read_file(r_l_path)
        road_links['transformed_geometry'] = road_links['geometry'].apply(transform_geometry)
        if len(road_links['transformed_geometry']) > 1:
            total_road_links.append(road_links['transformed_geometry'])
    except Exception as e:
        print("Error: ",e)
        error_files.append(r_l_path)

    ##
    if len(total_road_links) == 10:
        break


print(error_files)





# 조정 변수 초기화
x_ad, y_ad = 0, 0
ratio = 1
image_num = 0


image_list = glob("/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/국토정보플랫폼/국토위성이미지_사진중앙경도위도표시/nobox/*.png")
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

    # total_road_links가 GeoDataFrames 또는 GeoSeries의 리스트라고 가정
    for geom_collection in tqdm(total_road_links, desc="Drawing Road Datas"):
        # 컬렉션 내 각 지오메트리를 처리
        for geometry in geom_collection.geometry:  # 지오메트리를 반복 처리
            try:
                pixel_coords = convert_geometry_to_pixels(geometry)
                image = draw_roads(image, [pixel_coords])  # draw_roads 함수가 리스트를 기대한다면 리스트로 묶어서 전달
            except Exception as e:
                print(f"지오메트리 처리 중 오류: {e}")

        # cv2.imshow('Road Map', image)
        # cv2.imshow('original', cv2.imread(image_path))
        # cv2.waitKey(0)
    if x_ad != 0 or y_ad != 0:
        save_root = image_path.replace("box", f"box_drew/x{x_ad}y{y_ad}")

    else:
        save_root = image_path.replace('box', 'box_drew')

    # cv2.imwrite(save_root, image)
    print(f"save {save_root}")
    image_num += 1



