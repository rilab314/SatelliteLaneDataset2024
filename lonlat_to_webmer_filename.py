from pyproj import Transformer
import cv2
import os
from glob import glob

lonlat_to_web = Transformer.from_crs("epsg:4326", "epsg:3857")
web_to_lonlat = Transformer.from_crs("epsg:3857", "epsg:4326")

root_folder = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/국토정보플랫폼/국토위성이미지_사진중앙경도위도표시"

image_list = glob(os.path.join(root_folder, "*.png"))


def lonlat_to_webmercator_for_side(lon, lat):
    x1, y2 = lonlat_to_web.transform(lat - 0.0010985, lon - 0.002641)
    x2, y1 = lonlat_to_web.transform(lat + 0.0010985, lon + 0.002641)
    return x1, y2, x2, y1

file_num = 1
for i in image_list:
    file_name = i.split("/")[-1]
    lon1, lat1 = float(file_name.split(",")[0]), float(file_name.split(",")[1][:-4])
    x1, y2, x2, y1 = lonlat_to_webmercator_for_side(lon1, lat1)
    re_named = i.replace(file_name, f"{str(file_num).zfill(5)}_{x1},{y2},{x2},{y1}.png")
    os.rename(i, re_named)
    file_num += 1