from pyproj import Transformer
import cv2
import os
import shutil
from glob import glob



def lonlat_to_webmercator_for_side(lon, lat, img_size):
    half_height_webm = 0.002641 / 1832 * img_size[0]
    half_width_webm = 0.0010985 / 956 * img_size[1]
    x1, y2 = lonlat_to_web.transform(lat - half_width_webm, lon - half_height_webm)
    x2, y1 = lonlat_to_web.transform(lat + half_width_webm, lon + half_height_webm)
    return x1, y2, x2, y1


def crop_center(image_path, output_size):
    img = cv2.imread(image_path)

    height, width = img.shape[:2]

    crop_width, crop_height = output_size

    start_x = width // 2 - crop_width // 2
    start_y = height // 2 - crop_height // 2

    if start_x < 0 or start_y < 0 or (start_x + crop_width > width) or (start_y + crop_height > height):
        raise ValueError("이미지 크기가 요구하는 크롭 크기보다 작습니다.")

    cropped_img = img[start_y:start_y + crop_height, start_x:start_x + crop_width]
    return cropped_img



if __name__ == '__main__':

    crop_size = [768, 768]
    root_folder = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/국토정보플랫폼/국토위성이미지_크롤러_241016/origin_lonlat"

    image_list = glob(os.path.join(root_folder, "*.png"))
    image_list.sort()

    lonlat_to_web = Transformer.from_crs("epsg:4326", "epsg:3857")
    web_to_lonlat = Transformer.from_crs("epsg:3857", "epsg:4326")
    crop_img_root = root_folder.replace("origin_lonlat", f"{crop_size[0]}x{crop_size[1]}/origin_image")
    if not os.path.exists(crop_img_root):
        os.makedirs(crop_img_root)

    for img_path in image_list:
        file_name = img_path.split("/")[-1]
        lon1, lat1 = float(file_name.split(",")[0][6:]), float(file_name.split(",")[1][:-4])
        x1, y2, x2, y1 = lonlat_to_webmercator_for_side(lon1, lat1, crop_size)
        cropped_image = crop_center(img_path, crop_size)
        re_named = img_path.replace("origin_lonlat", f"{crop_size[0]}x{crop_size[1]}").replace(file_name.split("_")[-1], f"{x1},{y2},{x2},{y1}.png")
        save_file_name = file_name.replace(file_name.split("_")[-1], f"{x1},{y2},{x2},{y1}.png")
        re_named = os.path.join(crop_img_root, save_file_name)
        cv2.imwrite(re_named, cropped_image)