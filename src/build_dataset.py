import os
import cv2
import json
import copy
import numpy as np
import open3d as o3d
from glob import glob
from pyproj import Transformer


import src.config.config as cfg
from src.utils.json_file_io import JsonFileReader, write_to_json
from src.utils.icp_algorithm import IcpApplier


def build_dataset():
    """
    align label coordinates
    crop images
    rename images and labels in new folder
    :return:
    """
    image_save_path = os.path.join(cfg.DATASET_PATH, 'image')
    label_save_path = os.path.join(cfg.DATASET_PATH, 'label')

    image_process_save(cfg.ORIGINAL_IMAGE_PATH, image_save_path)
    label_align_save(cfg.LABEL_PATH, image_save_path, label_save_path)



def image_process_save(image_root_path, image_save_path):
    lonlat_to_web = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)

    image_list = glob(os.path.join(image_root_path, '*.png'))
    image_list.sort()
    for img_path in image_list:
        cropped_image = crop_center(img_path, [cfg.IMAGE_SIZE_h, cfg.IMAGE_SIZE_w])
        renamed_filename = rename_file_lonlat2webmercator(img_path, lonlat_to_web)

        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        cv2.imwrite(os.path.join(image_save_path, renamed_filename), cropped_image)




def crop_center(image_path, output_size):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    crop_width, crop_height = output_size

    start_x = width // 2 - crop_width // 2
    start_y = height // 2 - crop_height // 2

    cropped_img = img[start_y:start_y + crop_height, start_x:start_x + crop_width]
    return cropped_img

def rename_file_lonlat2webmercator(img_path, transformer):
    file_name = img_path.split('/')[-1]
    lon, lat = float(file_name.split(',')[0].split('_')[1]), float(file_name.split(",")[1][:-4])
    webmercator_cx, webmercator_cy = transformer.transform(lat, lon)
    renamed_file = file_name.replace(str(lon), str(webmercator_cx)).replace(str(lat), str(webmercator_cy))
    return renamed_file


def label_align_save(label_root_path, image_root_path, aligned_label_save_root_path):
    reader = JsonFileReader(label_root_path)
    label_list = reader.list_files()
    for label_path in label_list:
        origin_geometries = reader.read(label_path)

        label_name_coord = os.path.basename(label_path).split('.')[0]
        image_path = glob(os.path.join(image_root_path, f"*{label_name_coord}.png"))[0]
        image = cv2.imread(image_path)


        metadata = [origin_geometries[0]]
        obj_data = origin_geometries[1:]
        line_mask = create_line_mask(image, obj_data)
        filtered_image = filter_road_objects(image, line_mask)

        # ICP
        transform = get_icp_transform(source_image=line_mask, target_image=filtered_image, vis=0)
        transformed_data = transform_data(transform, obj_data)

        # save_drawn_image(image, line_mask, filtered_image, transformed_data, obj_data, image_path, cfg.DATASET_PATH)


        # label data save
        label_save_path = os.path.join(aligned_label_save_root_path, os.path.basename(image_path))
        write_to_json(label_save_path, metadata+transformed_data)


def filter_road_objects(src_image, obj_mask):
    output_mask = filter_by_color(src_image)
    output_mask = filter_large_objects(output_mask)
    masked_image = filter_by_mask(output_mask, obj_mask)
    cv2.imshow('filter by mask', masked_image)
    return masked_image

def filter_by_mask(self, src_image, obj_mask):
    image = src_image.copy()
    dilated_mask = cv2.dilate(obj_mask, np.ones((3, 3), np.uint8), iterations=5)
    image[dilated_mask == 0] = 0
    return image

def filter_by_color(self, src_image):
    hsv = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    # yellow_range = (45, 75)
    # yellow_mask = hsv[:, :, 0]
    # yellow_mask[np.logical_and(yellow_mask >= yellow_range[0], yellow_mask <= yellow_range[1])] = 255
    # yellow_mask[np.logical_or(yellow_mask < yellow_range[0], yellow_mask > yellow_range[1])] = 0
    # yellow_mask[value < 150] = 0
    # cv2.imshow('yellow', yellow_mask)
    cv2.imshow('value', value)
    binary = cv2.adaptiveThreshold(value, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, -10)
    cv2.imshow('binary', binary)
    return binary

def filter_large_objects(self, object_mask):
    print('object_mask', object_mask.shape)
    eroded = cv2.erode(object_mask, np.ones((3, 3), np.uint8), iterations=1)
    dilated = cv2.dilate(eroded, np.ones((3, 3), np.uint8), iterations=1)
    cv2.imshow('eroded', dilated)
    object_mask[dilated > 0] = 0
    cv2.imshow('object_mask', object_mask)
    return object_mask

def align_masks(self, src_image, object_image):
    align_matrix = np.eye(2, 3)
    result_img = np.zeros_like(src_image)
    for x in range(5):
        for y in range(5):
            align_matrix[:, 2] = (x, y)
            warped_image = cv2.warpAffine(src_image, align_matrix, (object_image.shape[1], object_image.shape[0]))
            score = np.sum(warped_image * object_image)
            result_img[np.logical_and(warped_image > 0, object_image > 0)] = (255,255,255)
            result_img[np.logical_and(warped_image == 0, object_image > 0)] = (0,0,255)
            result_img[np.logical_and(warped_image > 0, object_image == 0)] = (255,0,0)
            # show lanes on rgb image


def objects_image_filter(self, image, line_mask):
    output = self.filter_white_yellow(image)
    output = self.filter_and_remove_large_areas(output, 500)
    output = self.filter_mask_overlab(output, line_mask)
    return output

def get_icp_transform(source_image, target_image, vis=0):
    icp_applier = IcpApplier()
    transform = icp_applier.icp_apply(source_image, target_image, vis)
    return transform

def transform_data(transform, data):
    data_cp = copy.deepcopy(data)
    for obj_dict in data_cp:
        modified_points = []
        for point in obj_dict['image_points']:
            homogeneous_point = np.array([point[0], point[1], 1])
            transformed_point = np.dot(transform, homogeneous_point)
            modified_points.append([int(np.round(transformed_point[0])), int(np.round(transformed_point[1]))])
        obj_dict['image_points'] = modified_points
    return data_cp




def matching_try(filtered_image, data):

    max_overlap_area = None
    optimal_data = None
    for x_ratio in np.arange(0.95, 1.05, 0.01):
        for y_ratio in np.arange(0.95, 1.05, 0.01):
            for x_shift in range(-5 ,5):
                for y_shift in range(-5 ,5):
                    data_cp = copy.deepcopy(data)

                    for obj_dict in data_cp:
                        modified_points = []
                        for point in obj_dict["pixel_points"]:
                            new_x = int((point[0] + x_shift) * x_ratio)
                            new_y = int((point[1] + y_shift) * y_ratio)
                            modified_points.append([new_x, new_y])
                        obj_dict["pixel_points"] = modified_points

                    overlap_area = self.calculate_overlap(filtered_image, data_cp)
                    if max_overlap_area is None:
                        max_overlap_area = overlap_area
                    if overlap_area < max_overlap_area:
                        max_overlap_area = overlap_area
                        optimal_data = data_cp
                        print(f"max: {max_overlap_area=}, x_ratio: {x_ratio}, y_ratio{y_ratio}, x_shift{x_shift}, y_shift{y_shift}")

    return optimal_data

def filter_white_yellow(image):
    output = image.copy()
    lower_white = np.array([150, 150, 150])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(output, lower_white, upper_white)
    white_image = cv2.bitwise_and(output, output, mask=white_mask)

    img_hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    yellow_image = cv2.bitwise_and(output, output, mask=yellow_mask)

    output = cv2.addWeighted(white_image, 1.0, yellow_image, 1.0, 0.0)
    return output

def filter_and_remove_large_areas(image, area_threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_threshold:
            cv2.drawContours(image, [contour], -1, (0, 0, 0), -1)
    return image

def filter_mask_overlab(image, line_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dilated_line_mask = cv2.dilate(line_mask, None, iterations=4)
    overlab_area = cv2.bitwise_and(gray, dilated_line_mask)
    return overlab_area


def calculate_overlap(filtered_image, data):
    gray_filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    line_mask = create_line_mask(filtered_image, data)
    overlap_mask = cv2.bitwise_and(gray_filtered_image, line_mask)
    overlap_area = np.sum(overlap_mask > 0)
    # imshow____(np.hstack((line_mask, overlap_mask)))
    return overlap_area


def create_line_mask(image, geometry):
    line_mask = np.zeros_like(image[:, :, 0])
    for obj in geometry:
        points = np.array(obj['image_points'], dtype=np.int32)
        if obj['type_id'] in ['1', '5']:
           cv2.fillPoly(line_mask, [points], 255)
        else:
            cv2.polylines(line_mask, [points], False, 255, 1)
    return line_mask

def draw_objects(image, data):
    image = image.copy()
    for obj in data:
        if obj["type_id"] in ["1", "5"]:
            if len(obj["pixel_points"]) > 1:
                cv2.polylines(image, [np.array(obj["pixel_points"], dtype=np.int32)], True, (255, 255, 0), 1)
        else:
            prev_point = None
            if np.all(np.isnan(obj["pixel_points"])) or np.all(np.isnan(obj["pixel_points"])):
                return image
            for point in obj["pixel_points"]:
                if prev_point is not None:
                    cv2.line(image, prev_point, tuple(point), (0, 255, 255), 1)
                prev_point = tuple(point)

    return image

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_drawn_image(image, line_mask, filtered_image, transformed_data, obj_data, image_path, save_path):
    drawn_image = draw_objects(image, transformed_data)
    origin_drawn_image = draw_objects(image, obj_data)

    hstack_filter = np.hstack(
        (np.repeat(line_mask[:, :, np.newaxis], 3, axis=-1), np.repeat(filtered_image[:, :, np.newaxis], 3, axis=-1)))
    hstack_images = np.hstack((origin_drawn_image, drawn_image))
    vstack_all_images = np.vstack((hstack_images, hstack_filter))
    cv2.imshow("stack images", vstack_all_images)

    # image save
    image_name = image_path.split("/")[-1]
    save_root = os.path.join(save_path, "road_matching", "241015_adap_icp")
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    save_path = os.path.join(save_root, image_name)
    cv2.imwrite(save_path, vstack_all_images)