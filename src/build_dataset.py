import os
import cv2
import copy
import numpy as np
from glob import glob
from tqdm import tqdm

import src.config.config as cfg
from src.utils.json_file_io import JsonFileReader, write_to_json
from src.utils.icp_algorithm import IcpApplier
from src.dto import GeometryType


def build_dataset():
    """
    align label coordinates
    crop images
    rename images and labels in new folder
    :return:
    """
    image_save_path = cfg.IMAGE_PATH
    label_save_path = cfg.LABEL_PATH

    # image_process_save(cfg.ORIGINAL_IMAGE_PATH, image_save_path)
    label_align_save(cfg.UNMATCHED_LABEL_PATH, image_save_path, label_save_path)


def image_process_save(image_root_path, image_save_path):
    os.makedirs(image_save_path, exist_ok=True)
    image_list = sorted(glob(os.path.join(image_root_path, '*.png')))
    for img_path in tqdm(image_list, desc='Crop images...'):
        cropped_image = crop_center(img_path, [cfg.IMAGE_SIZE_h, cfg.IMAGE_SIZE_w])
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        cv2.imwrite(os.path.join(image_save_path, os.path.basename(img_path)), cropped_image)


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
    webmercator_cx, webmercator_cy = transformer.transform(lon, lat)
    renamed_file = file_name.replace(str(lon), str(webmercator_cx)).replace(str(lat), str(webmercator_cy))
    return renamed_file


def label_align_save(label_root_path, image_root_path, aligned_label_save_root_path):
    os.makedirs(aligned_label_save_root_path, exist_ok=True)
    reader = JsonFileReader(label_root_path)
    label_list = reader.list_files()
    for label_path in tqdm(label_list, desc='Generating labels...'):
        origin_geometries = reader.read(label_path)
        label_name_coord = os.path.basename(label_path).split('.json')[0]
        try:
            image_path = glob(os.path.join(image_root_path, f"*{label_name_coord}.png"))[0]
        except Exception as e:
            print(label_name_coord)
            continue

        image = cv2.imread(image_path)
        cv2.imshow('image', image)
        metadata = [origin_geometries[0]]
        obj_data_list = origin_geometries[1:]
        line_mask = create_line_mask(image, obj_data_list)
        filtered_image = filter_road_objects(image, line_mask)
        # ICP
        transform = get_icp_transform(source_image=line_mask, target_image=filtered_image, vis=0)
        transformed_data = transform_data(transform, obj_data_list)
        # save_drawn_image(image, line_mask, filtered_image, transformed_data, obj_data_list, image_path, cfg.DATASET_PATH)
        # label data save
        label_save_path = os.path.join(aligned_label_save_root_path, os.path.basename(image_path).split('.png')[0]+'.json')
        write_to_json(label_save_path, metadata+transformed_data)


def create_line_mask(image, obj_data_list):
    line_mask = np.zeros_like(image[:, :, 0])
    for obj in obj_data_list:
        if obj.geometry_type in ['MULTILINE_STRING', 'MULTIPOLYGON']:
            for obj_points in obj.image_points:
                draw_road_object(line_mask, obj_points, obj.type_id)
        else:
            draw_road_object(line_mask, obj.image_points, obj.type_id)
    cv2.imshow('line mask', line_mask)
    return line_mask

def draw_road_object(line_mask, object_points, object_type_id):
    points = np.array(object_points, dtype=np.int32)
    if object_type_id in ['1', '5']:
        # Create a temporary mask for the polygon
        temp_mask = np.zeros_like(line_mask, dtype=np.uint8)

        # Fill the polygon on the temporary mask
        cv2.fillPoly(temp_mask, [points], 255)

        # Erode the polygon's boundary to shrink by 1 pixel
        eroded_mask = cv2.erode(temp_mask, kernel=np.ones((3, 3), np.uint8), iterations=1)

        # Copy the eroded mask to the main mask
        line_mask[eroded_mask > 0] = 255
    else:
        cv2.polylines(line_mask, [points], False, 255, 1)

def filter_road_objects(src_image, obj_mask):
    output_mask = filter_by_color(src_image)
    output_mask = filter_large_objects(output_mask)
    masked_image = filter_by_mask(output_mask, obj_mask)
    cv2.imshow('filter by mask', masked_image)
    return masked_image


def filter_by_color(src_image):
    hsv = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    cv2.imshow('value', value)
    binary = cv2.adaptiveThreshold(value, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, -10)
    cv2.imshow('binary', binary)
    return binary


def filter_large_objects(object_mask):
    eroded = cv2.erode(object_mask, np.ones((3, 3), np.uint8), iterations=1)
    dilated = cv2.dilate(eroded, np.ones((3, 3), np.uint8), iterations=1)
    cv2.imshow('eroded', dilated)
    object_mask[dilated > 0] = 0
    cv2.imshow('object_mask', object_mask)
    return object_mask


def filter_by_mask(src_image, obj_mask):
    image = src_image.copy()
    dilated_mask = cv2.dilate(obj_mask, np.ones((3, 3), np.uint8), iterations=5)
    image[dilated_mask == 0] = 0
    return image


def get_icp_transform(source_image, target_image, vis=0):
    icp_applier = IcpApplier()
    transform = icp_applier.icp_apply(source_image, target_image, vis)
    return transform


def transform_data(transform, data):
    data_cp = copy.deepcopy(data)
    for road_obj in data_cp:
        if road_obj.geometry_type in ['MULTILINE_STRING', 'MULTIPOLYGON']:
            modified_points = []
            for lane_points in road_obj.image_points:
                transformed_lane = transform_points(lane_points, transform)
                modified_points.append(transformed_lane)
            road_obj.image_points = modified_points
        else:
            road_obj.image_points = transform_points(road_obj.image_points, transform)

    return data_cp

def transform_points(points, transform):
    return [transform_point(point, transform) for point in points]

def transform_point(point, transform):
    homogeneous_point = np.array([point[0], point[1], 1])
    transformed_point = np.dot(transform, homogeneous_point)
    return [int(np.round(transformed_point[0])), int(np.round(transformed_point[1]))]

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
    # cv2.waitKey(0)
    cv2.imwrite(save_path, vstack_all_images)


def draw_objects(image, data):
    image = image.copy()
    for obj in data:
        if obj.type_id in ['1', '5']:
            if obj.geometry_type == 'POLYGON':
                if len(obj.image_points) > 1:
                    cv2.polylines(image, [np.array(obj.image_points, dtype=np.int32)], True, (255, 255, 0), 1)
            elif obj.geometry_type == 'MULTIPOLYGON':
                for polygon in obj.image_points:
                    if len(polygon) > 1:
                        cv2.polylines(image, [np.array(polygon, dtype=np.int32)], True, (255, 255, 0), 1)
        else:
            prev_point = None
            if obj.geometry_type == 'LINE_STRING':
                if np.all(np.isnan(obj.image_points)) or np.all(np.isnan(obj.image_points)):
                    return image
                for point in obj.image_points:
                    if prev_point is not None:
                        cv2.line(image, prev_point, tuple(point), (0, 255, 255), 1)
                    prev_point = tuple(point)
            elif obj.geometry_type == 'MULTILINE_STRING':
                for line in obj.image_points:
                    if np.all(np.isnan(line)) or np.all(np.isnan(line)):
                        return image
                    for point in line:
                        if prev_point is not None:
                            cv2.line(image, prev_point, tuple(point), (0, 255, 255), 1)
                        prev_point = tuple(point)
    return image


if __name__ == '__main__':
    build_dataset()