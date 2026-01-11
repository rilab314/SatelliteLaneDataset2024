import os
import sys
import glob
import cv2
import numpy as np
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.json_file_io import JsonFileReader, write_to_json
from src.utils.icp_algorithm import IcpApplier


class LabelAligner:
    def __init__(self):
        self.label_path = None
        pass

    def align_main(self, src_label_path, dst_label_path):
        self.label_path = src_label_path
        json_files = glob.glob(os.path.join(src_label_path, '*.json'))
        twisted_files = glob.glob(os.path.join(src_label_path.replace('label', 'twisted'), '*.png'))
        reader = JsonFileReader()
        for idx, image_file in enumerate(twisted_files):
            json_file = image_file.replace('twisted', 'label').replace('.png', '.json')
            print(f'[{idx+1}/{len(json_files)}] {os.path.basename(json_file)}')
            label = reader.read(json_file)
            metadata = [label[0]]
            obj_data_list = label[1:]
            image = cv2.imread(image_file)
            aligned_label = self.align(image, obj_data_list, json_file)
            dst_file = os.path.join(dst_label_path, os.path.basename(json_file))
            write_to_json(dst_file, metadata + aligned_label)
    
    def align(self, image, label, json_file):
        line_mask = self.create_line_mask(image, label)
        filtered_image = self.filter_road_objects(image, line_mask)
        cv2.imshow('src image', image)
        self.show_result(image, filtered_image, line_mask, title='before align')

        icp_applier = IcpApplier()
        transform = icp_applier.icp_apply(source_image=line_mask, target_image=filtered_image)
        print('transform\n', transform)
        aligned_label = self.transform_data(transform, label)
        aligned_line_mask = self.create_line_mask(image, aligned_label)
        self.show_result(image, filtered_image, aligned_line_mask, title='after align')
    
        twisted_path = self.label_path.replace('label', 'twisted')
        os.makedirs(twisted_path, exist_ok=True)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            cv2.imwrite(os.path.join(twisted_path, os.path.basename(json_file).replace('.json', '.png')), image)
        else:
            cv2.waitKey(0)
        return aligned_label

    def create_line_mask(self, image, obj_data_list):
        line_mask = np.zeros_like(image[:, :, 0])
        for obj in obj_data_list:
            if obj.category not in ['center_line', 'stop_line']:
                continue
            if obj.geometry_type in ['MULTILINE_STRING', 'MULTIPOLYGON']:
                for obj_points in obj.image_points:
                    self.draw_road_object(line_mask, obj_points, obj.type_id)
            else:
                self.draw_road_object(line_mask, obj.image_points, obj.type_id)
        return line_mask

    def draw_road_object(self, line_mask, object_points, object_type_id):
        points = np.array(object_points, dtype=np.int32)
        if object_type_id in ['1', '5']:
            temp_mask = np.zeros_like(line_mask, dtype=np.uint8)
            cv2.fillPoly(temp_mask, [points], 255)
            eroded_mask = cv2.erode(temp_mask, kernel=np.ones((3, 3), np.uint8), iterations=1)
            line_mask[eroded_mask > 0] = 255
        else:
            cv2.polylines(line_mask, [points], False, 255, 1)

    def filter_road_objects(self, src_image, obj_mask):
        color_mask = self.filter_by_color(src_image)
        output_mask = self.filter_large_objects(color_mask)
        masked_image = self.filter_by_mask(output_mask, obj_mask)
        return masked_image

    def filter_by_color(self, src_image):
        hsv = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
        binary = cv2.adaptiveThreshold(value, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, -10)
        return binary

    def filter_large_objects(self, src_mask):
        object_mask = src_mask.copy()
        eroded = cv2.erode(object_mask, np.ones((3, 3), np.uint8), iterations=2)
        dilated = cv2.dilate(eroded, np.ones((3, 3), np.uint8), iterations=2)
        object_mask[dilated > 0] = 0
        return object_mask

    def filter_by_mask(self, src_image, obj_mask):
        image = src_image.copy()
        road_mask = cv2.dilate(obj_mask, np.ones((3, 3), np.uint8), iterations=4)
        image[road_mask == 0] = 0
        return image

    def transform_data(self, transform, data):
        data_cp = copy.deepcopy(data)
        for road_obj in data_cp:
            if road_obj.geometry_type in ['MULTILINE_STRING', 'MULTIPOLYGON']:
                modified_points = []
                for lane_points in road_obj.image_points:
                    transformed_lane = self.transform_points(lane_points, transform)
                    modified_points.append(transformed_lane)
                road_obj.image_points = modified_points
            else:
                road_obj.image_points = self.transform_points(road_obj.image_points, transform)

        return data_cp

    def transform_points(self, points, transform):
        return [self.transform_point(point, transform) for point in points]

    def transform_point(self, point, transform):
        homogeneous_point = np.array([point[0], point[1], 1])
        transformed_point = np.dot(transform, homogeneous_point)
        return [int(np.round(transformed_point[0])), int(np.round(transformed_point[1]))]
    
    def show_result(self, image, filtered_image, line_mask, alpha=0.5, title='result'):
        """
        image: (H, W, 3) uint8 - 원본 이미지
        filtered_image: (H, W) - 예측된 이진 마스크 (0 or 255/1)
        line_mask: (H, W) - 정답 이진 마스크 (0 or 255/1)
        alpha: 투명도 (0.0 ~ 1.0)
        """
        img_display = image.astype(np.uint8).copy()
        pred_mask = filtered_image > 0
        gt_mask = line_mask > 0
        intersection = pred_mask & gt_mask
        pred_only = pred_mask & ~gt_mask
        gt_only = ~pred_mask & gt_mask
        mask_indices = intersection | pred_only | gt_only

        color_layer = np.zeros_like(img_display)
        color_layer[intersection] = [0, 0, 255]
        color_layer[pred_only] = [255, 0, 0]
        color_layer[gt_only] = [0, 255, 0]        
        
        if mask_indices.any():
            blended = img_display * (1 - alpha) + color_layer * alpha
            img_display[mask_indices] = blended[mask_indices].astype(np.uint8)
        cv2.imshow(title, img_display)


if __name__ == '__main__':
    label_path = '/media/dolphin/My Book/Ongoing/youn_ws/dataset/satellite_good_matching_250206/label'
    dst_path = label_path.replace('label', 'aligned')
    os.makedirs(dst_path, exist_ok=True)

    label_aligner = LabelAligner()
    label_aligner.align_main(src_label_path=label_path, dst_label_path=dst_path)

