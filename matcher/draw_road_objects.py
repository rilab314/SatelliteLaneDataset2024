import cv2
import os
import json
import copy
import numpy as np
import open3d as o3d
from glob import glob


from matcher.config.config import ImagesFolder
from matcher.icp_algorithm import IcpApplier

def imshow____(image, image_name="image", wait=0, run=1):
    if run:
        cv2.imshow(image_name, image)
        cv2.waitKey(wait)
        cv2.destroyAllWindows()


class DrawRoadObjects:
    def __init__(self, root_path):
        self.json_list = sorted(glob(os.path.join(root_path, "label", "*.json")))
        self.icp_vis = 0


    def process(self):
        for json_path in self.json_list:
            image_path = json_path.replace("label", "origin_image").replace(".json", ".png")
            image = cv2.imread(image_path)
            obj_data = self.load_json(json_path)
            line_mask = self.create_line_mask(image, obj_data)
            filtered_image = self.objects_image_filter(image, line_mask)
            # # imshow____(line_mask)
            # # imshow____(filtered_image, "filtered_image")

            # align_matrix = self.align_masks(object_image, filtered_image)

            # ICP
            transform = self.get_icp_transform(source_image=line_mask, target_image=filtered_image)
            transformed_data = self.transform_data(transform, obj_data)
            drawn_image = self.draw_objects(image, transformed_data)
            origin_drawn_image = self.draw_objects(image, obj_data)

            imshow____(np.hstack((origin_drawn_image, drawn_image)))
            #
            #
            # over = self.calculate_overlap(filtered_image, obj_data)
            # # optimal_data = self.matching_try(filtered_image, obj_data)
            #
            # # drawn_image = self.draw_objects(image, optimal_data)
            # origin_drawn_image = self.draw_objects(image, obj_data)
            # # imshow____(np.hstack((origin_drawn_image, drawn_image)))
            #
            # json_file_name = json_path.split("/")[-1]


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
        output = self.filter_and_remove_large_areas(output, 50)
        output = self.filter_mask_overlab(output, line_mask)
        return output

    def get_icp_transform(self, source_image, target_image):
        icp_applier = IcpApplier()
        transform = icp_applier.icp_apply(source_image, target_image, vis=self.icp_vis)
        return transform

    def transform_data(self, transform, data):
        data_cp = copy.deepcopy(data)
        for obj_dict in data_cp:
            modified_points = []
            for point in obj_dict["points"]:
                homogeneous_point = np.array([point[0], point[1], 1])
                transformed_point = np.dot(transform, homogeneous_point)
                modified_points.append([int(np.round(transformed_point[0])), int(np.round(transformed_point[1]))])
            obj_dict["points"] = modified_points
        return data_cp




    def matching_try(self, filtered_image, data):

        max_overlap_area = None
        optimal_data = None
        for x_ratio in np.arange(0.95, 1.05, 0.01):
            for y_ratio in np.arange(0.95, 1.05, 0.01):
                for x_shift in range(-5 ,5):
                    for y_shift in range(-5 ,5):
                        data_cp = copy.deepcopy(data)

                        for obj_dict in data_cp:
                            modified_points = []
                            for point in obj_dict["points"]:
                                new_x = int((point[0] + x_shift) * x_ratio)
                                new_y = int((point[1] + y_shift) * y_ratio)
                                modified_points.append([new_x, new_y])
                            obj_dict["points"] = modified_points

                        overlap_area = self.calculate_overlap(filtered_image, data_cp)
                        if max_overlap_area is None:
                            max_overlap_area = overlap_area
                        if overlap_area < max_overlap_area:
                            max_overlap_area = overlap_area
                            optimal_data = data_cp
                            print(f"max: {max_overlap_area=}, x_ratio: {x_ratio}, y_ratio{y_ratio}, x_shift{x_shift}, y_shift{y_shift}")

        return optimal_data

    def filter_white_yellow(self, image):
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

    def filter_and_remove_large_areas(self, image, area_threshold):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > area_threshold:
                cv2.drawContours(image, [contour], -1, (0, 0, 0), -1)
        return image

    def filter_mask_overlab(self, image, line_mask):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dilated_line_mask = cv2.dilate(line_mask, None, iterations=4)
        overlab_area = cv2.bitwise_and(gray, dilated_line_mask)
        return overlab_area


    def calculate_overlap(self, filtered_image, data):
        gray_filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
        line_mask = self.create_line_mask(filtered_image, data)
        overlap_mask = cv2.bitwise_and(gray_filtered_image, line_mask)
        overlap_area = np.sum(overlap_mask > 0)
        # imshow____(np.hstack((line_mask, overlap_mask)))
        return overlap_area


    def create_line_mask(self, image, data):
        line_mask = np.zeros_like(image[:, :, 0])
        for obj in data:
            points = np.array(obj['points'], dtype=np.int32)
            if obj['type_id'] in ['1', '5']:
               cv2.fillPoly(line_mask, [points], 255)
            else:
                cv2.polylines(line_mask, [points], False, 255, 1)
        return line_mask

    def draw_objects(self, image, data):
        image = image.copy()
        for obj in data:
            if obj["type_id"] in ["1", "5"]:
                if len(obj["points"]) > 1:
                    cv2.polylines(image, [np.array(obj["points"], dtype=np.int32)], True, (255, 255, 0), 1)
            else:
                prev_point = None
                if np.all(np.isnan(obj["points"])) or np.all(np.isnan(obj["points"])):
                    return image
                for point in obj["points"]:
                    if prev_point is not None:
                        cv2.line(image, prev_point, tuple(point), (0, 255, 255), 1)
                    prev_point = tuple(point)

        return image

    def load_json(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data


if __name__ == '__main__':
    root_path = ImagesFolder
    draw_road_obj = DrawRoadObjects(root_path)
    draw_road_obj.process()