import cv2
import numpy as np
import json
import os
from glob import glob


def draw_objects(image, data):
    image = image.copy()
    for obj in data:
        #
        # if obj["category"] == "center_line":
        prev_point = None
        if obj["geometry_type"] == "LINE_STRING":
            if np.all(np.isnan(obj["image_points"])) or np.all(np.isnan(obj["image_points"])):
                return image
            for point in obj["image_points"]:
                if prev_point is not None:
                    cv2.line(image, prev_point, tuple(point), (0, 255, 255), 1)
                prev_point = tuple(point)
        elif obj["geometry_type"] == "MULTILINE_STRING":
            for line in obj["image_points"]:
                if np.all(np.isnan(line)) or np.all(np.isnan(line)):
                    return image
                for point in line:
                    if prev_point is not None:
                        cv2.line(image, prev_point, tuple(point), (0, 255, 255), 1)
                    prev_point = tuple(point)
        elif obj["geometry_type"] == "POLYGON":
            if len(obj["image_points"]) > 1:
                cv2.polylines(image, [np.array(obj["image_points"], dtype=np.int32)], True, (255, 255, 0), 1)
                # cv2.fillPoly(image, [np.array(obj["image_points"], dtype=np.int32)], color=(255, 255, 0))
        elif obj["geometry_type"] == "MULTIPOLYGON":
            for polygon in obj["image_points"]:
                if len(polygon) > 1:
                    # cv2.polylines(image, [np.array(polygon, dtype=np.int32)], True, (255, 255, 0), 1)
                    cv2.fillPoly(image, [np.array(polygon, dtype=np.int32)], color=(255, 255, 0))

    return image

if __name__ == '__main__':
    # img_paths = sorted(glob(os.path.join("/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/satellite_dataset_241111/image", "*.png")))
    label_paths = sorted(glob(os.path.join("/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/incheon_241111/label", "*.json")))
    # label_paths = sorted(glob(os.path.join("/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/incheon_241111/segmentation_label", "*.json")))
    label_paths = sorted(glob(os.path.join("/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/total_origin_lonlat/label", "*.json")))

    for label_path in label_paths:
        img_path = label_path.replace("label", "image").replace(".json", ".png")
        with open(label_path, "r") as f:
            data = json.load(f)
        data = data[1:]

        image = cv2.imread(img_path)
        drawn_image = draw_objects(image, data)
        # cv2.imshow("image", image)
        # cv2.imshow(f"drawn_image", drawn_image)
        # cv2.waitKey(0)
        cv2.imwrite(img_path.replace("image", "drawn_image"), drawn_image)