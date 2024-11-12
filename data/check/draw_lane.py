import cv2
import numpy as np
import json
import os


def draw_objects(image, data):
    image = image.copy()
    for obj in data:
        if obj["type_id"] in ["1", "5"]:
            if len(obj["image_points"]) > 1:
                cv2.polylines(image, [np.array(obj["image_points"], dtype=np.int32)], True, (255, 255, 0), 1)
        else:
            prev_point = None
            if np.all(np.isnan(obj["image_points"])) or np.all(np.isnan(obj["image_points"])):
                return image
            for point in obj["image_points"]:
                if prev_point is not None:
                    cv2.line(image, prev_point, tuple(point), (0, 255, 255), 1)
                prev_point = tuple(point)

    return image

if __name__ == '__main__':
    img_path = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/archive/국토정보플랫폼/unused_data/convert_for_train/국토위성이미지_크롤러_240930_완/test_image_label/image/00002_14118026.832884334,4507482.642017038,14118273.3263203,4507730.258118873.png"
    label_path = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/archive/국토정보플랫폼/unused_data/convert_for_train/국토위성이미지_크롤러_240930_완/test_image_label/label/00002_126.8255,37.489844.json"
    with open(label_path, "r") as f:
        data = json.load(f)
    data = data[1:]
    image = cv2.imread(img_path)
    drawn_image = draw_objects(image, data)

    cv2.imshow("", drawn_image)
    cv2.waitKey(0)