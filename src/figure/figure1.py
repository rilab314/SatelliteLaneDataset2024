import os
import json
import cv2
import numpy as np
from glob import glob

from src.figure.category_colormap import KindDictColors, KindDictColors_pastel

def draw_lane_on_image(image_path, label_path, output_path, scale=4, thickness=2, radius=2):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return

    height, width = image.shape[:2]
    high_res_image = cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_LINEAR)

    with open(label_path, 'r') as f:
        label_data = json.load(f)

    for obj in label_data:
        if obj['class'] == 'RoadObject':
            color = KindDictColors[obj['category_id']]
            # color = (color[2], color[1], color[0])
            scaled_points = [(int(x * scale), int(y * scale)) for x, y in obj['image_points']]

            if obj['geometry_type'] == 'LINE_STRING':
                draw_line_string(high_res_image, scaled_points, color, thickness)
                draw_points(high_res_image, scaled_points, color, radius, step=2)
            elif obj['geometry_type'] == 'POLYGON':
                draw_polygon(high_res_image, scaled_points, color, int(thickness/1.5))

    final_image = cv2.resize(high_res_image, (width*2, height*2), interpolation=cv2.INTER_AREA)

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(output_file, final_image)
    # cv2.imshow('image', final_image)
    # cv2.waitKey(0)
    print(f"Saved: {output_file}")


def draw_line_string(image, points, color=(0, 255, 0), thickness=2):
    points = np.array(points, dtype=np.int32)
    cv2.polylines(image, [points], isClosed=False, color=color, thickness=thickness)


def draw_polygon(image, points, color=(255, 0, 0), thickness=2):
    points = np.array(points, dtype=np.int32)
    # cv2.fillPoly(image, [points], color=color)
    cv2.polylines(image, [points], True, color, thickness)

def draw_points(image, points, color=(0, 0, 255), radius=3, step=2):
    cv2.circle(image, (int(points[0][0]), int(points[0][1])), radius, color, -1)
    for i in range(step, len(points) - 1, step):
        point = points[i]
        cv2.circle(image, (int(point[0]), int(point[1])), radius, color, -1)
    if len(points) > 1:  # Ensure there is more than one point
        cv2.circle(image, (int(points[-1][0]), int(points[-1][1])), radius, color, -1)


def process_all_images(image_folder, label_folder, output_folder, scale=4, thickness=1, radius=2.):
    image_files = glob(os.path.join(image_folder, '*.png'))

    for image_file in image_files:
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        label_file = os.path.join(label_folder, f"{base_name}.json")

        if not os.path.exists(label_file):
            print(f"Label not found for image: {image_file}")
            continue

        draw_lane_on_image(image_file, label_file, output_folder, scale, thickness * scale, int(radius * scale))


if __name__ == '__main__':
    image_folder = '/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/figure/image'
    label_folder = '/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/figure/label'
    output_folder = '/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/figure/figure1/output'

    process_all_images(image_folder, label_folder, output_folder, scale=100, thickness=1, radius=2.4)
