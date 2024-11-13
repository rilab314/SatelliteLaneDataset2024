import os
import cv2
import json
import numpy as np
from glob import glob

def convert_src_to_detectron(src_dir, output_dir, output_name):
    src_label_paths = sorted(glob(os.path.join(src_dir, '*.json')))
    for src_label_path in src_label_paths:
        print(src_label_path)
        semantic_image = generate_semantic_image(src_label_path)
        cv2.imwrite(src_label_path.replace('segmentation_label', 'detectron_label').replace('.json', '.png'), semantic_image)
def generate_semantic_image(src_label_path):
    semantic_image = np.zeros((768, 768), dtype=np.uint8)
    with open(src_label_path, 'r') as f:
        data = json.load(f)
    for road_object in data[1:]:
        points = road_object['image_points']
        polygon_points = np.array(points, dtype=np.int32)
        cv2.fillPoly(semantic_image, [polygon_points], 1)

    return semantic_image

if __name__ == '__main__':
    src_path = '/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/satellite_segmentation_241113/segmentation_label'
    convert_src_to_detectron(src_path, 1, 1)