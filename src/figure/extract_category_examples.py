import os
import json
import cv2
import numpy as np
from glob import glob
from typing import List, Dict, Tuple

from src.config.config import KindDict
from src.figure.category_colormap import KindDictColors


def upscale_image(image: np.ndarray, scale: int) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)


def downscale_image(image: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, original_size, interpolation=cv2.INTER_AREA)


def crop_image(image: np.ndarray, bounding_box: Tuple[int, int, int, int], padding: int = 50) -> np.ndarray:
    """
    Crops an image based on the given bounding box and adds padding.

    Args:
        image (np.ndarray): The input image.
        bounding_box (Tuple[int, int, int, int]): The bounding box (x_min, y_min, x_max, y_max).
        padding (int): The number of pixels to add as padding.

    Returns:
        np.ndarray: The cropped image.
    """
    height, width = image.shape[:2]
    x_min, y_min, x_max, y_max = bounding_box
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)
    return image[y_min:y_max, x_min:x_max]


def calculate_bounding_box(points: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Calculate the bounding box of a set of points.

    Args:
        points (np.ndarray): The points to calculate the bounding box for.

    Returns:
        Tuple[int, int, int, int]: The bounding box (x_min, y_min, x_max, y_max).
    """
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)


def extract_files_for_category(image_folder: str, label_folder: str, output_folder: str, category_id: str,
                               max_images: int = 10, upscale_factor: int = 4, output_scale_factor: int = 2, padding: int = 50):
    os.makedirs(output_folder, exist_ok=True)

    image_files = glob(os.path.join(image_folder, "*.png"))
    extracted_count = 0

    for image_file in image_files:
        if extracted_count >= max_images:
            break

        base_name = os.path.splitext(os.path.basename(image_file))[0]
        label_file = os.path.join(label_folder, f"{base_name}.json")
        if not os.path.exists(label_file):
            continue

        with open(label_file, 'r') as f:
            label_data = json.load(f)

        category_present = any(
            obj['class'] == 'RoadObject' and obj['category_id'] == category_id for obj in label_data
        )
        if not category_present:
            continue

        image = cv2.imread(image_file)
        if image is None:
            continue

        original_size = image.shape[1] * output_scale_factor, image.shape[0] * output_scale_factor
        image_upscaled = upscale_image(image, upscale_factor)

        overlay_image_upscaled, bounding_box = overlay_category_objects(image_upscaled, label_data, category_id, upscale_factor)
        # cropped_image = crop_image(overlay_image_upscaled, bounding_box, padding)

        overlay_image = downscale_image(overlay_image_upscaled, original_size)

        combined_image = attach_original_image(image, overlay_image)

        output_file = os.path.join(output_folder, f"{base_name}_{KindDict[category_id]}.png")
        cv2.imwrite(output_file, combined_image)
        extracted_count += 1
        print(f"Extracted {extracted_count}/{max_images} for category: {KindDict[category_id]}")

def attach_original_image(original_image: np.ndarray, modified_image: np.ndarray) -> np.ndarray:
    """
    Attach the original image to the left of the modified image.

    Args:
        original_image (np.ndarray): The original image.
        modified_image (np.ndarray): The image with modifications.

    Returns:
        np.ndarray: Combined image with the original on the left and modified on the right.
    """
    height_orig, width_orig = original_image.shape[:2]
    height_mod, width_mod = modified_image.shape[:2]

    # Resize the original image if needed to match heights
    if height_orig != height_mod:
        original_image = cv2.resize(original_image, (width_mod, height_mod), interpolation=cv2.INTER_AREA)

    # Concatenate the original and modified images
    combined_image = np.concatenate((original_image, modified_image), axis=1)
    return combined_image



def overlay_category_objects(image: np.ndarray, label_data: List[Dict], category_id: str, scale: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    overlay = image.copy()
    category_points = []

    for obj in label_data:
        if obj['class'] != 'RoadObject' or obj['category_id'] != category_id:
            continue

        points = np.array(obj['image_points'], dtype=np.int32) * scale
        category_points.append(points)

        color = KindDictColors[obj['category_id']]
        # color = (color[2], color[1], color[0])
        if obj['geometry_type'] == 'LINE_STRING':
            cv2.polylines(overlay, [points], isClosed=False, color=color, thickness=1 * scale)
            draw_points(overlay, points, color, radius=int(1.5 * scale), step=2)
        elif obj['geometry_type'] == 'POLYGON':
            cv2.polylines(overlay, [points], isClosed=True, color=color, thickness=1 * scale)

    if category_points:
        all_points = np.concatenate(category_points, axis=0)
        bounding_box = calculate_bounding_box(all_points)
    else:
        bounding_box = (0, 0, image.shape[1], image.shape[0])

    return overlay, bounding_box


def draw_points(image, points, color=(0, 0, 255), radius=3, step=2):
    cv2.circle(image, (int(points[0][0]), int(points[0][1])), radius, color, -1)
    for i in range(step, len(points) - 1, step):
        point = points[i]
        cv2.circle(image, (int(point[0]), int(point[1])), radius, color, -1)
    if len(points) > 1:
        cv2.circle(image, (int(points[-1][0]), int(points[-1][1])), radius, color, -1)


def process_categories(image_folder: str, label_folder: str, output_folder: str, max_images_per_category: int = 10,
                       upscale_factor: int = 4, output_scale_factor: int = 2, padding: int = 50):
    for category_id in KindDict.keys():
        if category_id is None:
            continue
        category_output_folder = os.path.join(output_folder, KindDict[category_id])
        extract_files_for_category(image_folder, label_folder, category_output_folder, category_id,
                                   max_images_per_category, upscale_factor, output_scale_factor, padding)


if __name__ == "__main__":
    dataset_root = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/satellite_good_matching_241122"
    image_folder = os.path.join(dataset_root, "image")
    label_folder = os.path.join(dataset_root, "label")

    output_folder = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/figure/ex_categories"

    process_categories(image_folder, label_folder, output_folder, max_images_per_category=40, upscale_factor=10,
                       output_scale_factor=2, padding=100)
