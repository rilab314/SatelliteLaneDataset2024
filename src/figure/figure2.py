import os
import cv2
import numpy as np
import json
from pyproj import Transformer
from typing import List, Tuple, Dict


def lonlat_to_pixel(
        lon: float, lat: float, lon_min: float, lon_max: float, lat_min: float, lat_max: float, img_width: int,
        img_height: int
) -> Tuple[int, int]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = transformer.transform(lon, lat)
    x_min, y_min = transformer.transform(lon_min, lat_min)
    x_max, y_max = transformer.transform(lon_max, lat_max)
    pixel_x = int((x - x_min) / (x_max - x_min) * img_width)
    pixel_y = int((y_max - y) / (y_max - y_min) * img_height)
    return pixel_x, pixel_y


def draw_dataset_points(
        image: np.ndarray, dataset: Dict[str, List[str]], lon_min: float, lon_max: float, lat_min: float, lat_max: float,
        colors: Dict[str, Tuple[int, int, int]], radius: int, alpha: float
):
    overlay = image.copy()
    img_height, img_width = image.shape[:2]

    for split, coordinates in dataset.items():
        color = colors[split]
        for coord in coordinates:
            lon, lat = map(float, coord.split(","))
            pixel_x, pixel_y = lonlat_to_pixel(lon, lat, lon_min, lon_max, lat_min, lat_max, img_width, img_height)
            cv2.circle(overlay, (pixel_x, pixel_y), radius, color, -1)

    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def crop_and_draw_dataset(
        image: np.ndarray, dataset: Dict[str, List[str]], lon_min: float, lon_max: float, lat_min: float, lat_max: float,
        crop_box: Tuple[int, int, int, int], upscale_factor: int, colors: Dict[str, Tuple[int, int, int]],
        radius: int, alpha: float
) -> np.ndarray:
    x1, y1, x2, y2 = crop_box
    cropped_image = image[y1:y2, x1:x2].copy()

    cropped_upscaled = upscale_image(cropped_image, upscale_factor)

    draw_dataset_points(
        cropped_upscaled, dataset, lon_min, lon_max, lat_min, lat_max,
        colors, radius * upscale_factor, alpha
    )

    cropped_with_points = downscale_image(cropped_upscaled, (x2 - x1, y2 - y1))

    image[y1:y2, x1:x2] = cropped_with_points
    return image


def upscale_image(image: np.ndarray, scale: int) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)


def downscale_image(image: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, original_size, interpolation=cv2.INTER_AREA)


def load_dataset(dataset_path: str) -> Dict[str, List[str]]:
    with open(dataset_path, "r") as f:
        return json.load(f)


def process_image_with_dataset(
        dataset_path: str, background_image_path: str, output_path: str, lon_min: float, lon_max: float,
        lat_min: float, lat_max: float, crop_box: Tuple[int, int, int, int], upscale_factor: int = 20,
        colors: Dict[str, Tuple[int, int, int]] = None, radius: int = 5, alpha: float = 0.5
):
    if colors is None:
        colors = {
            "train": (0, 255, 0),  # Green
            "validation": (255, 255, 0),  # Yellow
            "test": (0, 0, 255)  # Red
        }

    image = cv2.imread(background_image_path)
    if image is None:
        print(f"Failed to load image: {background_image_path}")
        return

    dataset = load_dataset(dataset_path)

    image_with_points = crop_and_draw_dataset(
        image, dataset, lon_min, lon_max, lat_min, lat_max,
        crop_box, upscale_factor, colors, radius, alpha
    )

    cv2.imwrite(output_path, image_with_points)
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    root_dir = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/figure/figure2"
    dataset_path = os.path.join(root_dir, "dataset.json")

    colors = {
        "train": (0, 255, 0),  # Green
        "validation": (255, 255, 0),  # Yellow
        "test": (0, 0, 255)  # Red
    }

    process_image_with_dataset(
        dataset_path=dataset_path,
        background_image_path=os.path.join(root_dir, "seoul_nobox.png"),
        output_path=os.path.join(root_dir, "output_seoul_dataset.png"),
        lon_min=126.8255, lon_max=127.1516,
        lat_min=37.4738, lat_max=37.6202,
        crop_box=(556, 144, 2452, 1214),
        upscale_factor=20,
        colors=colors,
        radius=3,
        alpha=0.5
    )

    process_image_with_dataset(
        dataset_path=dataset_path,
        background_image_path=os.path.join(root_dir, "incheon_nobox.png"),
        output_path=os.path.join(root_dir, "output_incheon_dataset.png"),
        lon_min=126.5987, lon_max=126.8068,
        lat_min=37.3648, lat_max=37.5910,
        crop_box=(134, 858, 1340, 2514),
        upscale_factor=20,
        colors=colors,
        radius=5,
        alpha=0.5
    )
