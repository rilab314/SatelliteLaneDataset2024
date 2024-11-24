import os
import cv2
import numpy as np
from pyproj import Transformer
from typing import List, Tuple


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


def extract_coordinates_from_filenames(folder_path: str) -> List[Tuple[float, float]]:
    coordinates = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            try:
                lon, lat = map(float, filename.replace(".png", "").split(","))
                coordinates.append((lon, lat))
            except ValueError:
                print(f"Skipping invalid filename: {filename}")
    return coordinates


def draw_transparent_points(
        image: np.ndarray, coordinates: List[Tuple[float, float]], lon_min: float, lon_max: float, lat_min: float,
        lat_max: float, color: Tuple[int, int, int] = (255, 191, 0), radius: int = 5, alpha: float = 0.65
):
    overlay = image.copy()
    img_height, img_width = image.shape[:2]
    for lon, lat in coordinates:
        pixel_x, pixel_y = lonlat_to_pixel(lon, lat, lon_min, lon_max, lat_min, lat_max, img_width, img_height)
        cv2.circle(overlay, (pixel_x, pixel_y), radius, color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def upscale_image(image: np.ndarray, scale: int) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)


def downscale_image(image: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, original_size, interpolation=cv2.INTER_AREA)


def process_image(
        images_path: str, background_image_path: str, output_path: str, lon_min: float, lon_max: float,
        lat_min: float, lat_max: float, upscale_factor: int = 20, point_color: Tuple[int, int, int] = (0, 255, 255),
        point_radius: int = 5, transparency: float = 0.3
):
    image = cv2.imread(background_image_path)
    if image is None:
        print(f"Failed to load image: {background_image_path}")
        return

    coordinates = extract_coordinates_from_filenames(images_path)
    original_size = image.shape[1], image.shape[0]

    # Upscale the image
    image_upscaled = upscale_image(image, upscale_factor)

    # Draw points on the upscaled image
    draw_transparent_points(
        image_upscaled, coordinates, lon_min, lon_max, lat_min, lat_max,
        color=point_color, radius=point_radius * upscale_factor, alpha=transparency
    )

    # Downscale the image back to the original size
    image_final = downscale_image(image_upscaled, original_size)

    # Save the output
    cv2.imwrite(output_path, image_final)
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    root_dir = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/figure/figure2"
    images_path = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/satellite_good_matching_241122/image"

    process_image(
        images_path=images_path,
        background_image_path=os.path.join(root_dir, "seoul.png"),
        output_path=os.path.join(root_dir, "output_seoul.png"),
        lon_min=126.8255, lon_max=127.1516,
        lat_min=37.4738, lat_max=37.6202
    )

    process_image(
        images_path=images_path,
        background_image_path=os.path.join(root_dir, "incheon.png"),
        output_path=os.path.join(root_dir, "output_incheon.png"),
        lon_min=126.5987, lon_max=126.8068,
        lat_min=37.3648, lat_max=37.5910
    )
