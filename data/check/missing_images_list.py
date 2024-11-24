import os
import json
from typing import List, Tuple


def load_coordinates_from_json(file_path: str) -> List[Tuple[float, float]]:
    """
    Load coordinates from a JSON file.
    :param file_path: Path to the JSON file.
    :return: List of coordinates as tuples (lon, lat).
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [(item[0], item[1]) for item in data]


def extract_filenames_as_coordinates(folder_path: str) -> List[Tuple[float, float]]:
    """
    Extract coordinates from filenames in the specified folder.
    Filenames should follow the format 'lon,lat.png'.
    :param folder_path: Path to the folder containing image files.
    :return: List of coordinates as tuples (lon, lat).
    """
    coordinates = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            try:
                lon, lat = map(float, filename.replace(".png", "").split(","))
                coordinates.append((lon, lat))
            except ValueError:
                print(f"Skipping invalid filename: {filename}")
    return coordinates


def find_missing_coordinates(
        json_coordinates: List[Tuple[float, float]], folder_coordinates: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """
    Identify coordinates that are in the JSON file but not in the folder.
    :param json_coordinates: List of coordinates from the JSON file.
    :param folder_coordinates: List of coordinates extracted from the folder filenames.
    :return: List of missing coordinates.
    """
    return [coord for coord in json_coordinates if coord not in folder_coordinates]


def save_coordinates_to_json(file_path: str, coordinates: List[Tuple[float, float]]):
    """
    Save a list of coordinates to a JSON file.
    :param file_path: Path to save the JSON file.
    :param coordinates: List of coordinates to save.
    """
    with open(file_path, 'w') as f:
        json.dump(coordinates, f)
    print(f"Missing coordinates saved to: {file_path}")


def main():
    # Paths
    json_file_path = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/archive/국토정보플랫폼/400x400array_data.json"
    image_folder_path = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/total_origin_lonlat/image"
    output_json_path = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/archive/국토정보플랫폼/missing_coordinates_array_data.json"

    json_coordinates = load_coordinates_from_json(json_file_path)

    folder_coordinates = extract_filenames_as_coordinates(image_folder_path)

    missing_coordinates = find_missing_coordinates(json_coordinates, folder_coordinates)
    p=1
    # save_coordinates_to_json(output_json_path, missing_coordinates)


if __name__ == "__main__":
    main()
