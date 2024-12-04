import os
from PIL import Image


def create_output_folders(base_path, folders):
    """
    Create output folders for each input folder and a shared GT folder.
    """
    output_folders = {folder: os.path.join(base_path, f"{folder}_cut") for folder in folders}
    gt_folder = os.path.join(base_path, "GT")
    os.makedirs(gt_folder, exist_ok=True)
    for output_folder in output_folders.values():
        os.makedirs(output_folder, exist_ok=True)
    return output_folders, gt_folder


def get_image_files(folder_path, coordinates):
    """
    Get a list of image file paths that match the given coordinates.
    """
    files = []
    for coord in coordinates:
        file_path = os.path.join(folder_path, f"{coord}.png")
        if os.path.exists(file_path):
            files.append(file_path)
    return files


def crop_and_save_images(image_paths, output_folder, gt_folder):
    """
    Crop the images into left and right halves and save them to respective folders.
    """
    for image_path in image_paths:
        with Image.open(image_path) as img:
            width, height = img.size
            # Crop left and right halves
            left_half = img.crop((0, 0, width // 2, height))
            right_half = img.crop((width // 2, 0, width, height))

            # Save left half to GT folder
            file_name = os.path.basename(image_path)
            gt_path = os.path.join(gt_folder, file_name)
            left_half.save(gt_path)

            # Save right half to the specific output folder
            output_path = os.path.join(output_folder, file_name)
            right_half.save(output_path)


def process_folders(base_path, folders, coordinates):
    """
    Process images in each folder and save the cropped versions to respective output folders and GT folder.
    """
    # Create output folders
    output_folders, gt_folder = create_output_folders(base_path, folders)

    # Process each folder
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        output_folder = output_folders[folder]

        # Get the image files that match the coordinates
        image_paths = get_image_files(folder_path, coordinates)

        # Crop and save images
        crop_and_save_images(image_paths, output_folder, gt_folder)


if __name__ == "__main__":
    # Define base path and folders
    base_path = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/figure/predict"  # Adjust if script is not in the same directory
    folders = ["deformable_detr", "dino", "yolox-l"]

    # Coordinate list
    coordinates = [
        "126.7160496,37.4045113",
        "126.7160496,37.4203959",
        "126.7160496,37.4389278",
        "126.7171567,37.4380454",
        "127.0463,37.574647999999996",
        "127.05918000000001,37.498248",
        "127.07114,37.541031999999994"
    ]

    # Process the folders
    process_folders(base_path, folders, coordinates)
    print("Processing completed!")


