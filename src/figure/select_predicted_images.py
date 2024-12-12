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
            # img.save(output_path)


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


def detection_predicted_images():
    # Define base path and folders
    base_path = "/home/falcon/Downloads/241210_detection"  # Adjust if script is not in the same directory
    folders = ["deformable_detr", "dino", "yolox", "grounding_dino"]

    # Coordinate list
    coordinates = [
        "126.7160496,37.4045113",
        "126.7160496,37.4203959",
        "126.7160496,37.4389278",
        "126.7171567,37.4380454",
        "127.0463,37.574647999999996",
        "127.05918000000001,37.498248",
        "127.07114,37.541031999999994",
        "127.0647,37.594511999999995",  # crossroads

        # safety_zone
        "126.7226921,37.5757113",
        "126.7282274,37.4018639",
        "127.05642,37.551728",
        "126.7226921,37.5165856",
        "127.05826,37.475328",
        "126.721585,37.5165856"

    ]

    # Process the folders
    process_folders(base_path, folders, coordinates)
    print("Processing completed!")

def segmentation_predicted_images():
    # Define base path and folders
    base_path = "/home/humpback/Downloads/241211_segmentation"  # Adjust if script is not in the same directory
    folders = ["mask2former", "segformer", "upernet", "pspnet"]

    # Coordinate list
    coordinates = [
        "test_126.721585,37.5165856.png_131",
        "test_126.721585,37.4865814.png_123",
        "test_126.721585,37.5757113.png_137",
        "test_126.7160496,37.4027464.png_3",
        "test_126.7171567,37.4389278.png_33",
        "test_126.7293345,37.4909938.png_311",
        "test_126.7304416,37.4053938.png_335",
        "test_126.7304416,37.4389278.png_339",
        "test_126.7315486,37.4398103.png_367",
        "test_127.0463,37.537212.png_418",
        "test_127.0647,37.505123999999995.png_944",
        "test_127.04722000000001,37.576176.png_443",
        "test_127.05550000000001,37.566244.png_668",
        "test_127.05366000000001,37.495956.png_609",
        "test_127.05734000000001,37.57694.png_744"

    ]

    # Process the folders
    process_folders(base_path, folders, coordinates)
    print("Processing completed!")


if __name__ == '__main__':
    # detection_predicted_images()
    segmentation_predicted_images()

