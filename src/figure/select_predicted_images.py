import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from PIL import Image



def detection_predicted_images():
    # Define base path and folders
    base_path = "/home/humpback/Downloads/241223_detection"  # Adjust if script is not in the same directory
    folders = ["deformable-detr", "dino", "yolox", "grounding-dino"]

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
        "126.721585,37.5165856",

        "127.05918000000001,37.476091999999994",
        "127.07114,37.561659999999996",

    ]

    # Process the folders
    # process_folders(base_path, folders, coordinates)
    # process_cut_images(base_path, folders, coordinates)

    cut_folders = ["GT", "deformable-detr_cut", "grounding-dino_cut", "dino_cut", "yolox_cut"]
    output_folder = os.path.join(base_path, "all_images_merged")
    folder_paths = [os.path.join(base_path, folder) for folder in cut_folders]
    merge_images_with_labels(folder_paths, output_folder)

    print("Processing completed!")

def segmentation_predicted_images():
    # Define base path and folders
    base_path = "/home/humpback/Downloads/241223_segmentation"  # Adjust if script is not in the same directory
    folders = ["mask2former", "segformer", "upernet"]

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
    # process_folders(base_path, folders, coordinates)
    # process_cut_images(base_path, folders, coordinates)

    cut_folders = ["image", "GT", "upernet_cut", "segformer_cut", "maskdino_cut", "mask2former_cut"]
    output_folder = os.path.join(base_path, "merged_output")
    folder_paths = [os.path.join(base_path, folder) for folder in cut_folders]
    merge_images_with_labels(folder_paths, output_folder)
    print("Processing completed!")


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

def process_cut_images(base_path, folders, coordinates):
    """
    Process images in each folder and save the cropped versions to respective output folders and GT folder.
    """
    # Create output folders
    output_folders, gt_folder = create_output_folders(base_path, folders)

    # Process each folder
    for folder in tqdm(folders):
        folder_path = os.path.join(base_path, folder)
        output_folder = output_folders[folder]

        # Get the image files that match the coordinates
        image_paths = glob(os.path.join(folder_path, "*.png"))

        # Crop and save images
        crop_and_save_images(image_paths, output_folder, gt_folder)



def add_label_to_image(image, label):
    """
    Add a label box with the folder name at the top-left corner of the image.
    """
    font_scale = 1
    font_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get text size
    text_size = cv2.getTextSize(label.replace("_cut", ""), font, font_scale, font_thickness)[0]
    text_width, text_height = text_size

    # Define text box coordinates
    margin = 10
    box_coords = ((0, 0), (text_width + 2 * margin, text_height + 2 * margin))

    # Draw a filled rectangle for the label background
    cv2.rectangle(image, box_coords[0], box_coords[1], (50, 50, 50), -1)

    # Add the text label
    text_position = (margin, text_height + margin)
    cv2.putText(image, label.replace("_cut", ""), text_position, font, font_scale, (255, 255, 255), font_thickness)
    return image


def merge_images_with_labels(folder_paths, output_folder):
    """
    Add labels to images from specified folders and merge them horizontally.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Get list of image files in the first folder (GT) as reference
    reference_folder = folder_paths[0]
    image_files = sorted(os.listdir(reference_folder))

    for file_name in image_files:
        images = []
        for folder_path in folder_paths:
            folder_name = os.path.basename(folder_path)
            image_path = glob(os.path.join(folder_path, f"*{file_name}*"))[0]
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                if img is not None:
                    labeled_img = add_label_to_image(img, folder_name)
                    images.append(labeled_img)

            else:
                print(f"Image {file_name} not found in {folder_path}, skipping.")

        # Merge all images horizontally
        if images:
            merged_img = np.hstack(images)

            # Save the merged image
            output_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_path, merged_img)
    print(f"Merged images saved in {output_folder}")

if __name__ == '__main__':
    detection_predicted_images()
    # segmentation_predicted_images()

