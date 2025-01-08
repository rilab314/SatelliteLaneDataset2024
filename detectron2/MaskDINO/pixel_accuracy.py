import os
import numpy as np
from PIL import Image


def calculate_pixel_accuracy_ignore_mask(validation_folder, image_folder, mask_value=1):
    """
    Calculate pixel accuracy between images in the validation folder and the image folder,
    ignoring pixels where the validation image has the specified mask value.

    Args:
        validation_folder (str): Path to the folder containing validation images (ground truth).
        image_folder (str): Path to the folder containing predicted images.
        mask_value (int): The pixel value in validation images to be ignored in the calculation.

    Returns:
        float: Pixel accuracy ignoring masked values.
    """
    validation_files = sorted(os.listdir(validation_folder))
    image_files = sorted(os.listdir(image_folder))

    if len(validation_files) != len(image_files):
        raise ValueError("The number of images in both folders must be the same.")

    total_pixels = 0
    correct_pixels = 0

    for val_file, img_file in zip(validation_files, image_files):
        if val_file != img_file:
            raise ValueError
        val_path = os.path.join(validation_folder, val_file)
        img_path = os.path.join(image_folder, img_file)

        # Open images
        with Image.open(val_path) as val_img, Image.open(img_path) as img:
            # Convert images to numpy arrays
            val_array = np.array(val_img)
            img_array = np.array(img)

            # Check dimensions
            if val_array.shape != img_array.shape:
                raise ValueError(f"Image shapes do not match: {val_file} vs {img_file}")

            # Create a mask of valid pixels (where validation image is not equal to mask_value)
            valid_mask = val_array != mask_value

            # Count valid pixels
            total_pixels += np.sum(valid_mask)

            # Count correct pixels among valid pixels
            correct_pixels += np.sum((val_array == img_array) & valid_mask)

    # Avoid division by zero
    if total_pixels == 0:
        raise ValueError("No valid pixels to compare. Check your mask value and input images.")

    # Calculate pixel accuracy
    pixel_accuracy = correct_pixels / total_pixels
    return pixel_accuracy


if __name__ == "__main__":
    # Define folder paths
    validation_folder = "/home/humpback/shin_workspace/projects/MaskDINO/datasets/satellite_ade20k_241213/annotations/validation"
    image_folder = "/home/humpback/shin_workspace/projects/MaskDINO/inference_result/pred_mask"

    # Calculate pixel accuracy ignoring mask value
    accuracy = calculate_pixel_accuracy_ignore_mask(validation_folder, image_folder, mask_value=1)
    print(f"Pixel Accuracy (ignoring mask): {accuracy:.4f}")
