import os
import cv2
import glob
import shutil

# Paths for folders
folder_path = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/total_origin_lonlat/missing_images/drawn_image"
destination_folder = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/total_origin_lonlat/missing_images/false_image"
break_image_folder = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/total_origin_lonlat/missing_images/break_image"

# Create folders if they don't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
if not os.path.exists(break_image_folder):
    os.makedirs(break_image_folder)

# Load only .png files
images = sorted(glob.glob(os.path.join(folder_path, "*.png")))
current_index = 0

if not images:
    print("No .png images found in the folder.")
    exit()

while True:
    current_image_path = images[current_index]

    image = cv2.imread(current_image_path)
    if image is None:
        print(f"Unable to load image {current_image_path}.")
        break

    scale = 1.2
    resized_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    print(f"Current image: {os.path.basename(images[current_index])} ({current_index + 1} of {len(images)})")
    cv2.imshow("Image Viewer", resized_image)

    # Wait for key input
    key = cv2.waitKey(0)

    if key == ord('q'):  # Move image to destination folder
        destination_path = os.path.join(destination_folder, os.path.basename(current_image_path))
        print(f"Moving image {os.path.basename(images[current_index])} to {destination_folder}")
        shutil.move(current_image_path, destination_path)
        del images[current_index]

        if not images:  # Exit if all images are moved
            print("All images have been moved.")
            break

    elif key == ord('w'):  # Move image to break folder
        break_image_path = os.path.join(break_image_folder, os.path.basename(current_image_path))
        print(f"Moving image {os.path.basename(images[current_index])} to {break_image_folder}")
        shutil.move(current_image_path, break_image_path)
        del images[current_index]

        if not images:  # Exit if all images are moved
            print("All images have been moved.")
            break
    elif key == ord('d'):  # Go to next image
        current_index = (current_index + 1) % len(images)
    elif key == ord('a'):  # Go to previous image
        current_index = (current_index - 1) % len(images)
    elif key == 27:  # Exit on ESC key
        print("Program terminated.")
        break

cv2.destroyAllWindows()
