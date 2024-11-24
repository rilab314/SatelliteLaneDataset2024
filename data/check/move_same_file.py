import shutil
from glob import glob

image_path = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/total_origin_lonlat/missing_images/drawn_image"

image_list = glob(image_path + "/*.png")
image_list.sort()
for image_path in image_list:
    shutil.copy(image_path.replace("drawn_image", "image"),
                image_path.replace("drawn_image", "good_image"))

    shutil.copy(image_path.replace("drawn_image", "label").replace(".png", ".json"),
                image_path.replace("drawn_image", "good_label").replace(".png", ".json"))