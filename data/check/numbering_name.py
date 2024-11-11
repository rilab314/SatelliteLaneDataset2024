import os
import shutil
from glob import glob

root_path = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/archive/국토정보플랫폼/unused_data/convert_for_train/인천_위성이미지"
images_list = sorted(glob(os.path.join(root_path, "image", "*.png")))
labels_list = sorted(glob(os.path.join(root_path, "label", "*.json")))

for i, (image_path, label_path) in enumerate(zip(images_list, labels_list)):
    re_image_path = image_path.replace("image", "image_numbering").replace(os.path.basename(image_path), f"{str(i).zfill(7)}.png")
    re_label_path = label_path.replace("label", "label_numbering").replace(os.path.basename(label_path), f"{str(i).zfill(7)}.json")
    shutil.copy(image_path, re_image_path)
    shutil.copy(label_path, re_label_path)