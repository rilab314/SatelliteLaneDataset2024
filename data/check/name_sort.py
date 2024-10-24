import os
import re
import shutil
from glob import glob
from matcher.config.config import TotalDataset

image_list = glob(os.path.join(TotalDataset, "origin_image", "*.png"))
label_list = glob(os.path.join(TotalDataset, "label", "*.json"))


image_list = sorted(image_list, key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
label_list = sorted(label_list, key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))

for i, (image_path, label_path) in enumerate(zip(image_list, label_list)):
    image_filename = os.path.basename(image_path)
    os.rename(image_path, image_path.replace(image_filename, f"{str(i+1).zfill(7)}.png"))
    label_filename = os.path.basename(label_path)
    os.rename(label_path, label_path.replace(label_filename, f"{str(i+1).zfill(7)}.json"))
    print(f"{image_filename} -> {str(i+1).zfill(7)}.png")
    p=1
p=1