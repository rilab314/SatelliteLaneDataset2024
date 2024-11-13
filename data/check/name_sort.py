import os
import re
import shutil
from glob import glob
from matcher.config.config import TotalDataset

# image_list = glob(os.path.join(TotalDataset, "origin_image", "*.png"))
# label_list = glob(os.path.join(TotalDataset, "label", "*.json"))
#
#
# image_list = sorted(image_list, key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
# label_list = sorted(label_list, key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
#
# for i, (image_path, label_path) in enumerate(zip(image_list, label_list)):
#     image_filename = os.path.basename(image_path)
#     os.rename(image_path, image_path.replace(image_filename, f"{str(i+1).zfill(7)}.png"))
#     label_filename = os.path.basename(label_path)
#     os.rename(label_path, label_path.replace(label_filename, f"{str(i+1).zfill(7)}.json"))
#     print(f"{image_filename} -> {str(i+1).zfill(7)}.png")
#     p=1
# p=1

import os
from glob import glob
# with open("/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/archive/국토정보플랫폼/incheon_array_data.json", "r") as f:
#     coordinates = json.load(f)
#
# rounded_coordinates = [[round(lon, 7), round(lat, 7)] for lon, lat in coordinates]
#
#
# # Save the JSON to a file
# output_file_path = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/archive/국토정보플랫폼/rounded_coordinates.json"
# with open(output_file_path, "w") as json_file:
#     json.dump(rounded_coordinates, json_file)

image_paths = sorted(glob("/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/satellite_segmentation_241113/image" + "/*.png"))
label_paths = sorted(glob("/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/satellite_segmentation_241113/segmentation_label" + "/*.json"))
# for image_path in image_paths:
#     os.rename(image_path, image_path.replace(os.path.basename(image_path), os.path.basename(image_path).split('_')[1]))
# for label_path in label_paths:
#     os.rename(label_path, label_path.replace(os.path.basename(label_path), os.path.basename(label_path).split('_')[1]))

# a = []
# for label_path in label_paths:
#     if not os.path.exists(label_path.replace('segmentation_label', 'image').replace('.json', '.png')):
#         print(label_path)
#         a.append(label_path)
# print(len(a))