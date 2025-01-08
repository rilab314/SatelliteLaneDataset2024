import os
import shutil

coordinates = [
    "126.8301,37.531864",
    "126.8301,37.533392",
    "126.8807,37.48944",
    "126.6695526,37.4759918",
    "126.85034,37.496719999",
    "126.85218,37.494428",
    "126.85218,37.497484",
    "126.85218,38.521932",
    "126.85218,37.556312",
    "126.86414,37.4929",
    "126.86873,37.572356",
    "126.87886,37.503595999",
    "126.88254,37.517348",
    "126.88346,37.510472",
    "126.88714,37.50436",
    "126.88714,37.480675999",
    "126.88898,37.54648",

    "126.8991,37.479147999",
    "126.9175,37.497484",
    "126.9175,37.48220399999",
    "126.9175,37.5280439999",
    "126.9221,37.606736",
    "126.9359,37.48526",
    "126.9497,37.50436",
    "126.9543,37.531864",
    "126.9589,37.60062399",
    "126.9727,37.576176",
    "126.9727,37.58763599",

    "126.97362000000001,37.583051999999995",
    "126.98190000000001,37.48526",
    "126.98190000000001,37.510472",

    "126.9221,37.606736",
    "126.9727,37.596804",
    "126.9773,37.601388",
    "126.85034,37.496719999999996",
    "126.88990000000001,37.5502",

]

src_folder = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/satellite_good_matching_241119/image"
src_label_folder = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/total_origin_lonlat/label"

image_folder = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/figure/image"
label_folder = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/figure/label"
os.makedirs(image_folder, exist_ok=True)
os.makedirs(label_folder, exist_ok=True)


for coord in coordinates:
    for file_name in os.listdir(src_folder):
        if coord in file_name:
            src_file = os.path.join(src_folder, file_name)
            dst_file = os.path.join(image_folder, file_name)
            shutil.copy(src_file, dst_file)
            print(f"Moved: {file_name}")

for coord in coordinates:
    for file_name in os.listdir(src_label_folder):
        if coord in file_name:
            src_file = os.path.join(src_label_folder, file_name)
            dst_file = os.path.join(label_folder, file_name)
            shutil.copy(src_file, dst_file)
            print(f"Moved: {file_name}")



