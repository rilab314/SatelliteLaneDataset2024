import shutil
from glob import glob

image_path = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/국토정보플랫폼/dataset/a/768x768/origin_image"

image_list = glob(image_path + "/*.png")
image_list.sort()
for image_path in image_list:
    label_path = image_path.replace("origin_image", "label").replace(".png", ".json")
    shutil.copy(label_path, label_path.replace("label", "mv_label"))