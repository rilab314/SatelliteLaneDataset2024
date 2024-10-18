import os
import json
import shutil
from glob import glob



if __name__ == '__main__':
    json_path = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/국토정보플랫폼/400x400array_data.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    image_drive = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/국토정보플랫폼/국토위성이미지_크롤러_240930/origin_lonlat"
    image_list = glob(os.path.join(image_drive, "*.png"))
    image_list.sort()

    save_drive = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/국토정보플랫폼/dataset/a"
    for image_path in image_list:
        lon, lat = os.path.basename(image_path)[:-4].split("_")[1].split(",")
        if [float(lon), float(lat)] in data:
            shutil.copy(image_path, os.path.join(save_drive, os.path.basename(image_path)))

    p=1