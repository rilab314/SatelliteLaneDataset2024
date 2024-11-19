import os
import cv2
import glob
import shutil
from tqdm import tqdm

if __name__ == '__main__':

    drawn_image_dir = '/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/total_origin_lonlat/drawn_image'

    for drawn_image_path in tqdm(sorted(glob.glob(os.path.join(drawn_image_dir, '*.png')))):
        origin_image_path = drawn_image_path.replace('drawn_image', 'image')
        shutil.copyfile(origin_image_path, origin_image_path.replace('image', 'good_image'))
        origin_label_path = drawn_image_path.replace('drawn_image', 'label').replace('.png', '.json')
        shutil.copyfile(origin_label_path, origin_label_path.replace('label', 'good_label'))
        p=1