import cv2
import numpy as np
import os
from glob import glob


class YellowBoxCrop:
    def __init__(self):
        self.box_pixel_coord = [168, 350, 1192, 1374]    # yellow box y1, x1, y2, x2 in image

    def image_crop(self, image_path):
        image = cv2.imread(image_path)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        crop_image = image[self.box_pixel_coord[0]:self.box_pixel_coord[2], self.box_pixel_coord[1]:self.box_pixel_coord[3]]

        self.image_save(crop_image, image_path)



    def image_save(self, image, image_path):
        output_path = image_path.replace("original", "1024x1024")
        cv2.imwrite(output_path, image)
        p=1

if __name__ == '__main__':
    box_crop = YellowBoxCrop()

    drive_path = ("/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/NOAA/gangnam/gangnam_around_dogok_station/original/box")
    img_list = glob(os.path.join(drive_path, "*.png"))
    for img_path in img_list:
        box_crop.image_crop(img_path)