import os
import json
import shutil
from glob import glob
from tqdm import tqdm

def generate_train_val_coords(drive, json_save_path, lon_range, lat_range):
    label_paths = glob(os.path.join(drive, "*.json"))

    dataset = {'train': [],
               'validation': []}
    for label_path in label_paths:
        label_lon, label_lat = os.path.basename(label_path)[:-5].split(',')
        if lon_range[0] < float(label_lon) < lon_range[1] and lat_range[0] < float(label_lat) < lat_range[1]:
            dataset['validation'].append(os.path.basename(label_path)[:-5])
        else:
            dataset['train'].append(os.path.basename(label_path)[:-5])
    with open(json_save_path, 'w') as f:
        json.dump(dataset, f)

def divide_train_val(src_path, dst_path, coords_json_path):
    with open(coords_json_path, 'r') as f:
        coords = json.load(f)

    # # detectron2
    # os.makedirs(os.path.join(dst_path, 'images', 'training'), exist_ok=True)
    # os.makedirs(os.path.join(dst_path, 'images', 'validation'), exist_ok=True)
    # os.makedirs(os.path.join(dst_path, 'annotations_detectron2', 'training'), exist_ok=True)
    # os.makedirs(os.path.join(dst_path, 'annotations_detectron2', 'validation'), exist_ok=True)
    #
    # for coord in tqdm(coords['train'], desc='Copy training dataset'):
    #     label_path = os.path.join(src_path, 'segmentation_label', str(coord)+'.json')
    #     moved_label_path = os.path.join(dst_path, 'annotations_detectron2', 'training', str(coord)+'.json')
    #     shutil.copy(label_path, moved_label_path)
    #     image_path = os.path.join(src_path, 'image', str(coord)+'.png')
    #     moved_image_path = os.path.join(dst_path, 'images', 'training', str(coord)+'.png')
    #     shutil.copy(image_path, moved_image_path)
    #
    # for coord in tqdm(coords['validation'], desc='Copy validation dataset'):
    #     label_path = os.path.join(src_path, 'segmentation_label', str(coord)+'.json')
    #     moved_label_path = os.path.join(dst_path, 'annotations_detectron2', 'validation', str(coord)+'.json')
    #     shutil.copy(label_path, moved_label_path)
    #     image_path = os.path.join(src_path, 'image', str(coord)+'.png')
    #     moved_image_path = os.path.join(dst_path, 'images', 'validation', str(coord)+'.png')
    #     shutil.copy(image_path, moved_image_path)

    # coco
    os.makedirs(os.path.join(dst_path, 'label', 'training'), exist_ok=True)
    os.makedirs(os.path.join(dst_path, 'label', 'validation'), exist_ok=True)
    os.makedirs(os.path.join(dst_path, 'train2017'), exist_ok=True)
    os.makedirs(os.path.join(dst_path, 'val2017'), exist_ok=True)
    os.makedirs(os.path.join(dst_path, 'test2017'), exist_ok=True)
    for coord in tqdm(coords['train'], desc='Copy training dataset'):
        label_path = os.path.join(src_path, 'label', str(coord)+'.json')
        moved_label_path = os.path.join(dst_path, 'label', 'training', str(coord)+'.json')
        shutil.copy(label_path, moved_label_path)
        image_path = os.path.join(src_path, 'image', str(coord) + '.png')
        moved_image_path = os.path.join(dst_path, 'train2017', str(coord) + '.png')
        shutil.copy(image_path, moved_image_path)

    for coord in tqdm(coords['validation'], desc='Copy validation dataset'):
        label_path = os.path.join(src_path, 'label', str(coord)+'.json')
        moved_label_path = os.path.join(dst_path, 'label', 'validation', str(coord)+'.json')
        shutil.copy(label_path, moved_label_path)
        image_path = os.path.join(src_path, 'image', str(coord) + '.png')
        moved_image_path = os.path.join(dst_path, 'val2017', str(coord) + '.png')
        shutil.copy(image_path, moved_image_path)
        moved_image_path = os.path.join(dst_path, 'test2017', str(coord) + '.png')
        shutil.copy(image_path, moved_image_path)

if __name__ == '__main__':
    label_drive = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/satellite_good_matching_241119/label"
    coords_save_path = '/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/satellite_good_matching_241119/dataset.json'
    generate_train_val_coords(label_drive, coords_save_path, lon_range=[127.0648, 127.1695], lat_range=[37.4776, 37.5636])


    root_path = '/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/satellite_good_matching_241119'
    save_path = '/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/satellite_coco_241119'
    divide_train_val(root_path, save_path, coords_save_path)
