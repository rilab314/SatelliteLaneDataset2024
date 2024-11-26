import os
import json
from glob import glob

def generate_train_val_test_coords(drive, json_save_path, dataset_ratio, regions_config):
    label_paths = glob(os.path.join(drive, "*.json"))
    dataset = {"train": [], "validation": [], "test": []}

    for region in regions_config:
        label_coords = []
        lon_range = region['LONGITUDE_RANGE']
        lat_range = region['LATITUDE_RANGE']

        for label_path in label_paths:
            try:
                label_lon, label_lat = map(float, os.path.basename(label_path)[:-5].split(','))
                if lon_range[0] <= label_lon <= lon_range[1] and lat_range[0] <= label_lat <= lat_range[1]:
                    label_coords.append((label_lon, label_lat, os.path.basename(label_path)[:-5]))
            except ValueError:
                print(f"Skipping invalid file: {label_path}")

        label_coords.sort(key=lambda x: x[0])

        total = len(label_coords)
        train_end = int(total * dataset_ratio['train'])
        val_end = train_end + int(total * dataset_ratio['validation'])

        dataset["train"].extend([coord[2] for coord in label_coords[:train_end]])  # Names for train
        dataset["validation"].extend(
            [coord[2] for coord in label_coords[train_end:val_end]])  # Names for validation
        dataset["test"].extend([coord[2] for coord in label_coords[val_end:]])  # Names for test

    # Save the dataset split into JSON
    with open(json_save_path, 'w') as f:
        json.dump(dataset, f)