import os
import json
import pandas as pd
from glob import glob
from collections import defaultdict, Counter
from typing import Dict, Tuple
from src.config.config import KindDict, TypeDict

def load_dataset_split(file_path: str) -> Dict[str, list]:
    """
    Load train, validation, and test splits from a JSON file.

    Args:
        file_path (str): Path to the dataset split JSON file.

    Returns:
        Dict[str, list]: Dictionary containing train, validation, and test splits.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_counts(label_folder: str, file_list: list) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Calculate category and type counts for a specific dataset split.

    Args:
        label_folder (str): Path to the folder containing label files.
        file_list (list): List of file identifiers for the dataset split.

    Returns:
        Tuple[Dict[str, int], Dict[str, int]]: Category counts and type counts.
    """
    class_counts = defaultdict(int)
    type_counts = defaultdict(int)

    for identifier in file_list:
        label_file = os.path.join(label_folder, f"{identifier}.json")
        if not os.path.exists(label_file):
            continue

        with open(label_file, 'r') as f:
            data = json.load(f)

            for obj in data:
                if obj['class'] == 'RoadObject' and obj['category'] != 'None' and obj['type'] != 'undefined':
                    class_counts[obj['category']] += 1
                    type_counts[obj['type']] += 1

    return dict(class_counts), dict(type_counts)

def save_combined_counts_to_csv(split_data: Dict[str, list], label_folder: str, output_path: str):
    """
    Save combined category and type counts for all splits into a single CSV file.

    Args:
        split_data (Dict[str, list]): Dataset split information (train, validation, test).
        label_folder (str): Path to the folder containing label files.
        output_path (str): Path to save the combined CSV file.
    """
    rows = []

    # Calculate counts for each split
    counts = {}
    for split_name, file_list in split_data.items():
        class_counts, type_counts = calculate_counts(label_folder, file_list)
        counts[split_name] = {'class': class_counts, 'type': type_counts}

    # Merge counts into rows
    all_categories = set()
    all_types = set()
    for split_counts in counts.values():
        all_categories.update(split_counts['class'].keys())
        all_types.update(split_counts['type'].keys())

    for category in KindDict.values():
        rows.append({
            'name': category,
            'type': 'category',
            'train_count': counts['train']['class'].get(category, 0),
            'val_count': counts['validation']['class'].get(category, 0),
            'test_count': counts['test']['class'].get(category, 0)
        })

    for t in TypeDict.values():
        rows.append({
            'name': t,
            'type': 'type',
            'train_count': counts['train']['type'].get(t, 0),
            'val_count': counts['validation']['type'].get(t, 0),
            'test_count': counts['test']['type'].get(t, 0)
        })

    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    label_folder_path = "/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2024_SATELLITE/datasets/satellite_good_matching_241125/label"
    dataset_split_path = "/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2024_SATELLITE/datasets/satellite_good_matching_241125/dataset.json"

    output_csv_path = "/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2024_SATELLITE/datasets/figure/combined_counts_241213.csv"

    dataset_splits = load_dataset_split(dataset_split_path)
    save_combined_counts_to_csv(dataset_splits, label_folder_path, output_csv_path)
