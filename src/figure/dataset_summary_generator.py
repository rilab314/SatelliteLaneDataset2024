import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict
from typing import Tuple, List, Dict


def calculate_statistics(label_folder: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Calculate dataset statistics and class counts.

    Args:
        label_folder (str): Path to the folder containing label files.

    Returns:
        Tuple[pd.DataFrame, Dict[str, int]]: Dataframe of overall statistics and dictionary of class counts.
    """
    label_files = glob(os.path.join(label_folder, "*.json"))

    total_images = len(label_files)
    total_instances = 0
    total_points = 0
    class_counts = defaultdict(int)

    for label_file in label_files:
        with open(label_file, 'r') as f:
            data = json.load(f)
            for obj in data:
                if obj['class'] == 'RoadObject' and obj['category'] != 'None':
                    total_instances += 1
                    total_points += len(obj['image_points'])
                    class_counts[obj['category']] += 1

    # Calculate averages
    avg_instances_per_image = total_instances / total_images if total_images else 0
    avg_points_per_instance = total_points / total_instances if total_instances else 0

    # Create a DataFrame for the overall statistics
    statistics_df = pd.DataFrame({
        "total number": [total_images, total_instances, total_points],
        "average": [None, f"{avg_instances_per_image:.2f} per image", f"{avg_points_per_instance:.2f} per instance"]
    }, index=["images", "instances", "points"])

    return statistics_df, dict(class_counts)


def plot_class_distribution(class_counts: Dict[str, int], output_path: str):
    """
    Plot a bar graph for the distribution of classes.

    Args:
        class_counts (Dict[str, int]): Dictionary of class counts.
        output_path (str): Path to save the class distribution plot.
    """
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(12, 6))
    plt.bar(classes, counts)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def generate_dataset_summary(label_folder: str, output_plot_path: str):
    """
    Generate and display dataset statistics and class distribution plot.

    Args:
        label_folder (str): Path to the folder containing label files.
        output_plot_path (str): Path to save the class distribution plot.
    """
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    statistics_df, class_counts = calculate_statistics(label_folder)

    # Display the statistics as a DataFrame
    print("Dataset Statistics:")
    print(statistics_df)

    # Plot and save the class distribution
    plot_class_distribution(class_counts, output_plot_path)


if __name__ == "__main__":
    label_folder_path = "/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2024_SATELLITE/datasets/satellite_good_matching_241125/label"
    output_plot_path = "/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2024_SATELLITE/datasets/figure/summary/class_distribution.png"

    generate_dataset_summary(label_folder_path, output_plot_path)
