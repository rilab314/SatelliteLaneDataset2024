import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict, Counter
from typing import Tuple, List, Dict

from matplotlib.pyplot import figure


def calculate_statistics(label_folder: str) -> Tuple[pd.DataFrame, Dict[str, int], Counter, Counter]:
    """
    Calculate dataset statistics, class counts, type counts, and instances per image.

    Args:
        label_folder (str): Path to the folder containing label files.

    Returns:
        Tuple[pd.DataFrame, Dict[str, int], Counter, Counter]: Dataframe of overall statistics,
                                                              dictionary of class counts,
                                                              Counter of type counts,
                                                              Counter of instances per image.
    """
    label_files = glob(os.path.join(label_folder, "*.json"))

    total_images = len(label_files)
    total_instances = 0
    total_points = 0
    class_counts = defaultdict(int)
    type_counts = Counter()
    instances_per_image = Counter()

    for label_file in label_files:
        with open(label_file, 'r') as f:
            data = json.load(f)
            image_instances = 0
            for obj in data:
                if obj['class'] == 'RoadObject' and obj['category'] != 'None' and obj['type'] != 'undefined' and obj['category'] != 'others':
                    total_instances += 1
                    total_points += len(obj['image_points'])
                    class_counts[obj['category']] += 1
                    type_counts[obj['type']] += 1
                    image_instances += 1
            instances_per_image[image_instances] += 1

    # Calculate averages
    avg_instances_per_image = total_instances / total_images if total_images else 0
    avg_points_per_instance = total_points / total_instances if total_instances else 0

    # Create a DataFrame for the overall statistics
    statistics_df = pd.DataFrame({
        "total number": [total_images, total_instances, total_points],
        "average": [None, f"{avg_instances_per_image:.2f} per image", f"{avg_points_per_instance:.2f} per instance"]
    }, index=["images", "instances", "points"])

    return statistics_df, dict(class_counts), type_counts, instances_per_image


def plot_sorted_bar(data: Dict, x_label: str, y_label: str, title: str, output_path: str, figure_size: Tuple = (12, 6),font_size: int = 12, add_font_size: int = 12):
    """
    Plot a sorted bar graph for the given data.

    Args:
        data (Dict): Dictionary to plot.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the plot.
        output_path (str): Path to save the plot.
        font_size (int): Font size for labels and title.
    """
    sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))
    x = list(sorted_data.keys())
    y = list(sorted_data.values())

    plt.figure(figsize=figure_size)
    plt.bar(x, y)
    plt.xticks(rotation=45, ha='right', fontsize=add_font_size)
    plt.yticks(fontsize=font_size*0.9)
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    plt.title(title, fontsize=font_size)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def plot_instances_per_image(instances_per_image: Counter, output_path: str, font_size: int = 12):
    """
    Plot the number of images having a specific number of instances.

    Args:
        instances_per_image (Counter): Counter with the number of instances per image.
        output_path (str): Path to save the plot.
        font_size (int): Font size for labels and title.
    """
    sorted_data = dict(sorted(instances_per_image.items()))
    x = list(sorted_data.keys())
    y = list(sorted_data.values())

    # Limit the x-axis to 400
    x_limited = [val for val in x if val <= 400]
    y_limited = y[:len(x_limited)]

    plt.figure(figsize=(12, 6))
    plt.bar(x_limited, y_limited)
    plt.xticks(rotation=45, ha='right', fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel("# of instances per image", fontsize=font_size)
    plt.ylabel("# of images", fontsize=font_size)
    plt.title("Distribution of Images by Number of Instances", fontsize=font_size)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def generate_dataset_summary(label_folder: str, output_dir: str, font_size: int = 12):
    """
    Generate and display dataset statistics and class/type/category distribution plots.

    Args:
        label_folder (str): Path to the folder containing label files.
        output_dir (str): Directory to save the output plots.
        font_size (int): Font size for labels and title.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Calculate statistics
    statistics_df, class_counts, type_counts, instances_per_image = calculate_statistics(label_folder)

    # Display the statistics as a DataFrame
    print("Dataset Statistics:")
    print(statistics_df)

    # Plot class distribution
    plot_sorted_bar(class_counts, "Category Names", "# of Instances", "Instances per Category",
                    os.path.join(output_dir, "category_distribution.png"), figure_size=(17, 7), font_size=font_size+1, add_font_size=font_size+1)

    # Plot type distribution
    plot_sorted_bar(type_counts, "Type Names", "# of Instances", "Instances per Type",
                    os.path.join(output_dir, "type_distribution.png"), figure_size=(14, 7.5), font_size=font_size, add_font_size=font_size)

    # Plot instances per image distribution
    plot_instances_per_image(instances_per_image, os.path.join(output_dir, "instances_per_image_distribution.png"),
                             font_size=font_size)


if __name__ == "__main__":
    label_folder_path = "/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2024_SATELLITE/datasets/satellite_good_matching_241125/label"
    output_dir = "/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2024_SATELLITE/datasets/figure/summary"

    # Font size parameter
    font_size = 25

    generate_dataset_summary(label_folder_path, output_dir, font_size=font_size)
