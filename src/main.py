import build_dataset
from src.utils import labels_to_segmentation_labels

if __name__ == '__main__':
    # shape_to_json_main.shape_to_json()
    # generate_labels.generate_labels()
    build_dataset.build_dataset()
    labels_to_segmentation_labels.generate_segmentation_labels()