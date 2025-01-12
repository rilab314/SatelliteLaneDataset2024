import build_dataset
import shape_to_json_main
import generate_labels
import generate_coord_list

if __name__ == '__main__':
    shape_to_json_main.shape_to_json()
    generate_coord_list.generate_coord_list()
    generate_labels.generate_labels()
    build_dataset.build_dataset()