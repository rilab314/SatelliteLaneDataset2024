import os
from tqdm import tqdm

from src.utils.shape_file_reader import ShapeFileReader
from src.utils.json_file_io import write_to_json
import src.config.config as cfg


def shape_to_json():
    reader = ShapeFileReader(cfg.SHAPE_PATH)
    shape_list = reader.list_files()
    for shape_path in tqdm(shape_list, desc='Shapes to JSON'):
        try:
            geometries = reader.read(shape_path)
            save_path = os.path.join(cfg.JSON_PATH, shape_path.split('/')[-3]+'.json')
            write_to_json(save_path, geometries)
        except Exception as e:
            print(f"Error shape_path: {shape_path}")
            print(e)

if __name__ == '__main__':
    shape_to_json()
