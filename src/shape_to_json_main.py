from src.utils.shape_file_reader import ShapeFileReader
from src.utils.json_file_io import write_to_json
import src.config.config as cfg


def shape_to_json():
    reader = ShapeFileReader(cfg.SHAPE_PATH)
    shape_list = reader.list_files()
    for shape_path in shape_list:
        geometries = reader.read(shape_path)
        write_to_json(cfg.JSON_PATH, geometries)
