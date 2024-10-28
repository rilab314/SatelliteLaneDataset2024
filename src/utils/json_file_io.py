from typing import List

from src.dto import GeometryObject, RoadObject


def write_to_json(save_path: str, geometries: List):
    '''
    :param save_path: save path
    :param geometries: list of any dataclass objects
    :return:
    list of dicts로 serialize 하고
    json 파일에 쓰는건, GPT에게 물어봐서 직접 구현하기
    [
        {
            'class': 'GeometryObject',
            'id': '',
            'kind': '',
            'geometry': [[1, 2], [2, 3], ...]  # 한줄에 나오게
        }
    ]
    '''
    pass


class JsonFileReader:
    def __init__(self, root_path=None):
        self.root_path = root_path
        self.shape_list = self.list_files(root_path)

    def list_files(self, root_path) -> List[str]:
        if root_path is None:
            return []
        pass

    def get_file_list(self):
        return self.shape_list

    def read(self, shape_path: str) -> List:
        if not os.path.exists(shape_path):
            return []
        pass


