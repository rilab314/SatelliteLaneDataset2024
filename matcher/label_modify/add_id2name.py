import json
import os
import glob

import matcher.file_io as file_io
import matcher.config.ID_name_mapping as ID_name_mapping
from matcher.dataclasses_roadobject import RoadObject



def add_names_to_json(json_data):
    dict_list = []
    for item in json_data:
        item['category_id'] = item.pop('category')  # 이름 변경
        item['type_id'] = item.pop('type')

        if item['category_id'] not in ID_name_mapping.KindDict:
            raise KeyError(f'Category ID {item["category_id"]} not found in KindDict.')
        if item['type_id'] not in ID_name_mapping.TypeDict:
            raise KeyError(f'Type ID {item["type_id"]} not found in TypeDict.')

        item['category'] = ID_name_mapping.KindDict.get(item['category_id'], 'Unknown Category')
        item['type'] = ID_name_mapping.TypeDict.get(item['type_id'], 'Unknown Type')
        obj_dict = RoadObject(id=item["id"], category_id=item["category_id"], type_id=item["type_id"], category=item["category"], type=item["type"], points=item["points"])
        dict_list.append(obj_dict)
    return dict_list


def load_and_update_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    updated_data = add_names_to_json(data)
    updated_data = file_io.serialize_dataclass(updated_data)
    file_io.save_json_with_custom_indent(updated_data, file_path)

    # with open(file_path, 'w', encoding='utf-8') as file:
    #     json.dump(updated_data, file, ensure_ascii=False, indent=4)
root_path = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/국토정보플랫폼/국토위성이미지_크롤러_240930/768x768/label"
path_list = glob.glob(os.path.join(root_path, "*.json"))
path_list.sort()
for file in path_list:
    load_and_update_json(file)