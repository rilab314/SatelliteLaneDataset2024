import os
import json

if __name__ == '__main__':
    coco_data = "/media/falcon/IanBook8T1/ArchiveDrive/PublicDataset/unzips/coco/annotations/instances_train2017.json"
    with open(coco_data, 'r', encoding='utf-8') as file:
        data = json.load(file)

    print()