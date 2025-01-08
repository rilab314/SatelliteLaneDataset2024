import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# 등록할 데이터 셋의 id(이미지 클래스 값) 설정
RILAB_LG_15 = [{'id': 0, 'name': 'None', 'color':(0, 0, 0)},
               {'id': 1, 'name': 'ignore_label', 'color':(0, 0, 0)},
               {'id': 2, 'name': 'center_line', 'color': (77, 77, 255)},
               {'id': 3, 'name': 'u_turn_zone_line', 'color':(77, 178, 255)},
               {'id': 4, 'name': 'lane_line', 'color':(77, 255, 77)},
               {'id': 5, 'name': 'bus_only_lane', 'color':(255, 153, 77)},
               {'id': 6, 'name': 'edge_line', 'color':(255, 77, 77)},
               {'id': 7, 'name': 'path_change_restriction_line', 'color':(178, 77, 255)},
               {'id': 8, 'name': 'no_parking_stopping_line', 'color':(77, 255, 178)},
               {'id': 9, 'name': 'guiding_line', 'color':(255, 178, 77)},
               {'id': 10, 'name': 'stop_line', 'color':(77, 102, 255)},
               {'id': 11, 'name': 'safety_zone', 'color':(255, 77, 128)},
               {'id': 12, 'name': 'bicycle_lane', 'color':(128, 255, 77)},
               ]


def get_satellite_2024_full_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.
    stuff_ids = [k["id"] for k in RILAB_LG_15]
    # assert len(stuff_ids) == 847, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in RILAB_LG_15]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret


def register_sem_seg_satellite_2024(root):
    root = os.path.join(root, 'satellite_ade20k_241213') # dataset 경로 설정
    meta = get_satellite_2024_full_meta()
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations", dirname)
        name = f"satellite_detectron_{name}" # config에서 사용될 데이터셋 이름 설정
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=None
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_sem_seg_satellite_2024(_root)
