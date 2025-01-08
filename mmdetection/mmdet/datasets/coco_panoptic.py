# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Optional, Sequence, Union

from mmdet.registry import DATASETS
from .api_wrappers import COCOPanoptic
from .coco import CocoDataset


@DATASETS.register_module()
class CocoPanopticDataset(CocoDataset):
    """Coco dataset for Panoptic segmentation.

    The annotation format is shown as follows. The `ann` field is optional
    for testing.

    .. code-block:: none

        [
            {
                'filename': f'{image_id:012}.png',
                'image_id':9
                'segments_info':
                [
                    {
                        'id': 8345037, (segment_id in panoptic png,
                                        convert from rgb)
                        'category_id': 51,
                        'iscrowd': 0,
                        'bbox': (x1, y1, w, h),
                        'area': 24315
                    },
                    ...
                ]
            },
            ...
        ]

    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        data_prefix (dict, optional): Prefix for training data. Defaults to
            ``dict(img=None, ann=None, seg=None)``. The prefix ``seg`` which is
            for panoptic segmentation map must be not None.
        filter_cfg (dict, optional): Config for filter data. Defaults to None.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Defaults to None which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy. Defaults
            to True.
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Defaults to False.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Defaults to 1000.
    """

    METAINFO = {
        'classes':
            ('center_line', 'variable_lane', 'u_turn_zone_line', 'lane_line', 'bus_only_lane', 'edge_line', 'others',
             'path_change_restriction_line', 'no_parking_stopping_line', 'guiding_line', 'stop_line', 'safety_zone',
             'bicycle_lane', 'no_stopping_zone', 'crosswalk', 'raised_crosswalk', 'bicycle_crosswalk', 'straight',
             'left_turn',
             'right_turn', 'left_right_turn', 'all_directions', 'straight_and_left_turn', 'straight_and_right_turn',
             'straight_and_u_turn', 'u_turn', 'left_turn_and_u_turn', 'lane_change_merge_left',
             'lane_change_merge_right',
             'uphill_slope'),
        'thing_classes':
            ('center_line', 'variable_lane', 'u_turn_zone_line', 'lane_line', 'bus_only_lane', 'edge_line', 'others',
             'path_change_restriction_line', 'no_parking_stopping_line', 'guiding_line', 'stop_line', 'safety_zone',
             'bicycle_lane', 'no_stopping_zone', 'crosswalk', 'raised_crosswalk', 'bicycle_crosswalk', 'straight',
             'left_turn',
             'right_turn', 'left_right_turn', 'all_directions', 'straight_and_left_turn', 'straight_and_right_turn',
             'straight_and_u_turn', 'u_turn', 'left_turn_and_u_turn', 'lane_change_merge_left',
             'lane_change_merge_right',
             'uphill_slope'),
        'stuff_classes':
            (),
        'palette':
            [(77, 77, 255),    # center_line
             (77, 128, 255),   # variable_lane
             (77, 178, 255),   # u_turn_zone_line
             (77, 255, 77),    # lane_line
             (255, 153, 77),   # bus_only_lane
             (255, 77, 77),    # edge_line
             (255, 77, 153),   # others
             (178, 77, 255),   # path_change_restriction_line
             (77, 255, 178),   # no_parking_stopping_line
             (255, 178, 77),   # guiding_line
             (77, 102, 255),   # stop_line
             (255, 77, 128),   # safety_zone
             (128, 255, 77),   # bicycle_lane
             (77, 255, 153),   # no_stopping_zone
             (255, 255, 77),   # crosswalk
             (128, 77, 255),   # raised_crosswalk
             (102, 77, 255),   # bicycle_crosswalk
             (255, 77, 178),   # straight
             (255, 128, 77),   # left_turn
             (128, 77, 77),    # right_turn
             (255, 77, 255),   # left_right_turn
             (128, 128, 255),  # all_directions
             (77, 255, 77),    # straight_and_left_turn
             (153, 255, 77),   # straight_and_right_turn
             (255, 255, 128),  # straight_and_u_turn
             (255, 102, 77),   # u_turn
             (128, 255, 255),  # left_turn_and_u_turn
             (255, 77, 178),   # lane_change_merge_left
             (102, 255, 77),   # lane_change_merge_right
             (153, 77, 255)    # uphill_slope
             ]
    }
    COCOAPI = COCOPanoptic
    # ann_id is not unique in coco panoptic dataset.
    ANN_ID_UNIQUE = False

    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=None, ann=None, seg=None),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 backend_args: dict = None,
                 **kwargs) -> None:
        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            backend_args=backend_args,
            **kwargs)

    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``.

        Returns:
            dict: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']
        # filter out unmatched annotations which have
        # same segment_id but belong to other image
        ann_info = [
            ann for ann in ann_info if ann['image_id'] == img_info['img_id']
        ]
        data_info = {}

        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].replace('.jpg', '.png'))
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['thing_classes']
            data_info['stuff_text'] = self.metainfo['stuff_classes']
            data_info['custom_entities'] = True  # no important

        instances = []
        segments_info = []
        for ann in ann_info:
            instance = {}
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            category_id = ann['category_id']
            contiguous_cat_id = self.cat2label[category_id]

            is_thing = self.coco.load_cats(ids=category_id)[0]['isthing']
            if is_thing:
                is_crowd = ann.get('iscrowd', False)
                instance['bbox'] = bbox
                instance['bbox_label'] = contiguous_cat_id
                if not is_crowd:
                    instance['ignore_flag'] = 0
                else:
                    instance['ignore_flag'] = 1
                    is_thing = False

            segment_info = {
                'id': ann['id'],
                'category': contiguous_cat_id,
                'is_thing': is_thing
            }
            segments_info.append(segment_info)
            if len(instance) > 0 and is_thing:
                instances.append(instance)
        data_info['instances'] = instances
        data_info['segments_info'] = segments_info
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter images too small or without ground truth.

        Returns:
            List[dict]: ``self.data_list`` after filtering.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        ids_with_ann = set()
        # check whether images have legal thing annotations.
        for data_info in self.data_list:
            for segment_info in data_info['segments_info']:
                if not segment_info['is_thing']:
                    continue
                ids_with_ann.add(data_info['img_id'])

        valid_data_list = []
        for data_info in self.data_list:
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_with_ann:
                continue
            if min(width, height) >= min_size:
                valid_data_list.append(data_info)

        return valid_data_list
