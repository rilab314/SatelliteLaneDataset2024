# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ADE20KDataset(BaseSegDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=('ignore',
                 'center_line',
                 'variable_lane',
                 'u_turn_zone_line',
                 'lane_line',
                 'bus_only_lane',
                 'edge_line',
                 'others',
                 'path_change_restriction_line',
                 'no_parking_stopping_line',
                 'guiding_line',
                 'stop_line',
                 'safety_zone',
                 'bicycle_lane'
                 ),
        palette=[
            (0, 0, 0),
            (77, 77, 255),  # center_line
            (77, 128, 255),  # variable_lane
            (77, 178, 255),  # u_turn_zone_line
            (77, 255, 77),  # lane_line
            (255, 153, 77),  # bus_only_lane
            (255, 77, 77),  # edge_line
            (255, 77, 153),  # others
            (178, 77, 255),  # path_change_restriction_line
            (77, 255, 178),  # no_parking_stopping_line
            (255, 178, 77),  # guiding_line
            (77, 102, 255),  # stop_line
            (255, 77, 128),  # safety_zone
            (128, 255, 77),  # bicycle_lane
        ])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
