_base_ = './yolox_s_rotate_8xb8-300e_coco.py'
load_from = '/home/humpback/shin_workspace/OPEN_SOURCE/SatelliteMMdetection2024/pre_trained_checkpoints/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
# model settings
model = dict(
    backbone=dict(deepen_factor=1.0, widen_factor=1.0),
    neck=dict(
        in_channels=[256, 512, 1024], out_channels=256, num_csp_blocks=3),
    bbox_head=dict(in_channels=256, feat_channels=256))
