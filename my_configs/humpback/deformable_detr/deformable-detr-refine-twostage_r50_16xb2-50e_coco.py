_base_ = 'deformable-detr-refine_r50_16xb2-50e_coco.py'
load_from = '/home/humpback/shin_workspace/OPEN_SOURCE/SatelliteMMdetection2024/pre_trained_checkpoints/deformable-detr-refine-twostage_r50_16xb2-50e_coco_20221021_184714-acc8a5ff.pth'
model = dict(as_two_stage=True)
