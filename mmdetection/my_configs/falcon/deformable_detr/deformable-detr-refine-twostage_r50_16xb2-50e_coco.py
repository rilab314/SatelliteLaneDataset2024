_base_ = 'deformable-detr-refine_r50_16xb2-50e_coco.py'
load_from = '/home/falcon/shin_work/OPEN_SOURCE/mmdetection/pre_trained_checkpoints/deformable-detr-refine-twostage_r50_16xb2-50e_coco_20221021_184714-acc8a5ff.pth'
model = dict(as_two_stage=True)
