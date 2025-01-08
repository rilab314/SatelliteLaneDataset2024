_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
load_from = '/home/humpback/shin_workspace/OPEN_SOURCE/mmsegmentation/pre_train/upernet/upernet_r50_512x512_160k_ade20k_20200615_184328-8534de8d.pth'
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=150),
    auxiliary_head=dict(num_classes=150))
