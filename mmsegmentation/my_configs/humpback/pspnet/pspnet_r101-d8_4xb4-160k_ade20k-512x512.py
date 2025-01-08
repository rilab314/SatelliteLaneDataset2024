_base_ = './pspnet_r50-d8_4xb4-160k_ade20k-512x512.py'
load_from = '/home/humpback/shin_workspace/OPEN_SOURCE/mmsegmentation/pre_train/pspnet/pspnet_r101-d8_512x512_160k_ade20k_20200615_100650-967c316f.pth'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
