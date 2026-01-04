# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from argparse import ArgumentParser

import cv2
import numpy as np

from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model

def satellite_classes():
    return [
        'ignore', 'center_line', 'u_turn_zone_line', 'lane_line',
        'bus_only_lane', 'edge_line', 'path_change_restriction_line',
        'no_parking_stopping_line', 'guiding_line', 'stop_line',
        'safety_zone', 'bicycle_lane'
    ]

def satellite_palette():
    return [
        [0, 0, 0], [77, 77, 255], [77, 178, 255], [77, 255, 77],
        [255, 153, 77], [255, 77, 77], [178, 77, 255], [77, 255, 178],
        [255, 178, 77], [77, 102, 255], [255, 77, 128], [128, 255, 77]
    ]

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif', '.webp')

def is_image(path: str) -> bool:
    return path.lower().endswith(IMG_EXTS)

def list_images(root: str):
    for name in sorted(os.listdir(root)):
        p = osp.join(root, name)
        if osp.isfile(p) and is_image(p):
            yield p

def extract_label_mask(result):
    if hasattr(result, 'pred_sem_seg') and hasattr(result.pred_sem_seg, 'data'):
        return result.pred_sem_seg.data.squeeze(0).to('cpu').numpy().astype(np.int32)

    if isinstance(result, (list, tuple)):
        arr = result[0]
        return np.array(arr, dtype=np.int32)

    raise TypeError(f'Unsupported result type: {type(result)}')

def get_classes_and_palette(model, palette_name):
    classes = None
    palette = None

    if hasattr(model, 'dataset_meta') and isinstance(model.dataset_meta, dict):
        classes = model.dataset_meta.get('classes', None)
        palette = model.dataset_meta.get('palette', None)

    if classes is None:
        classes = getattr(model, 'CLASSES', None)
    if palette is None:
        palette = getattr(model, 'PALETTE', None)

    if palette_name:
        name = palette_name.lower()
        if name == 'satellite':
            classes = classes or satellite_classes()
            palette = satellite_palette()
        else:
            try:
                from mmseg.core.evaluation import get_palette as mmseg_get_palette
                palette = mmseg_get_palette(name)
            except Exception:
                pass

    if palette is None:
        n = len(classes) if classes is not None else 256
        rng = np.random.default_rng(123)
        palette = (rng.integers(0, 256, size=(n, 3))).tolist()

    return classes, palette

def colorize_mask(label_mask: np.ndarray, palette_rgb: list[list[int]]) -> np.ndarray:
    h, w = label_mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    max_idx = min(len(palette_rgb) - 1, label_mask.max() if label_mask.size else 0)
    for idx in range(max_idx + 1):
        color[label_mask == idx] = palette_rgb[idx][::-1]  # RGB->BGR
    return color

def save_raw_mask_png(path: str, label_mask: np.ndarray):
    if label_mask.dtype != np.uint16:
        label_mask = label_mask.astype(np.uint16)
    cv2.imwrite(path, label_mask)

def process_one(model, img_path, out_color_dir, out_raw_dir,
                palette_rgb, label_shift, overlay_out, overlay_alpha):
    result = inference_model(model, img_path)
    label = extract_label_mask(result)  # [H,W] int32

    if label_shift != 0:
        label = label.astype(np.int64) + int(label_shift)
        label[label < 0] = 0

    base = osp.splitext(osp.basename(img_path))[0]

    if out_color_dir:
        os.makedirs(out_color_dir, exist_ok=True)
        color = colorize_mask(label, palette_rgb)
        cv2.imwrite(osp.join(out_color_dir, f'{base}.png'), color)
    if out_raw_dir:
        os.makedirs(out_raw_dir, exist_ok=True)
        save_raw_mask_png(osp.join(out_raw_dir, f'{base}.png'), label)
    if overlay_out:
        os.makedirs(overlay_out, exist_ok=True)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        color = colorize_mask(label, palette_rgb)
        overlay = cv2.addWeighted(img, 1.0 - overlay_alpha, color, overlay_alpha, 0.0)
        cv2.imwrite(osp.join(overlay_out, f'{base}_overlay.png'), overlay)

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file OR a directory of images')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device for inference')

    parser.add_argument('--out-color-mask', default='masks_color',
                        help='Directory to save color masks (RGB palette applied). Use "" to disable.')
    parser.add_argument('--out-raw-mask', default='',
                        help='Directory to save raw label masks (single-channel PNG). Use "" to disable.')
    parser.add_argument('--overlay-out', default='', help='(Optional) Save overlay images to this dir. Use "" to disable.')
    parser.add_argument('--overlay-alpha', type=float, default=0.5, help='Overlay opacity (0~1).')
    parser.add_argument('--label-shift', type=int, default=0,
                        help='Add this value to predicted labels (e.g., -1 to shift 1..C -> 0..C-1).')
    parser.add_argument('--palette-name', default='',
                        help='Force palette by name (e.g., ade20k, cityscapes, cocostuff, satellite). Leave empty to use model meta.')

    args = parser.parse_args()

    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    classes, palette = get_classes_and_palette(model, args.palette_name)

    paths = [args.img] if osp.isfile(args.img) else list(list_images(args.img))

    if not paths:
        raise FileNotFoundError(f'No images found at: {args.img}')

    for p in paths:
        process_one(
            model=model,
            img_path=p,
            out_color_dir=args.out_color_mask if args.out_color_mask else None,
            out_raw_dir=args.out_raw_mask if args.out_raw_mask else None,
            palette_rgb=palette,
            label_shift=args.label_shift,
            overlay_out=args.overlay_out if args.overlay_out else None,
            overlay_alpha=args.overlay_alpha
        )

if __name__ == '__main__':
    main()
