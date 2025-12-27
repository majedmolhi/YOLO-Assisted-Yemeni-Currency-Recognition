"""
Build ROI datasets using YOLO detector with adaptive padding.

Input:
    folds/raw/<DATASET>/fold_i/{train,val}/{classes}

Output:
    folds/roi/<DATASET>/fold_i/{train,val}/{classes}

Dataset paths are provided via environment variables.
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ===================== CONFIG =====================
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

RAW_FOLDS_ROOT = Path(os.environ["RAW_FOLDS_ROOT"])
ROI_FOLDS_ROOT = Path(os.environ["ROI_FOLDS_ROOT"])
YOLO_WEIGHTS   = os.environ["YOLO_WEIGHTS"]

TARGET_FILL = 0.50
MIN_GROW_PX = 10
MAX_GROW_PX = 180

# ===================== YOLO =====================
det = YOLO(YOLO_WEIGHTS)

# ===================== HELPERS =====================
def best_box_xyxy(boxes, confs):
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * confs
    return boxes[int(np.argmax(areas))]

def adaptive_padded_box(box, img_shape,
                        target_fill=0.50,
                        min_grow_px=8,
                        max_grow_px=None):
    h, w = img_shape[:2]
    x1, y1, x2, y2 = box.astype(float)

    bw, bh = max(1.0, x2-x1), max(1.0, y2-y1)
    cx, cy = x1 + bw/2, y1 + bh/2

    crop_w = bw / target_fill
    crop_h = bh / target_fill

    grow_x = max((crop_w - bw)/2, min_grow_px)
    grow_y = max((crop_h - bh)/2, min_grow_px)

    if max_grow_px:
        grow_x = min(grow_x, max_grow_px)
        grow_y = min(grow_y, max_grow_px)

    grow_x = min(grow_x, cx, w-cx)
    grow_y = min(grow_y, cy, h-cy)

    x1p = int(round(cx - (bw/2 + grow_x)))
    x2p = int(round(cx + (bw/2 + grow_x)))
    y1p = int(round(cy - (bh/2 + grow_y)))
    y2p = int(round(cy + (bh/2 + grow_y)))

    x1p, y1p = max(0, x1p), max(0, y1p)
    x2p, y2p = min(w, x2p), min(h, y2p)

    return x1p, y1p, x2p, y2p

# ===================== BUILD ROI =====================
def build_roi_for_fold(raw_fold, roi_fold):
    if roi_fold.exists():
        shutil.rmtree(roi_fold)

    for split in ["train", "val"]:
        for cls_dir in (raw_fold / split).iterdir():
            if not cls_dir.is_dir():
                continue

            dst = roi_fold / split / cls_dir.name
            dst.mkdir(parents=True, exist_ok=True)

            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() not in EXTS:
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                r = det.predict(img, imgsz=640, conf=0.35, iou=0.5, verbose=False)[0]

                if r.boxes is not None and len(r.boxes) > 0:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    box = best_box_xyxy(boxes, confs)
                    x1, y1, x2, y2 = adaptive_padded_box(
                        box, img.shape,
                        TARGET_FILL, MIN_GROW_PX, MAX_GROW_PX
                    )
                    crop = img[y1:y2, x1:x2]
                else:
                    crop = img

                cv2.imwrite(str(dst / img_path.name), crop)

# ===================== MAIN =====================
raw_folds = sorted(RAW_FOLDS_ROOT.glob("fold_*"))

for fold in raw_folds:
    roi_fold = ROI_FOLDS_ROOT / fold.name
    print(f"Building ROI for {fold.name}")
    build_roi_for_fold(fold, roi_fold)

print("ROI generation completed successfully")
