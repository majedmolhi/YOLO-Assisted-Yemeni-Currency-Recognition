"""
Background-only bias ablation for ROI model.
"""

import os, json, shutil
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from ultralytics import YOLO

IMG_SIZE = (224,224)
BATCH = 32

TEST_ROOT    = Path(os.environ["TEST_ROOT"])
RESULTS_ROOT = Path(os.environ["RESULTS_ROOT"]) 
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS")

BG_ROOT = Path("/content/data_bg_only/test")
if BG_ROOT.exists(): shutil.rmtree(BG_ROOT)
BG_ROOT.mkdir(parents=True, exist_ok=True)

det = YOLO(YOLO_WEIGHTS)

def best_box(r):
    if r.boxes is None or len(r.boxes)==0: return None
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])*confs
    return boxes[int(np.argmax(areas))].astype(int)

# Build background-only test
for cls in TEST_ROOT.iterdir():
    if not cls.is_dir(): continue
    out = BG_ROOT/cls.name
    out.mkdir(parents=True, exist_ok=True)
    for p in cls.iterdir():
        img = cv2.imread(str(p))
        r = det.predict(img, imgsz=640, conf=0.35, verbose=False)[0]
        box = best_box(r)
        if box is not None:
            x1,y1,x2,y2 = box
            img[y1:y2, x1:x2] = 128
        cv2.imwrite(str(out/p.name), img)

# Load best ROI model
with open(RESULTS_ROOT/"cv_results.json") as f:
    cv = json.load(f)
best = int(np.argmax(cv["val_acc"]))
model = keras.models.load_model(RESULTS_ROOT/f"model_fold{best}.keras")

def make_ds(root):
    return keras.utils.image_dataset_from_directory(
        root, image_size=IMG_SIZE, batch_size=BATCH,
        label_mode="int", shuffle=False
    )

raw_ds = make_ds(TEST_ROOT)
bg_ds  = make_ds(BG_ROOT)

_, acc_raw = model.evaluate(raw_ds, verbose=0)
_, acc_bg  = model.evaluate(bg_ds,  verbose=0)

bbc = (acc_raw - acc_bg)/acc_raw*100

print("="*50)
print("BIAS ABLATION")
print("="*50)
print(f"RAW acc: {acc_raw*100:.2f}%")
print(f"BG  acc: {acc_bg*100:.2f}%")
print(f"BBC    : {bbc:.2f}%")

with open(RESULTS_ROOT/"bias_ablation.json","w") as f:
    json.dump({"raw_acc":acc_raw,"bg_acc":acc_bg,"bbc":bbc}, f)
