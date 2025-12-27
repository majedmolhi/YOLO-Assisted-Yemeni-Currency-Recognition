"""
Generic stratified split:
- 15% held-out test set
- Remaining data -> K-Fold Cross Validation (default 5 folds)

Controlled entirely by environment variables:
RAW_ROOT   : path to raw class folders
TEST_ROOT  : output path for test set
FOLDS_ROOT : output path for CV folds
"""

import os
import shutil
import random
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

# ================= CONFIG FROM ENV =================
RAW_ROOT   = Path(os.environ.get("RAW_ROOT", "data_raw"))
TEST_ROOT  = Path(os.environ.get("TEST_ROOT", "data_test"))
FOLDS_ROOT = Path(os.environ.get("FOLDS_ROOT", "folds/raw"))

TEST_RATIO = float(os.environ.get("TEST_RATIO", 0.15))
N_FOLDS    = int(os.environ.get("N_FOLDS", 5))
SEED       = int(os.environ.get("SEED", 1337))

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)
np.random.seed(SEED)

# ================= COLLECT IMAGES =================
images, labels = [], []

for cls_dir in sorted([d for d in RAW_ROOT.iterdir() if d.is_dir()]):
    for img in cls_dir.iterdir():
        if img.suffix.lower() in EXTS:
            images.append(img)
            labels.append(cls_dir.name)

images = np.array(images)
labels = np.array(labels)

print(f"Total images: {len(images)}")
print(f"Classes: {sorted(set(labels))}")

# ================= TEST SPLIT =================
imgs_cv, imgs_test, labels_cv, labels_test = train_test_split(
    images,
    labels,
    test_size=TEST_RATIO,
    stratify=labels,
    random_state=SEED
)

if TEST_ROOT.exists():
    shutil.rmtree(TEST_ROOT)

for img, lbl in zip(imgs_test, labels_test):
    dst = TEST_ROOT / lbl
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, dst / img.name)

print(f"Held-out test set created at: {TEST_ROOT}")

# ================= K-FOLD SPLIT =================
if FOLDS_ROOT.exists():
    shutil.rmtree(FOLDS_ROOT)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(imgs_cv, labels_cv), start=1):
    fold_dir = FOLDS_ROOT / f"fold_{fold_idx}"

    for idx in train_idx:
        src = imgs_cv[idx]
        lbl = labels_cv[idx]
        dst = fold_dir / "train" / lbl
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst / src.name)

    for idx in val_idx:
        src = imgs_cv[idx]
        lbl = labels_cv[idx]
        dst = fold_dir / "val" / lbl
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst / src.name)

    print(f"Fold {fold_idx} created")

print(f"{N_FOLDS}-Fold CV split completed successfully")
