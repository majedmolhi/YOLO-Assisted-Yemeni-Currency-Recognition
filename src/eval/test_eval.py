"""
Final test evaluation for RAW / ROI.
Selects best fold by CV val_acc and evaluates on held-out test set.
"""

import os, json
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

IMG_SIZE = (int(os.environ.get("IMG_SIZE", 224)),
            int(os.environ.get("IMG_SIZE", 224)))
BATCH = int(os.environ.get("BATCH", 32))

TEST_ROOT    = Path(os.environ["TEST_ROOT"])
RESULTS_ROOT = Path(os.environ["RESULTS_ROOT"]) / os.environ["EXP_NAME"]

# Load CV results
with open(RESULTS_ROOT/"cv_results.json", "r") as f:
    cv = json.load(f)

best_idx = int(np.argmax(cv["val_acc"]))
print(f"Best fold: {best_idx+1}")

# Load model
model = keras.models.load_model(RESULTS_ROOT/f"model_fold{best_idx}.keras")

# Test dataset
test_ds = keras.utils.image_dataset_from_directory(
    TEST_ROOT,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="int",
    shuffle=False
)

loss, acc = model.evaluate(test_ds, verbose=0)
print("="*50)
print("FINAL TEST RESULTS")
print("="*50)
print(f"Accuracy: {acc*100:.2f}%")
print(f"Loss: {loss:.4f}")

# Save
with open(RESULTS_ROOT/"test_results.json", "w") as f:
    json.dump({"loss": float(loss), "acc": float(acc)}, f)
