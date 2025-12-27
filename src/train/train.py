"""
training script for RAW / ROI experiments.

"""

import os, json
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ===================== CONFIG (from env)
SEED     = int(os.environ.get("SEED", 1337))
IMG_SIZE = (int(os.environ.get("IMG_SIZE", 224)),
            int(os.environ.get("IMG_SIZE", 224)))
BATCH    = int(os.environ.get("BATCH", 32))
EPOCHS   = int(os.environ.get("EPOCHS", 10))
N_FOLDS  = int(os.environ.get("N_FOLDS", 5))

FOLDS_ROOT   = Path(os.environ["FOLDS_ROOT"])
EXP_NAME     = os.environ.get("EXP_NAME", "EXP")
RESULTS_ROOT = Path(os.environ["RESULTS_ROOT"])
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# ===================== REPRODUCIBILITY
os.environ["TF_DETERMINISTIC_OPS"] = "1"
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ===================== DATA
def make_ds(root):
    train = keras.utils.image_dataset_from_directory(
        root/"train", image_size=IMG_SIZE, batch_size=BATCH,
        label_mode="int", shuffle=True, seed=SEED)
    val = keras.utils.image_dataset_from_directory(
        root/"val", image_size=IMG_SIZE, batch_size=BATCH,
        label_mode="int", shuffle=False)

    AUTOTUNE = tf.data.AUTOTUNE
    return (
        train.cache().prefetch(AUTOTUNE),
        val.cache().prefetch(AUTOTUNE),
        train.class_names
    )

def compute_class_weights(train_root, class_names):
    counts = [len(list((train_root/c).glob("*"))) for c in class_names]
    total = sum(counts)
    return {i: total/(len(class_names)*max(counts[i],1))
            for i in range(len(class_names))}

# ===================== AUGMENTATION
augment = keras.Sequential([
    layers.RandomFlip("horizontal", seed=SEED),
    layers.RandomRotation(0.04, fill_mode="reflect", seed=SEED),
    layers.RandomZoom(0.10, fill_mode="reflect", seed=SEED),
    layers.RandomContrast(0.20, seed=SEED),
    layers.RandomBrightness(0.15, seed=SEED),
], name="augment_common")

# ===================== MODEL
def build_model(num_classes):
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = augment(inputs)
    x = keras.applications.efficientnet_v2.preprocess_input(x)

    base = keras.applications.EfficientNetV2S(
        include_top=False, weights="imagenet",
        input_shape=IMG_SIZE + (3,)
    )
    for l in base.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False
    base.trainable = True

    x = base(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.AdamW(3e-4, weight_decay=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )
    return model

def callbacks(tag):
    return [
        keras.callbacks.ModelCheckpoint(
            RESULTS_ROOT/f"{tag}.keras",
            monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2)
    ]

# ===================== TRAIN
histories = []
cv_results = {"train_acc": [], "val_acc": []}

folds = sorted(FOLDS_ROOT.glob("fold_*"))

print("="*60)
print(f"Training Experiment: {EXP_NAME}")

print("="*60)

for i, fold in enumerate(folds):
    print(f"\nTraining Fold {i+1}/{N_FOLDS}")

    train_ds, val_ds, class_names = make_ds(fold)
    cw = compute_class_weights(fold/"train", class_names)

    model = build_model(len(class_names))
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks(f"model_fold{i}"),
        class_weight=cw,
        verbose=1
    )



    tr = max(hist.history.get("acc", [0]))
    va = max(hist.history.get("val_acc", [0]))

    cv_results["train_acc"].append(tr)
    cv_results["val_acc"].append(va)
    histories.append(hist.history)

    print(f"\n Fold {i+1} Results:")
    print(f"   Train={tr:.4f}, Val={va:.4f}")
    print("="*60)

# ===================== SAVE
with open(RESULTS_ROOT/"histories.json", "w") as f:
    json.dump(histories, f)

with open(RESULTS_ROOT/"cv_results.json", "w") as f:
    json.dump(cv_results, f)

mean_acc = np.mean(cv_results["val_acc"])
std_acc  = np.std(cv_results["val_acc"])

print("\nTraining completed.")
print(f"Mean Val Acc: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
