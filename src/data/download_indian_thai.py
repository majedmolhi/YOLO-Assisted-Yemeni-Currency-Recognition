"""
Download & prepare Indian + Thai Banknotes Dataset
Unified into 15 classes (Training + Validation merged)

Final output:
    /content/IndianThai_data/RAW/<15_classes>/*.jpg
"""

import zipfile
import shutil
import urllib.request
from pathlib import Path

# ================= CONFIG =================
DATA_URL = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/2kfz5yc7pt-1.zip"

ZIP1 = Path("/content/indian_thai_banknotes.zip")
TEMP1 = Path("/content/temp_indian_thai_1")
TEMP2 = Path("/content/temp_indian_thai_2")

FINAL_ROOT = Path("/content/IndianThai_data")
RAW_ROOT   = FINAL_ROOT / "RAW"

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ================= CLEAN =================
for p in [TEMP1, TEMP2, FINAL_ROOT]:
    if p.exists():
        shutil.rmtree(p)

TEMP1.mkdir(parents=True, exist_ok=True)
TEMP2.mkdir(parents=True, exist_ok=True)
RAW_ROOT.mkdir(parents=True, exist_ok=True)

# ================= DOWNLOAD =================
print("Downloading Indian & Thai banknotes dataset...")
urllib.request.urlretrieve(DATA_URL, ZIP1)
print("Download completed.")

# ================= EXTRACT ZIP 1 =================
print("Extracting outer ZIP...")
with zipfile.ZipFile(ZIP1, "r") as z:
    z.extractall(TEMP1)

# ================= EXTRACT ZIP 2 =================
inner_zip = list(TEMP1.rglob("Indian_Thai_BankNotes_Dataset.zip"))
if not inner_zip:
    raise RuntimeError("Inner dataset ZIP not found")

print("Extracting inner ZIP...")
with zipfile.ZipFile(inner_zip[0], "r") as z:
    z.extractall(TEMP2)

DATA_ROOT = TEMP2 / "Indian_Thai_BankNotes_Dataset"
if not DATA_ROOT.exists():
    raise RuntimeError("Dataset root folder not found")

# ================= ORGANIZE DATA =================
print("Organizing dataset into unified classes...")

total_images = 0
class_counts = {}

for currency_dir in DATA_ROOT.iterdir():
    if not currency_dir.is_dir():
        continue

    for split in ["Training", "Validation"]:
        split_dir = currency_dir / split
        if not split_dir.exists():
            continue

        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue

            cls_name = f"{currency_dir.name}_{cls_dir.name}"
            dst = RAW_ROOT / cls_name
            dst.mkdir(parents=True, exist_ok=True)

            for img in cls_dir.iterdir():
                if img.suffix.lower() in EXTS:
                    shutil.copy2(img, dst / img.name)
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    total_images += 1

# ================= CLEANUP =================
ZIP1.unlink()
shutil.rmtree(TEMP1)
shutil.rmtree(TEMP2)

# ================= SUMMARY =================
print("\n" + "="*60)
print("INDIAN + THAI DATASET READY")
print("="*60)

for cls in sorted(class_counts):
    print(f"{cls:40s}: {class_counts[cls]}")

print(f"\nTOTAL images: {total_images}")
print("\nFinal dataset location:")
print(RAW_ROOT.resolve())
