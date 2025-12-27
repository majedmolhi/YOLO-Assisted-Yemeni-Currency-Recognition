"""
Download Yemeni Currency Dataset from Mendeley
and extract it into a standard folder structure.

Output:
    data_raw/Yemeni/RAW/{50,100,200,250,500,1000}
"""

import zipfile
import shutil
import urllib.request
from pathlib import Path

#
DATA_URL = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/s56nbwsytx-2.zip"
DOWNLOAD_PATH = Path("/content/yemeni_dataset.zip")

DATA_ROOT = Path("/content/Yemeni_data")
FINAL_RAW_DIR = DATA_ROOT / "RAW"

#
print("Downloading Yemeni dataset...")
urllib.request.urlretrieve(DATA_URL, DOWNLOAD_PATH)
print("Download completed.")

#
print("Extracting dataset...")
with zipfile.ZipFile(DOWNLOAD_PATH, "r") as z:
    z.extractall("/content/temp_yemeni")

# ===================== LOCATE RAW FOLDER
temp_root = Path("/content/temp_yemeni")

raw_dir = list(temp_root.rglob("RAW"))
if not raw_dir:
    raise RuntimeError("RAW folder not found in extracted dataset")

raw_dir = raw_dir[0]

# ===================== MOVE TO FINAL LOCATION
if FINAL_RAW_DIR.exists():
    shutil.rmtree(FINAL_RAW_DIR)

FINAL_RAW_DIR.mkdir(parents=True, exist_ok=True)

for cls in raw_dir.iterdir():
    if cls.is_dir():
        shutil.copytree(cls, FINAL_RAW_DIR / cls.name)

# ===================== CLEANUP
DOWNLOAD_PATH.unlink()
shutil.rmtree(temp_root)

print("Yemeni dataset ready at:")
print(FINAL_RAW_DIR.resolve())
