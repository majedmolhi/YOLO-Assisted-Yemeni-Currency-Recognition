import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from fastapi import FastAPI, File, UploadFile, HTTPException
from pathlib import Path
import shutil
import uvicorn

from src.inference.predictor import YemeniCurrencyPredictor

# ================= CONFIG =================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "yemeni_roi_best.keras"
TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)

PORT = 8085

# ================= APP =================
api = FastAPI(title="Yemeni Currency Recognition API")

print(f"[INFO] Loading model: {MODEL_PATH}")
predictor = YemeniCurrencyPredictor(str(MODEL_PATH))

# ================= GET (health check) =================
@api.get("/")
def root():
    return {
        "message": "Yemeni Currency API is running",
        "status": "ok"
    }

# ================= POST (prediction) =================
@api.post("/predict")
def predict(file: UploadFile = File(...)):
    img_path = TMP_DIR / file.filename

    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = predictor.predict(str(img_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        img_path.unlink(missing_ok=True)

    return result

# ================= RUN =================
def run():
    uvicorn.run(
        api,
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )

if __name__ == "__main__":
    run()
