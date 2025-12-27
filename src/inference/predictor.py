
import tensorflow as tf
from pathlib import Path

from src.inference.preprocess import load_and_preprocess
from src.inference.utils import decode_prediction


class YemeniCurrencyPredictor:
    """
    End-to-End inference using the final released ROI model.
    """

    def __init__(self, model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"[INFO] Loading model: {model_path}")
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, image_path):
        img = load_and_preprocess(image_path)
        preds = self.model.predict(img, verbose=0)[0]

        label, conf = decode_prediction(preds)

        return {
            "currency": label,
            "confidence": round(conf, 4)
        }
