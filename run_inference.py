
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import argparse
from src.inference.predictor import YemeniCurrencyPredictor


def main():
    parser = argparse.ArgumentParser(
        description="Yemeni Currency Recognition - End-to-End Inference"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to local image"
    )
    args = parser.parse_args()

    predictor = YemeniCurrencyPredictor(
        model_path="models/yemeni_roi_best.keras"
    )

    result = predictor.predict(args.image)

    # ====== CLEAN OUTPUT ======
    print(result)


if __name__ == "__main__":
    main()
