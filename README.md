# YOLO-Assisted Yemeni Currency Recognition

A two-stage deep learning framework for Yemeni banknote recognition, combining YOLOv8-assisted ROI detection with EfficientNetV2S classification.

## Overview

The system follows a two-stage pipeline:

1. **Banknote Detection (YOLOv8)**: Automatic extraction of Region of Interest (ROI) to reduce background influence.
2. **Currency Classification (EfficientNetV2S)**: Fine-tuned model for denomination recognition.

The framework is evaluated on Yemeni banknotes and validated on Indian-Thai banknotes for cross-currency generalization.

## Key Contributions

- YOLO-assisted ROI extraction to reduce background bias
- Fair RAW vs ROI comparison under identical settings
- Background Bias Coefficient (BBC) metric for bias quantification
- Cross-currency generalization via zero-shot YOLO transfer
- Grad-CAM visualizations for model interpretability

## Results

### Yemeni Banknotes (6 classes, 1,691 images)

| Model | CV Accuracy | Test Accuracy | BBC |
|-------|-------------|---------------|-----|
| RAW | 99.72% ± 0.26% | 100% | 76.77% |
| ROI | 99.51% ± 0.36% | 99.61% | 79.84% |

### Cross-Currency (15 classes, 2,800 images)

| Dataset | YOLO Mode | Test Accuracy |
|---------|-----------|---------------|
| Indian + Thai | Zero-shot | 99.76% |

## Repository Structure

```
YOLO-Assisted-Yemeni-Currency-Recognition/
├── api/
│   └── main.py
├── src/
│   ├── data/
│   │   ├── download_yemeni.py
│   │   ├── download_indian_thai.py
│   │   ├── split_cv.py
│   │   └── build_roi.py
│   ├── train/
│   │   └── train.py
│   ├── eval/
│   │   ├── bias_ablation.py
│   │   └── test_eval.py
│   └── inference/
│       ├── predictor.py
│       ├── preprocess.py
│       └── utils.py
├── notebooks/
│   ├── Yemeni_Training.ipynb
│   └── Indian_Thai_Training.ipynb
├── models/                      
├── run_inference.py
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Installation

```bash
git clone https://github.com/majedmolhi/YOLO-Assisted-Yemeni-Currency-Recognition.git
cd YOLO-Assisted-Yemeni-Currency-Recognition
pip install -r requirements.txt
```

## Usage

### Reproduce Experiments

The notebooks are self-contained and ready to run. Dataset download scripts are included with Mendeley Data links.

**Yemeni Currency (Main Experiment):**
> Open and run: [`notebooks/Yemeni_Training.ipynb`](notebooks/Yemeni_Training.ipynb)

**Cross-Currency Validation:**
> Open and run: [`notebooks/Indian_Thai_Training.ipynb`](notebooks/Indian_Thai_Training.ipynb)

### Inference with Pretrained Models

Download models from [Zenodo](https://doi.org/10.5281/zenodo.18071819) and place in `models/` directory.

```bash
python run_inference.py --image path/to/banknote.jpg
```

### API Deployment

```bash
nohup python api/main.py > server.log 2>&1 &
```

```bash
curl -X POST "http://127.0.0.1:8085/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@banknote.jpg"
```

## Datasets

**Yemeni Currency Dataset**
- Classes: 6 (50, 100, 200, 250, 500, 1000 YER)
- Images: 1,691
- Source: [Mendeley Data](https://doi.org/10.17632/s56nbwsytx.2)

**Indian-Thai Currency Dataset**
- Classes: 15 (10 Indian + 5 Thai)
- Images: 2,800
- Source: [Mendeley Data](https://doi.org/10.17632/2kfz5yc7pt.1)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Input Size | 224 × 224 |
| Optimizer | AdamW (lr=3×10⁻⁴) |
| Batch Size | 32 |
| Max Epochs | 10 |
| Validation | 5-Fold Stratified CV |
| Test Split | 15% |

## Models

Trained models available on Zenodo: [https://doi.org/10.5281/zenodo.18071819](https://doi.org/10.5281/zenodo.18071819)

## Citation

> Molhi, M. (2025). YOLOv8-Assisted Detection and EfficientNetV2S Classification for Robust Yemeni Banknote Recognition (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.18071819

## License

MIT License

