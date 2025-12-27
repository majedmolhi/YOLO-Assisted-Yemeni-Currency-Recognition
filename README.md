# Yemeni Currency Recognition using YOLO-assisted CNN

This repository presents a **complete, reproducible, and academically structured pipeline**
for Yemeni paper currency recognition using a **YOLO-assisted ROI-based classification**
framework built on **EfficientNetV2S**.

The project is designed for **research reproducibility**, **fair experimental comparison**,
**cross-currency generalization analysis**, and **local deployment**.

---

## Overview

The proposed system follows a two-stage pipeline:

1. **Banknote Detection (YOLO)**  
   Automatic extraction of the Region of Interest (ROI) to suppress background influence.

2. **Currency Classification (CNN)**  
   Fine-tuned EfficientNetV2S model for Yemeni banknote denomination recognition.

The framework is evaluated on Yemeni banknotes (RAW vs ROI) and further tested on
Indian and Thai banknotes to assess cross-currency generalization.

---

## Key Contributions

- YOLO-assisted ROI extraction to reduce background bias  
- Fair RAW vs ROI comparison under identical experimental settings  
- Bias ablation analysis for robustness verification  
- Grad-CAM visual explanations for interpretability  
- Cross-currency generalization evaluation without retraining the detector  
- Fully reproducible and modular research pipeline  

---

## Repository Structure

```text
Yemeni-Currency-Recognition/
│
├── api/                    # FastAPI deployment
│   └── main.py
│
├── src/
│   ├── data/               # Dataset download & preprocessing
│   ├── train/              # Training & cross-validation
│   ├── eval/               # Bias ablation & evaluation
│   └── inference/          # Prediction utilities
│
├── models/                 # Model placeholders (no binaries stored)
│   └── README.md
│
├── notebooks/              # Reproducible experiments
│   ├── Yemeni_Currency_Recognition.ipynb
│   └── Indian_Thai_Generalization.ipynb
│
├── run_inference.py        # CLI inference
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
Datasets
Yemeni Currency Dataset
Denominations: 50, 100, 200, 250, 500, 1000

Publicly available via Mendeley Data

Official citation:

MOLHI, MAJED (2025), “Yemeni Currency Recognition Dataset”,
Mendeley Data, V2, doi: 10.17632/s56nbwsytx.2
https://doi.org/10.17632/s56nbwsytx.2

Indian & Thai Currency Dataset
Used only for cross-currency generalization experiments

Dataset source: (link to be added)

Datasets are not included in this repository.

Environment Setup
bash
Copy code
pip install -r requirements.txt
Tested with:

Python 3.10+

TensorFlow 2.16.1

Ultralytics YOLOv8

Usage Scenarios
1. Running the Yemeni Experiment (Main Pipeline)
The Yemeni notebook represents the primary experimental pipeline used in this study.
Dataset download and preprocessing are handled automatically via the provided scripts.

Running the notebook reproduces all main results reported in the paper:

text
Copy code
notebooks/Yemeni_Currency_Recognition.ipynb
2. Cross-Currency Evaluation (Indian & Thai)
The Indian–Thai notebook reuses the same pipeline and training configuration
to evaluate cross-currency generalization without retraining the YOLO detector.

text
Copy code
notebooks/Indian_Thai_Generalization.ipynb
3. Inference Without Training (Using Pretrained Model)
To perform inference without training:

Download the pretrained Yemeni ROI model from Zenodo
(placeholder link)
https://zenodo.org/record/XXXXXXXX

Place the model file at:

text
Copy code
models/yemeni_roi_best.keras
Run inference using the CLI:

bash
Copy code
python run_inference.py --image path/to/image.jpg
REST API (Local Deployment)
A RESTful API is provided for local deployment and reproducibility purposes.

Run API
bash
Copy code
python api/main.py
Available Endpoints
GET / – Health check

POST /predict – Currency prediction

Example request:

bash
Copy code
curl -X POST "http://127.0.0.1:8085/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
The API is intended for research demonstration only
and is not designed as a publicly hosted service.

Results Summary
Near-perfect accuracy on Yemeni banknotes

ROI-based models exhibit lower background dependency

Strong generalization performance on Indian & Thai banknotes

Grad-CAM visualizations confirm focus on discriminative regions

Detailed quantitative results and visual analyses are provided in the notebooks.

Models & Weights
All trained models and YOLO weights are archived on Zenodo.

Zenodo Record: (to be added)

DOI: (to be added after Zenodo release)

Code & Data Availability
Code: GitHub (this repository)

Models: Zenodo (link to be added)

Datasets: Mendeley Data (Yemeni dataset publicly available)

Citation
If you use this work, please cite it using the metadata provided in CITATION.cff.

License
This project is released under the MIT License.

yaml
Copy code
