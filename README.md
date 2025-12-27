\# Yemeni Currency Recognition using YOLO-assisted CNN



This repository presents a \*\*complete, reproducible, and academically structured pipeline\*\*

for Yemeni paper currency recognition using a \*\*YOLO-assisted ROI-based classification\*\*

framework built on \*\*EfficientNetV2S\*\*.



The project is designed for \*\*research reproducibility\*\*, \*\*fair experimental comparison\*\*,

\*\*cross-currency generalization analysis\*\*, and \*\*local deployment\*\*.



---



\## Overview



The proposed system follows a two-stage pipeline:



1\. \*\*Banknote Detection (YOLO)\*\*  

&nbsp;  Automatic extraction of the Region of Interest (ROI) to suppress background influence.



2\. \*\*Currency Classification (CNN)\*\*  

&nbsp;  Fine-tuned EfficientNetV2S model for Yemeni banknote denomination recognition.



The framework is evaluated on Yemeni banknotes (RAW vs ROI) and further tested on

Indian and Thai banknotes to assess cross-currency generalization.



---



\## Key Contributions



\- YOLO-assisted ROI extraction to reduce background bias  

\- Fair RAW vs ROI comparison under identical experimental settings  

\- Bias ablation analysis for robustness verification  

\- Grad-CAM visual explanations for interpretability  

\- Cross-currency generalization evaluation without retraining the detector  

\- Fully reproducible and modular research pipeline  



---



\## Repository Structure



```text

Yemeni-Currency-Recognition/

│

├── api/                    # FastAPI deployment

│   └── main.py

│

├── src/

│   ├── data/               # Dataset download \& preprocessing

│   ├── train/              # Training \& cross-validation

│   ├── eval/               # Bias ablation \& evaluation

│   └── inference/          # Prediction utilities

│

├── models/                 # Model pointers (no binaries stored)

│   └── README.md

│

├── notebooks/              # Reproducible experiments

│   ├── Yemeni\_Currency\_Recognition.ipynb

│   └── Indian\_Thai\_Generalization.ipynb

│

├── run\_inference.py        # CLI inference

├── requirements.txt

├── CITATION.cff

├── LICENSE

└── README.md

```



---



\## Datasets



\* \*\*Yemeni Currency Dataset\*\*

&nbsp; Denominations: 50, 100, 200, 250, 500, 1000

&nbsp; Publicly available via \*\*Mendeley Data\*\*



\* \*\*Indian \& Thai Currency Dataset\*\*

&nbsp; Used exclusively for cross-currency generalization experiments



> Datasets are \*\*not included\*\* in this repository.



---



\## Environment Setup



```bash

pip install -r requirements.txt

```



Tested with:



\* Python 3.10+

\* TensorFlow 2.16.1

\* Ultralytics YOLOv8



---



\## Training \& Evaluation



\### Train models (RAW or ROI)



```bash

python src/train/train.py

```



\### Bias Ablation Analysis



```bash

python src/eval/bias\_ablation.py

```



All experiments are fully reproducible using the provided scripts and notebooks.



---



\## Inference



\### Command-Line Interface (CLI)



```bash

python run\_inference.py --image path/to/image.jpg

```



\### REST API (Local Deployment)



A RESTful API is provided for \*\*local deployment and reproducibility purposes\*\*.



\#### Model Setup



Download the trained Yemeni ROI model from Zenodo and place it at:



```text

models/yemeni\_roi\_best.keras

```



\#### Run API



```bash

python api/main.py

```



\#### Endpoints



\* `GET /` : Health check

\* `POST /predict` : Currency prediction



Example request:



```bash

curl -X POST "http://127.0.0.1:8085/predict" \\

&nbsp; -H "accept: application/json" \\

&nbsp; -H "Content-Type: multipart/form-data" \\

&nbsp; -F "file=@image.jpg"

```



> The API is intended for research demonstration and reproducibility,

> not as a publicly hosted service.



---



\## Results Summary



\* Near-perfect accuracy on Yemeni banknotes

\* ROI-based models exhibit lower background dependency

\* Strong generalization performance on Indian \& Thai banknotes

\* Grad-CAM visualizations confirm focus on discriminative regions



Detailed quantitative results and visual analyses are provided in the notebooks.



---



\## Models \& Weights



All trained models and YOLO weights are archived on \*\*Zenodo\*\*.



DOI: \*\*(to be added after Zenodo release)\*\*



---



\## Code \& Data Availability



\* \*\*Code\*\*: GitHub (this repository)

\* \*\*Models\*\*: Zenodo

\* \*\*Datasets\*\*: Mendeley Data



---



\## Citation



If you use this work, please cite it using the metadata provided in `CITATION.cff`.



---



\## License



This project is released under the \*\*MIT License\*\*.





