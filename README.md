# Pneumonia Detection from Chest X-Ray Images

A deep learning project (Computer Vision — Medical Sector) that classifies
chest X-ray images as **NORMAL** or **PNEUMONIA**, built as a full
engineering pipeline: data preprocessing, a baseline CNN, transfer
learning with DenseNet121, hyperparameter tuning, and evaluation with
error analysis.

See `architecture_diagram.png` for the model architecture, and
`Pneumonia_CV_Project_Report_EN.docx` / `..._Themed_EN.pdf` for the full
write-up.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── architecture_diagram.png
├── train.py                     # trains the baseline CNN and/or DenseNet121 model
├── evaluate.py                  # evaluates a trained model on the test set
└── src/
    ├── __init__.py
    ├── data_preprocessing.py    # data generators (resize, rescale, augmentation)
    └── model.py                 # baseline CNN and DenseNet121 model builders
```

## Dataset

**Chest X-Ray Images (Pneumonia)** — Kermany / Mooney dataset, Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

- 5,848 total images (1,575 NORMAL + 4,273 PNEUMONIA)
- Provided as `train/`, `val/`, `test/` folders, each with `NORMAL/` and
  `PNEUMONIA/` subfolders
- Class imbalance: ~2.9:1 (PNEUMONIA:NORMAL) in the training set

The dataset is **not included** in this repository (it's ~1.2 GB). Download
it manually from the Kaggle link above and unzip it, or use the
`--download` flag on `train.py` (requires a Kaggle API token — see below).

### Kaggle API setup (optional, for `--download`)

1. Go to your Kaggle account settings → "Create New API Token" → downloads `kaggle.json`
2. Place it at `~/.kaggle/kaggle.json` (Linux/Mac) and run `chmod 600 ~/.kaggle/kaggle.json`

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Train

Train both models (baseline CNN + DenseNet121 transfer learning):

```bash
python train.py --data_dir /path/to/chest_xray --epochs 10 --model both
```

Train only the main model, with fine-tuning of the last 20 DenseNet121 layers:

```bash
python train.py --data_dir /path/to/chest_xray --model densenet \
                 --fine_tune_last_n 20 --learning_rate 0.00001
```

Download the dataset automatically (requires Kaggle API token) and train:

```bash
python train.py --data_dir ./chest_xray_data/chest_xray --download --model both
```

This saves `baseline_model.h5`, `densenet_model.h5`, and their training
curve plots into `./outputs/` (or wherever `--out_dir` points).

### 2. Evaluate

```bash
python evaluate.py --model_path outputs/densenet_model.h5 \
                    --data_dir /path/to/chest_xray --out_dir outputs
```

This prints and saves:
- `classification_report.txt` — precision, recall, F1-score per class
- `confusion_matrix.png` — confusion matrix heatmap
- `False_Positives.png`, `False_Negatives.png` — example misclassified
  images with the model's confidence score, for error analysis

## Model Architecture

Two models are compared (see `architecture_diagram.png`):

1. **Baseline CNN** (trained from scratch): `Conv2D(32) → MaxPool →
   Conv2D(64) → MaxPool → Flatten → Dense(64) → Dropout(0.5) →
   Dense(1, sigmoid)`
2. **Main model — DenseNet121 (transfer learning)**: frozen ImageNet-pretrained
   convolutional base → `GlobalAveragePooling2D → Dense(128, relu) →
   Dropout(0.5) → Dense(1, sigmoid)`

## Results Summary

| Model | Train Accuracy | Val Accuracy |
|---|---|---|
| Baseline CNN | 93.6% | 68.75% |
| DenseNet121 (main model) | 94.7% | 87.5% |

Final test-set evaluation (DenseNet121): **82.9% accuracy**, with
**98% recall on PNEUMONIA** (only 7 False Negatives out of 390 cases) —
prioritizing patient safety by rarely missing a true pneumonia case, at
the cost of a higher False Positive rate (100 of 624, 16.0%), consistent
with the class imbalance observed during EDA.

See the full project report for the complete evaluation, hyperparameter
experiments, and error analysis.

## Limitations & Future Work

- Dataset is from a single medical institution — generalization to other
  hospitals/equipment is untested.
- Binary classification only (does not distinguish viral vs. bacterial
  pneumonia).
- Class imbalance was mitigated with augmentation but not fully resolved.
- Future work: external dataset validation, multi-class classification,
  and Grad-CAM explainability.

## Tools & Libraries

- **TensorFlow / Keras** — model building, training, `ImageDataGenerator`
- **DenseNet121** (Keras Applications, ImageNet weights) — transfer learning backbone
- **scikit-learn** — classification report, confusion matrix
- **Matplotlib / Seaborn** — training curves, confusion matrix heatmap, error analysis visualizations
- **Google Colab** — training environment (free GPU)
