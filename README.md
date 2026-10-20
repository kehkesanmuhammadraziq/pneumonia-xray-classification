# Pneumonia Classification from Chest X-Ray Images

A computer-vision project comparing classical machine learning, a custom convolutional neural network (CNN), pretrained CNN architectures, and a Vision Transformer (ViT) for binary classification of chest X-ray images as **NORMAL** or **PNEUMONIA**.

> **Research/learning project only.** This notebook is not a medical device and must not be used to diagnose, screen, or make treatment decisions for patients.

## Project overview

The notebook explores six approaches:

- HOG feature extraction + Support Vector Machine (SVM)
- Custom CNN
- VGG16
- EfficientNet-B0
- ResNet18
- ViT-Base/16 (`vit_base_patch16_224`)

It uses PyTorch/Torchvision for deep learning, `efficientnet_pytorch` for EfficientNet-B0, `timm` for ViT, and scikit-learn for evaluation. The notebook includes image exploration, training/validation loops, test-set evaluation, learning curves, confusion matrices, and model comparison by accuracy and training time.

## Dataset

The notebook references the **Chest X-Ray Images (Pneumonia)** dataset hosted on Kaggle:

- Dataset page: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Download the dataset separately and review the dataset page's terms and license before use. Do not commit the dataset or patient images to this repository.

The notebook expects this directory layout:

```text
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The original notebook sets `data_dir = '../data'`, so run it from the `notebooks/` directory with the dataset in the repository root's `data/` directory, or update `data_dir` to your local dataset path.

## Environment and setup

Recommended: Python 3.10 or 3.11, with a compatible PyTorch build for your hardware. A CUDA-capable NVIDIA GPU can substantially reduce training time; CPU execution may be slow.

1. Create and activate a virtual environment.
2. Install the packages listed below, selecting the PyTorch installation command that matches your OS and CUDA/CPU setup from https://pytorch.org/get-started/locally/.
3. Download and extract the dataset into the structure shown above.
4. Open `notebooks/pneumonia-classifier.ipynb` in Jupyter Notebook or JupyterLab.
5. Run the notebook cells in order. Pretrained model weights may be downloaded automatically by the libraries.

Example environment setup:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the PyTorch installation from `requirements.txt` is incompatible with your machine, install the appropriate PyTorch/Torchvision build from the official PyTorch selector, then install the remaining dependencies.

## Dependencies

See [`requirements.txt`](requirements.txt). Versions are intentionally not fully pinned because PyTorch/Torchvision installation varies by operating system and CUDA support.

## Results recorded in the notebook

The saved notebook outputs show the following test accuracies from one recorded run:

| Approach | Test accuracy |
|---|---:|
| HOG + SVM | 75.32% |
| Custom CNN | 80.13% |
| VGG16 | 79.01% |
| EfficientNet-B0 | 80.45% |
| ResNet18 | 80.29% |
| ViT-Base/16 | 85.74% |

These are the notebook's existing recorded outputs, not a guarantee that a fresh run will reproduce them. Results can vary with random initialization, package versions, hardware, and data preparation.

## Important limitations and next steps

- The notebook's recorded validation split contains only **16 images**, which is too small for reliable model selection. Use a properly sized validation set and document the split methodology.
- Accuracy alone is not sufficient for a medical-image classification task. Report sensitivity/recall for pneumonia, specificity, precision, F1-score, ROC-AUC, and confidence intervals where appropriate.
- Inspect for data leakage, duplicates, class imbalance, and differences in image sources before interpreting results.
- Some confusion-matrix values in the notebook are manually specified in the plotting section. Recompute all matrices from the current predictions rather than relying on hard-coded values.
- The notebook saves model weights to relative paths under `../model/` and one plot under `../content/plots/`; create those directories or update the save paths before running those cells.
- This is an exploratory educational project, not a clinically validated model.

## Repository structure

```text
pneumonia-classifier/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── pneumonia-classifier.ipynb
```

## Suggested future improvements

1. Make paths configurable and remove machine-specific assumptions.
2. Add a reproducible random seed and record package versions.
3. Replace the tiny validation set with a suitable validation split.
4. Generate confusion matrices directly from predictions.
5. Add sensitivity, specificity, precision, recall, F1-score, and ROC-AUC.
6. Add a clean, reproducible inference example and tests.
7. Document the dataset license and intended-use limitations.

## Author

Add your name and a short professional description here if you want them on the public repository.
