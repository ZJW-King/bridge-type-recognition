# Bridge Type Recognition with EfficientNetB4

A deep learning pipeline for automated classification of bridge structural types — **None / Beam / Arch / Cable-stayed / Suspension** — built with PyTorch and EfficientNetB4.

[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![EfficientNet](https://img.shields.io/badge/Backbone-EfficientNetB4-0B7285)](https://arxiv.org/abs/1905.11946)
[![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-lightgrey)](#-license)

---

## 📌 Overview

This repository implements an end-to-end image classification workflow that recognizes the structural type of a bridge from a single RGB image. The model is based on **EfficientNet-B4**, fine-tuned on a custom dataset of bridge photographs, and supports both **training from scratch (ImageNet pretrained)** and **detailed inference reporting**, including per-class probability distributions and confidence statistics.

&gt; **Input**: a bridge image (`.jpg` for training, `.png` for inference)
&gt; **Output**: predicted structural type + per-class softmax probabilities

## 🏗️ Model Architecture

| Component | Specification |
|---|---|
| Backbone | EfficientNet-B4 (ImageNet pretrained) |
| Input resolution | 380 × 380 px |
| Classifier head | Dropout (p = 0.4) → Linear (1792 → 5) |
| Number of classes | 5 (None, beam, arch, cable, suspension) |

## 📂 Dataset Structure

```
data/
├── train/
│   ├── None/
│   ├── beam/
│   ├── arch/
│   ├── cable/
│   └── suspension/          # .jpg files
└── test/
    └── *.png                # unlabeled test images
```

The training set is automatically split **90 / 10** into train and validation subsets.
<img width="1096" height="710" alt="image" src="https://github.com/user-attachments/assets/5a116bc4-ea38-452f-a574-585569c72dd9" />
<img width="1091" height="722" alt="image" src="https://github.com/user-attachments/assets/bb1cf51a-9427-40cf-b228-719a78e3eafd" />
<img width="1098" height="723" alt="image" src="https://github.com/user-attachments/assets/d88c8dd0-953b-429e-85f2-93be5222532b" />
<img width="1101" height="816" alt="image" src="https://github.com/user-attachments/assets/964803b7-e089-46dd-bd61-1d13681d88fa" />
<img width="1087" height="718" alt="image" src="https://github.com/user-attachments/assets/6a35d228-f858-4283-8a16-e379c12faed4" />

## 🔧 Data Preprocessing

**Training** — resize to 380 × 380, random horizontal flip (p = 0.5), random rotation (±10°), color jitter (brightness / contrast / saturation 0.2, hue 0.1), normalization with ImageNet statistics.

**Inference** — resize to 380 × 380 + ImageNet normalization only.

## 🚀 Getting Started

### Requirements

- Python ≥ 3.8
- PyTorch (with CUDA recommended)
- torchvision, tqdm, numpy, seaborn, matplotlib, scikit-learn, Pillow

```bash
pip install torch torchvision tqdm numpy seaborn matplotlib scikit-learn pillow
```

### Training

```bash
python bridge_classifier.py
```

Training configuration:

| Hyperparameter | Value |
|---|---|
| Loss | Cross-Entropy Loss |
| Optimizer | Adam (lr = 0.01, weight_decay = 1e-4) |
| Scheduler | StepLR (step_size = 10, gamma = 0.1) |
| Batch size | 8 |
| Default epochs | 50 (adjustable) |

The best checkpoint (by validation accuracy) is saved as `best_bridge_classifier.pth`, and training curves are exported to `training_curves.png`.

### Inference & Evaluation

The script automatically reloads the best checkpoint and performs detailed inference on the test set, producing:

- **Sample predictions grid** — `test_results/sample_predictions.png`
- **Confidence distribution histogram** — `test_results/confidence_distribution.png`
- **Full prediction report** — `test_results/detailed_results.txt`

```
📊 Test Output Example
================================================================================
 1. bridge_001.png       -&gt; arch        (Confidence: 98.42%)
 2. bridge_002.png       -&gt; suspension  (Confidence: 96.17%)
...
Class Distribution:
  None       :  6 images (20.0%)
  beam       :  8 images (26.7%)
  arch       :  7 images (23.3%)
  cable      :  4 images (13.3%)
  suspension :  5 images (16.7%)
```

## 📁 Project Structure

```
├── bridge_classifier.py    # main training & inference script
├── data/
│   ├── train/              # labeled training images
│   └── test/               # unlabeled test images (.png)
├── best_bridge_classifier.pth
├── training_curves.png
└── test_results/
    ├── sample_predictions.png
    ├── confidence_distribution.png
    └── detailed_results.txt
```

## 👥 Authors

| Name | Affiliation |
|---|---|
| **Jianwei Zhang** | College of Civil Engineering, Tongji University |
| **Chao Yu** | College of Civil Engineering, Tongji University |
| **Xiya Liu** | College of Civil Engineering, Tongji University |

## 📜 License & Copyright

**Copyright © 2025 Jianwei Zhang, Chao Yu, Xiya Liu. All rights reserved.**

This project, including all source code, documentation, and model artifacts, is the intellectual property of the authors listed above. Unauthorized copying, modification, distribution, or use of this work, in whole or in part, for commercial or non-commercial purposes, is strictly prohibited without prior written permission from the copyright holders.

This project uses **PyTorch** and **torchvision** (BSD-style license) and a **pretrained EfficientNet-B4** backbone from torchvision, whose weights are derived from the ImageNet dataset and subject to their respective licenses.

&gt; ⚖️ **Note to repository maintainers**: if you intend to release this work under an open-source license (e.g., MIT), replace this section accordingly and include a `LICENSE` file. The MIT License text is reproduced below for convenience — it is **not** in effect unless explicitly adopted.

## 🙏 Acknowledgements

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) — Tan & Le, ICML 2019
- [torchvision pretrained models](https://pytorch.org/vision/stable/models.html)
