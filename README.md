# UFCOD: Unified Few-shot Cross-domain OOD Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **"From Density to Geometry: Few-Shot OOD Detection via Diffusion Trajectory Energy"**

## Overview

UFCOD is a unified framework for **few-shot cross-domain out-of-distribution (OOD) detection** that achieves competitive performance with only ~100 in-distribution samples, compared to methods requiring 50k-163k samples.

### Key Features

- **Training-free**: Uses a single pre-trained diffusion model as universal feature extractor
- **Few-shot**: Requires only ~100 ID samples per task
- **Cross-domain**: Generalizes across semantically diverse domains without adaptation
- **Theoretically grounded**: Based on information geometry of diffusion models

### Method

UFCOD extracts **2D energy features** from diffusion trajectories:

1. **Path Energy** ($f_1$): Integrated squared score function magnitude
2. **Dynamics Energy** ($f_2$): Score function smoothness (temporal variation)

These features form a discrete Sobolev norm that captures how reliably a sample interacts with the learned diffusion process.

## Installation

```bash
# Clone the repository
git clone https://github.com/anonymous/ufcod.git
cd ufcod

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Basic Usage

```python
from ufcod import EnergyOODDetector, DiffPathExtractor

# 1. Load pretrained diffusion model and extract features
extractor = DiffPathExtractor(
    model_path="path/to/pretrained_model.pt",
    n_ddim_steps=10,
    device="cuda"
)

# 2. Extract 2D energy features from images
train_features = extractor.extract_2d_features(train_images)
test_features = extractor.extract_2d_features(test_images)

# 3. Fit OOD detector on few-shot ID samples
detector = EnergyOODDetector(T=0.5, k=10)
detector.fit(train_features, method='facility_location', n_samples=100)

# 4. Score test samples
scores = detector.score_samples(test_features)
# Higher scores = more likely in-distribution
```

### Evaluation

```python
from ufcod.detectors import compute_auroc

# Compute AUROC
id_scores = detector.score_samples(id_test_features)
ood_scores = detector.score_samples(ood_test_features)
auroc = compute_auroc(id_scores, ood_scores)
print(f"AUROC: {auroc:.4f}")
```

## Project Structure

```
Ready_code/
├── ufcod/                      # Main package
│   ├── __init__.py
│   ├── models/                 # Diffusion models and feature extraction
│   │   ├── feature_extractor.py    # DiffPath feature extractor
│   │   └── improved_diffusion/     # Diffusion model implementation
│   ├── detectors/              # OOD detection methods
│   │   ├── energy_detector.py      # Main UFCOD detector
│   │   └── fewshot_detector.py     # GMM-based detector
│   └── utils/                  # Utilities
│       ├── scoring.py              # Temperature-scaled scoring
│       └── coreset.py              # Coreset selection methods
├── scripts/                    # Training and evaluation scripts
├── configs/                    # Configuration files
├── examples/                   # Example notebooks and scripts
└── tests/                      # Unit tests
```

## Pretrained Models

We provide pretrained diffusion models:

| Model | Dataset | Resolution | Download |
|-------|---------|------------|----------|
| DDPM | CelebA | 32×32 | [link]() |
| DDPM | CelebA | 64×64 | [link]() |

## Results

### Main Results (100 ID samples)

| Method | C10→SVHN | C10→CelebA | SVHN→C10 | CelebA→C10 | Average |
|--------|----------|------------|----------|------------|---------|
| Full-data baselines | ~58.5% | ~58.5% | ~95.5% | ~99.8% | ~58.5% |
| **UFCOD (Ours)** | **95.1%** | **96.5%** | **97.3%** | **99.5%** | **93.7%** |

### Sample Efficiency

- 100 samples achieve 97% of full-data performance
- ~500× reduction in data requirements

## Configuration

### Detector Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `T` | 0.5 | Temperature for soft-min scoring |
| `k` | 10 | Number of nearest neighbors |
| `n_samples` | 100 | Reference set size |
| `method` | 'facility_location' | Sample selection method |

### Feature Extractor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_ddim_steps` | 10 | Number of DDIM sampling steps |
| `image_size` | 32 | Input image resolution |

## Citation

```bibtex
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This work builds upon [improved-diffusion](https://github.com/openai/improved-diffusion) by OpenAI
- We thank the authors of [DiffPath](https://github.com/clear-nus/diffpath) for their foundational work
