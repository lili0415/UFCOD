#!/usr/bin/env python3
"""
UFCOD Quick Start Example

This script demonstrates the basic usage of UFCOD for OOD detection.
"""

import sys
from pathlib import Path

# Add parent directory to path for development usage
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Import UFCOD components
from ufcod import EnergyOODDetector
from ufcod.detectors import compute_auroc, compute_fpr_at_tpr


def generate_synthetic_data():
    """
    Generate synthetic 2D energy features for demonstration.

    In practice, these would be extracted from images using DiffPathExtractor.
    """
    np.random.seed(42)

    # ID samples: clustered around (4, 2)
    id_train = np.random.randn(1000, 2) * 0.5 + [4, 2]
    id_test = np.random.randn(500, 2) * 0.5 + [4, 2]

    # OOD samples: distributed differently
    ood_test = np.random.randn(500, 2) * 1.0 + [8, 5]

    return id_train, id_test, ood_test


def main():
    print("=" * 60)
    print("UFCOD Quick Start Example")
    print("=" * 60)

    # Generate synthetic data
    print("\n1. Generating synthetic 2D energy features...")
    id_train, id_test, ood_test = generate_synthetic_data()
    print(f"   ID train: {id_train.shape}")
    print(f"   ID test:  {id_test.shape}")
    print(f"   OOD test: {ood_test.shape}")

    # Initialize detector
    print("\n2. Initializing EnergyOODDetector...")
    detector = EnergyOODDetector(T=0.5, k=10)
    print(f"   Temperature: {detector.T}")
    print(f"   K neighbors: {detector.k}")

    # Fit detector with few-shot samples
    print("\n3. Fitting detector with Facility Location coreset selection...")
    detector.fit(
        id_train,
        method='facility_location',
        n_samples=100,
        random_state=42
    )
    print(f"   Reference set size: {len(detector.reference_features)}")

    # Score samples
    print("\n4. Scoring test samples...")
    id_scores = detector.score_samples(id_test, calibrate=True)
    ood_scores = detector.score_samples(ood_test, calibrate=True)
    print(f"   ID scores:  mean={id_scores.mean():.3f}, std={id_scores.std():.3f}")
    print(f"   OOD scores: mean={ood_scores.mean():.3f}, std={ood_scores.std():.3f}")

    # Compute metrics
    print("\n5. Computing evaluation metrics...")
    auroc = compute_auroc(id_scores, ood_scores)
    fpr95 = compute_fpr_at_tpr(id_scores, ood_scores, tpr_threshold=0.95)
    print(f"   AUROC:  {auroc * 100:.2f}%")
    print(f"   FPR95:  {fpr95 * 100:.2f}%")

    # Make predictions with threshold
    print("\n6. Making predictions (threshold=0)...")
    id_preds = detector.predict(id_test, threshold=0.0)
    ood_preds = detector.predict(ood_test, threshold=0.0)
    print(f"   ID accuracy:  {id_preds.mean() * 100:.1f}%")
    print(f"   OOD accuracy: {(1 - ood_preds.mean()) * 100:.1f}%")

    print("\n" + "=" * 60)
    print("Quick start complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
