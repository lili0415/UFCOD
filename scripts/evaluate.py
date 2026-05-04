#!/usr/bin/env python3
"""
Evaluation Script for UFCOD

Evaluates OOD detection performance on ID-OOD dataset pairs.
"""

import sys
from pathlib import Path

# Add parent directory to path for development usage
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
from datetime import datetime

from ufcod import EnergyOODDetector
from ufcod.detectors import compute_auroc, compute_fpr_at_tpr


def load_features(feature_dir, dataset, split):
    """Load precomputed features."""
    path = Path(feature_dir) / f"{dataset}_{split}_features.npz"
    if not path.exists():
        raise FileNotFoundError(f"Features not found: {path}")
    data = np.load(path)
    return data['features']


def run_evaluation(id_dataset, ood_dataset, feature_dir, args):
    """Run evaluation for a single ID-OOD pair."""

    # Load features
    id_train = load_features(feature_dir, id_dataset, 'train')
    id_test = load_features(feature_dir, id_dataset, 'test')
    ood_test = load_features(feature_dir, ood_dataset, 'test')

    # Extract 2D energy features (indices 1 and 4)
    id_train_2d = id_train[:, [1, 4]]
    id_test_2d = id_test[:, [1, 4]]
    ood_test_2d = ood_test[:, [1, 4]]

    results = []
    for seed in args.seeds:
        # Initialize and fit detector
        detector = EnergyOODDetector(T=args.temperature, k=args.k_neighbors)
        detector.fit(id_train_2d, method=args.method,
                    n_samples=args.n_samples, random_state=seed)

        # Score samples
        id_scores = detector.score_samples(id_test_2d, calibrate=True)
        ood_scores = detector.score_samples(ood_test_2d, calibrate=True)

        # Compute metrics
        auroc = compute_auroc(id_scores, ood_scores)
        fpr95 = compute_fpr_at_tpr(id_scores, ood_scores, 0.95)

        results.append({
            'seed': seed,
            'auroc': float(auroc),
            'fpr95': float(fpr95)
        })

    # Aggregate results
    aurocs = [r['auroc'] for r in results]
    fpr95s = [r['fpr95'] for r in results]

    return {
        'id_dataset': id_dataset,
        'ood_dataset': ood_dataset,
        'auroc_mean': float(np.mean(aurocs)),
        'auroc_std': float(np.std(aurocs)),
        'fpr95_mean': float(np.mean(fpr95s)),
        'fpr95_std': float(np.std(fpr95s)),
        'per_seed': results
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate UFCOD')
    parser.add_argument('--feature_dir', type=str, default='./features',
                       help='Directory containing precomputed features')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--id_datasets', type=str, nargs='+',
                       default=['cifar10', 'svhn'],
                       help='In-distribution datasets')
    parser.add_argument('--ood_datasets', type=str, nargs='+',
                       default=['cifar10', 'svhn', 'cifar100'],
                       help='Out-of-distribution datasets')
    parser.add_argument('--temperature', type=float, default=0.5,
                       help='Temperature for soft-min scoring')
    parser.add_argument('--k_neighbors', type=int, default=10,
                       help='Number of nearest neighbors')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of reference samples')
    parser.add_argument('--method', type=str, default='facility_location',
                       choices=['facility_location', 'stratified', 'random'],
                       help='Sample selection method')
    parser.add_argument('--seeds', type=int, nargs='+',
                       default=[42, 123, 456, 789, 1024],
                       help='Random seeds for evaluation')
    args = parser.parse_args()

    print("=" * 60)
    print("UFCOD Evaluation")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Temperature: {args.temperature}")
    print(f"K neighbors: {args.k_neighbors}")
    print(f"N samples: {args.n_samples}")
    print(f"Method: {args.method}")
    print()

    all_results = []

    for id_dataset in args.id_datasets:
        print(f"\nID Dataset: {id_dataset.upper()}")
        print("-" * 40)

        for ood_dataset in args.ood_datasets:
            if ood_dataset == id_dataset:
                continue

            try:
                result = run_evaluation(id_dataset, ood_dataset,
                                       args.feature_dir, args)
                all_results.append(result)

                print(f"  vs {ood_dataset:12s}: "
                      f"AUROC={result['auroc_mean']*100:.2f}% "
                      f"(±{result['auroc_std']*100:.2f}%), "
                      f"FPR95={result['fpr95_mean']*100:.2f}%")
            except FileNotFoundError as e:
                print(f"  vs {ood_dataset:12s}: SKIPPED ({e})")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'evaluation_results.json'

    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'temperature': args.temperature,
                'k_neighbors': args.k_neighbors,
                'n_samples': args.n_samples,
                'method': args.method,
                'seeds': args.seeds
            },
            'results': all_results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    avg_auroc = np.mean([r['auroc_mean'] for r in all_results])
    print(f"Average AUROC: {avg_auroc*100:.2f}%")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
