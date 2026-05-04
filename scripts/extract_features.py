#!/usr/bin/env python3
"""
Feature Extraction Script for UFCOD

Extracts 2D energy features from image datasets using a pretrained diffusion model.
"""

import sys
from pathlib import Path

# Add parent directory to path for development usage
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
from datetime import datetime

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataset(name, root, split='train', image_size=32):
    """Load dataset with appropriate transforms."""

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if name == 'cifar10':
        dataset = datasets.CIFAR10(root, train=(split == 'train'),
                                   download=True, transform=transform)
    elif name == 'cifar100':
        dataset = datasets.CIFAR100(root, train=(split == 'train'),
                                    download=True, transform=transform)
    elif name == 'svhn':
        svhn_split = 'train' if split == 'train' else 'test'
        dataset = datasets.SVHN(root, split=svhn_split,
                               download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description='Extract UFCOD features')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to pretrained diffusion model')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to model config YAML')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cifar10', 'cifar100', 'svhn'],
                       help='Dataset name')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'test'],
                       help='Dataset split')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Data root directory')
    parser.add_argument('--output_dir', type=str, default='./features',
                       help='Output directory for features')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for feature extraction')
    parser.add_argument('--n_ddim_steps', type=int, default=10,
                       help='Number of DDIM steps')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    args = parser.parse_args()

    print("=" * 60)
    print("UFCOD Feature Extraction")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"Model: {args.model_path}")
    print()

    # Import here to avoid loading torch before parsing args
    from ufcod.models import DiffPathExtractor

    # Load dataset
    print("Loading dataset...")
    dataset = get_dataset(args.dataset, args.data_root, args.split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4)
    print(f"Dataset size: {len(dataset)}")

    # Initialize feature extractor
    print("\nInitializing feature extractor...")
    extractor = DiffPathExtractor(
        model_path=args.model_path,
        config_path=args.config_path,
        n_ddim_steps=args.n_ddim_steps,
        device=args.device
    )

    # Extract features
    print("\nExtracting features...")
    features = extractor.extract_features_from_dataloader(dataloader, verbose=True)
    print(f"Features shape: {features.shape}")

    # Save features
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.dataset}_{args.split}_features.npz"

    np.savez(output_file,
             features=features,
             dataset=args.dataset,
             split=args.split,
             n_ddim_steps=args.n_ddim_steps)

    print(f"\nFeatures saved to: {output_file}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
