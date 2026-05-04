"""
Data loading utilities for UFCOD.

Provides dataset loading functions for common OOD detection benchmarks.
"""

import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_transform(image_size=32, normalize=True):
    """
    Get standard image transform.

    Args:
        image_size: Target image size
        normalize: Whether to normalize to [-1, 1]

    Returns:
        torchvision.transforms.Compose
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]

    if normalize:
        transform_list.append(
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )

    return transforms.Compose(transform_list)


def load_dataset(name, root='./data', split='train', image_size=32, download=True):
    """
    Load a dataset by name.

    Args:
        name: Dataset name ('cifar10', 'cifar100', 'svhn', 'mnist', 'fashionmnist')
        root: Data root directory
        split: 'train' or 'test'
        image_size: Target image size
        download: Whether to download if not present

    Returns:
        torch.utils.data.Dataset
    """
    transform = get_transform(image_size)
    root = Path(root)

    if name == 'cifar10':
        dataset = datasets.CIFAR10(
            root, train=(split == 'train'),
            download=download, transform=transform
        )
    elif name == 'cifar100':
        dataset = datasets.CIFAR100(
            root, train=(split == 'train'),
            download=download, transform=transform
        )
    elif name == 'svhn':
        svhn_split = 'train' if split == 'train' else 'test'
        dataset = datasets.SVHN(
            root, split=svhn_split,
            download=download, transform=transform
        )
    elif name == 'mnist':
        # Convert grayscale to RGB
        mnist_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.MNIST(
            root, train=(split == 'train'),
            download=download, transform=mnist_transform
        )
    elif name == 'fashionmnist':
        fmnist_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.FashionMNIST(
            root, train=(split == 'train'),
            download=download, transform=fmnist_transform
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return dataset


def create_dataloader(dataset, batch_size=32, shuffle=False, num_workers=4):
    """
    Create a DataLoader from a dataset.

    Args:
        dataset: torch.utils.data.Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers

    Returns:
        torch.utils.data.DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def subsample_dataset(dataset, n_samples, random_state=42):
    """
    Randomly subsample a dataset.

    Args:
        dataset: torch.utils.data.Dataset
        n_samples: Number of samples to select
        random_state: Random seed

    Returns:
        torch.utils.data.Subset
    """
    np.random.seed(random_state)
    n_total = len(dataset)

    if n_samples >= n_total:
        return dataset

    indices = np.random.choice(n_total, n_samples, replace=False)
    return Subset(dataset, indices)


def load_features(path):
    """
    Load precomputed features from .npz file.

    Args:
        path: Path to .npz file

    Returns:
        numpy array of features
    """
    data = np.load(path)
    return data['features']


def save_features(features, path, **metadata):
    """
    Save features to .npz file.

    Args:
        features: numpy array of features
        path: Output path
        **metadata: Additional metadata to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, features=features, **metadata)
