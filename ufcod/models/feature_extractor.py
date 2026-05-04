"""
DiffPath Feature Extractor for UFCOD.

Extracts 6D statistics from diffusion path for OOD detection:
- Path Energy features: eps_sum, eps_sum_sq, eps_sum_cb
- Dynamics Energy features: deps_dt, deps_dt_sq, deps_dt_cb
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import yaml
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from .improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)


class SimpleImageDataset(Dataset):
    """Simple dataset wrapper for images."""

    def __init__(self, images):
        """
        Args:
            images: numpy array of shape (N, H, W, C) or (N, C, H, W) or list of PIL images
        """
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[2] == 3:
                # (H, W, C) -> (C, H, W)
                img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float()
            # Normalize to [-1, 1]
            if img.max() > 1:
                img = img / 255.0
            img = img * 2 - 1
        return img


class DiffPathExtractor:
    """
    DiffPath feature extractor using diffusion models.

    Extracts 6D energy features from the diffusion trajectory:
    - f1 (eps_sum): Sum of noise predictions
    - f2 (eps_sum_sq): Path Energy - sum of squared noise predictions
    - f3 (eps_sum_cb): Sum of cubed noise predictions
    - f4 (deps_dt): Sum of temporal derivatives
    - f5 (deps_dt_sq): Dynamics Energy - sum of squared temporal derivatives
    - f6 (deps_dt_cb): Sum of cubed temporal derivatives

    For OOD detection, we primarily use f2 (Path Energy) and f5 (Dynamics Energy).
    """

    def __init__(self, model_path, config_path=None, n_ddim_steps=10, device='cuda'):
        """
        Initialize DiffPath extractor.

        Args:
            model_path: Path to pretrained diffusion model checkpoint
            config_path: Path to model config YAML (optional, will use defaults)
            n_ddim_steps: Number of DDIM steps for feature extraction
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device
        self.n_ddim_steps = n_ddim_steps

        # Load config
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Default CelebA config (32x32)
            config = {
                'image_size': 32,
                'num_channels': 128,
                'num_res_blocks': 3,
                'learn_sigma': True,
                'diffusion_steps': 4000,
                'noise_schedule': 'cosine',
                'class_cond': False,
                'dropout': 0.3,
                'use_zero_module': True,
                'clip_denoised': True
            }

        # Build args namespace
        defaults = model_and_diffusion_defaults()
        for key, value in config.items():
            defaults[key] = value

        # Set DDIM steps via timestep_respacing
        defaults['timestep_respacing'] = f'ddim{n_ddim_steps}'

        # Create model and diffusion
        self.model, self.diffusion = create_model_and_diffusion(
            image_size=defaults['image_size'],
            class_cond=defaults['class_cond'],
            learn_sigma=defaults['learn_sigma'],
            sigma_small=defaults.get('sigma_small', False),
            num_channels=defaults['num_channels'],
            num_res_blocks=defaults['num_res_blocks'],
            num_heads=defaults.get('num_heads', 4),
            num_heads_upsample=defaults.get('num_heads_upsample', -1),
            attention_resolutions=defaults.get('attention_resolutions', '16,8'),
            dropout=defaults.get('dropout', 0.0),
            diffusion_steps=defaults['diffusion_steps'],
            noise_schedule=defaults['noise_schedule'],
            timestep_respacing=defaults['timestep_respacing'],
            use_kl=defaults.get('use_kl', False),
            predict_xstart=defaults.get('predict_xstart', False),
            rescale_timesteps=defaults.get('rescale_timesteps', True),
            rescale_learned_sigmas=defaults.get('rescale_learned_sigmas', True),
            use_checkpoint=defaults.get('use_checkpoint', False),
            use_scale_shift_norm=defaults.get('use_scale_shift_norm', True),
            use_zero_module=defaults.get('use_zero_module', True),
        )

        # Load weights
        print(f"Loading model from {model_path}")
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

        self.image_size = defaults['image_size']
        self.clip_denoised = defaults.get('clip_denoised', True)

    def extract_features(self, images, batch_size=32, verbose=True):
        """
        Extract 6D DiffPath features from images.

        Args:
            images: numpy array (N, H, W, C) or (N, C, H, W) or list of PIL/numpy images
            batch_size: Batch size for processing
            verbose: Whether to show progress bar

        Returns:
            features: numpy array of shape (N, 6) containing:
                [eps_sum, eps_sum_sq, eps_sum_cb, deps_dt, deps_dt_sq, deps_dt_cb]
        """
        # Create dataset and dataloader
        dataset = SimpleImageDataset(images)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        return self.extract_features_from_dataloader(dataloader, verbose=verbose)

    def extract_features_from_dataloader(self, dataloader, verbose=True):
        """
        Extract features from a pre-built dataloader.

        Args:
            dataloader: PyTorch DataLoader yielding (images, ...) tuples
            verbose: Whether to show progress bar

        Returns:
            features: numpy array of shape (N, 6)
        """
        n_ddim_steps = len(self.diffusion.betas)

        eps_sum_arr = []
        eps_sum_sq_arr = []
        eps_sum_cb_arr = []
        deps_dt_arr = []
        deps_dt_sq_arr = []
        deps_dt_cb_arr = []

        iterator = tqdm(dataloader, desc="Extracting DiffPath features") if verbose else dataloader

        with torch.no_grad():
            for batch in iterator:
                if isinstance(batch, (list, tuple)):
                    x0 = batch[0]
                else:
                    x0 = batch
                x0 = x0.to(self.device)

                # Run DDIM reverse sampling to get eps at each timestep
                _, eps_list = self.diffusion.ddim_reverse_sample_loop(
                    self.model,
                    x0.shape,
                    x0,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=None,
                    return_eps=True,
                    return_xt=False
                )

                # eps_list is a list of tensors, one for each timestep
                # Stack and convert to numpy: (T, B, C, H, W) -> (B, T, C, H, W)
                eps = torch.stack(eps_list, dim=1).cpu().numpy()

                # Compute statistics (sum over T, C, H, W dimensions)
                eps_sum = np.sum(eps, axis=(1, 2, 3, 4))  # (B,)
                eps_sum_sq = np.sum(eps ** 2, axis=(1, 2, 3, 4))  # (B,)
                eps_sum_cb = np.sum(eps ** 3, axis=(1, 2, 3, 4))  # (B,)

                # Rate of change: d(eps)/dt
                eps_diff = np.diff(eps, axis=1) * n_ddim_steps  # (B, T-1, C, H, W)
                deps_dt = np.sum(eps_diff, axis=(1, 2, 3, 4))  # (B,)
                deps_dt_sq = np.sum(eps_diff ** 2, axis=(1, 2, 3, 4))  # (B,)
                deps_dt_cb = np.sum(eps_diff ** 3, axis=(1, 2, 3, 4))  # (B,)

                eps_sum_arr.extend(eps_sum.tolist())
                eps_sum_sq_arr.extend(eps_sum_sq.tolist())
                eps_sum_cb_arr.extend(eps_sum_cb.tolist())
                deps_dt_arr.extend(deps_dt.tolist())
                deps_dt_sq_arr.extend(deps_dt_sq.tolist())
                deps_dt_cb_arr.extend(deps_dt_cb.tolist())

        # Stack features: (N, 6)
        features = np.column_stack([
            eps_sum_arr,
            eps_sum_sq_arr,
            eps_sum_cb_arr,
            deps_dt_arr,
            deps_dt_sq_arr,
            deps_dt_cb_arr
        ])

        return features

    def extract_2d_features(self, images, batch_size=32, verbose=True):
        """
        Extract 2D energy features (Path Energy + Dynamics Energy).

        This is the recommended method for OOD detection as these two features
        provide optimal discrimination according to our theoretical analysis.

        Args:
            images: numpy array (N, H, W, C) or (N, C, H, W)
            batch_size: Batch size for processing
            verbose: Whether to show progress bar

        Returns:
            features: numpy array of shape (N, 2) containing:
                [Path Energy (f2), Dynamics Energy (f5)]
        """
        full_features = self.extract_features(images, batch_size, verbose)
        # Return f2 (index 1) and f5 (index 4)
        return full_features[:, [1, 4]]


def get_transform(image_size=32, dataset_name='cifar10'):
    """
    Get appropriate image transform for dataset.

    Args:
        image_size: Target image size
        dataset_name: Name of dataset ('cifar10', 'svhn', 'celeba', etc.)

    Returns:
        torchvision.transforms.Compose: Image transformation pipeline
    """
    if dataset_name in ['celeba']:
        return transforms.Compose([
            transforms.CenterCrop(140),
            transforms.Resize((image_size, image_size),
                            interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size),
                            interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
