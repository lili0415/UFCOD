"""
Few-Shot OOD Detection with GMM-based modeling.

This module provides an alternative detector using Gaussian Mixture Models
for more sophisticated density estimation when sufficient samples are available.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class FewShotOODDetector:
    """
    Few-Shot OOD Detector using Gaussian Mixture Models.

    This detector uses GMM to model the in-distribution density in the
    2D energy feature space. It includes optional Ledoit-Wolf shrinkage
    for robust covariance estimation with limited samples.
    """

    def __init__(self, n_components=5, covariance_type='full',
                 use_ledoit_wolf=True, verbose=False):
        """
        Initialize the detector.

        Args:
            n_components: Number of GMM components
            covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
            use_ledoit_wolf: Whether to use Ledoit-Wolf shrinkage
            verbose: Whether to print progress
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.use_ledoit_wolf = use_ledoit_wolf
        self.verbose = verbose

        self.scaler = StandardScaler()
        self.gmm = None

    def get_adaptive_budget(self, train_size):
        """
        Get adaptive sample budget based on training set size.

        Args:
            train_size: Total size of training set

        Returns:
            k: Number of samples to select
            n_layers: Number of stratification layers
        """
        if train_size <= 60000:  # Small (CIFAR-10 scale)
            return 150, 5
        elif train_size <= 100000:  # Medium (SVHN scale)
            return 100, 4
        else:  # Large (CelebA scale)
            return 80, 3

    def stratified_sample(self, features, k, n_layers, random_state=42):
        """
        Perform stratified sampling with quantile-based stratification.

        Args:
            features: Feature array (N, D)
            k: Number of samples to select
            n_layers: Number of stratification layers
            random_state: Random seed

        Returns:
            selected_indices: Indices of selected samples
            selected_features: Selected feature array
        """
        np.random.seed(random_state)
        N = len(features)

        if N <= k:
            return np.arange(N), features

        # Use L2 norm for stratification
        strat_values = np.linalg.norm(features, axis=1)

        # Compute quantile boundaries
        quantiles = np.linspace(0, 100, n_layers + 1)
        boundaries = np.percentile(strat_values, quantiles)

        # Assign samples to layers
        layer_indices = []
        for i in range(n_layers):
            if i == n_layers - 1:
                mask = (strat_values >= boundaries[i]) & (strat_values <= boundaries[i+1])
            else:
                mask = (strat_values >= boundaries[i]) & (strat_values < boundaries[i+1])
            layer_indices.append(np.where(mask)[0])

        # Sample proportionally from each layer
        samples_per_layer = k // n_layers
        remainder = k % n_layers

        selected_indices = []
        for i, indices in enumerate(layer_indices):
            n_samples = samples_per_layer + (1 if i < remainder else 0)
            if len(indices) >= n_samples:
                selected = np.random.choice(indices, n_samples, replace=False)
                selected_indices.extend(selected)
            else:
                selected_indices.extend(indices)

        selected_indices = np.array(selected_indices[:k])
        return selected_indices, features[selected_indices]

    def fit(self, features, train_size=None, random_state=42):
        """
        Fit the few-shot OOD detector.

        Args:
            features: Feature array from ID training samples (N, D)
            train_size: Total training set size (for adaptive budget)
            random_state: Random seed
        """
        N = len(features)
        if train_size is None:
            train_size = N

        # Get adaptive budget
        k, n_layers = self.get_adaptive_budget(train_size)
        if self.verbose:
            print(f"Adaptive budget: k={k}, n_layers={n_layers} (train_size={train_size})")

        # Stratified sampling
        selected_indices, selected_features = self.stratified_sample(
            features, k, n_layers, random_state
        )
        if self.verbose:
            print(f"Selected {len(selected_features)} samples via stratified sampling")

        # Normalize features
        self.scaler.fit(selected_features)
        normalized_features = self.scaler.transform(selected_features)

        # Fit GMM
        n_comp = min(self.n_components, len(normalized_features) // 2)
        self.gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type=self.covariance_type,
            max_iter=200,
            random_state=random_state
        )
        self.gmm.fit(normalized_features)

        # Apply Ledoit-Wolf shrinkage if enabled
        if self.use_ledoit_wolf and self.covariance_type == 'full':
            self._apply_ledoit_wolf(normalized_features)

        if self.verbose:
            print(f"Fitted GMM with {self.gmm.n_components} components")

        return self

    def _apply_ledoit_wolf(self, features):
        """Apply Ledoit-Wolf shrinkage to GMM covariances."""
        lw = LedoitWolf()
        lw.fit(features)
        shrinkage = lw.shrinkage_

        # Apply shrinkage to each component's covariance
        for i in range(self.gmm.n_components):
            cov = self.gmm.covariances_[i]
            # Shrink toward diagonal
            target = np.diag(np.diag(cov))
            self.gmm.covariances_[i] = (1 - shrinkage) * cov + shrinkage * target

        # Update precisions
        self.gmm.precisions_cholesky_ = np.array([
            np.linalg.cholesky(np.linalg.inv(cov))
            for cov in self.gmm.covariances_
        ])

    def score_samples(self, features):
        """
        Compute OOD scores for samples.
        Higher scores indicate more likely to be in-distribution.

        Args:
            features: Feature array (N, D)

        Returns:
            scores: OOD scores (N,)
        """
        normalized = self.scaler.transform(features)
        return self.gmm.score_samples(normalized)

    def predict(self, features, threshold=None):
        """
        Predict whether samples are in-distribution or out-of-distribution.

        Args:
            features: Feature array (N, D)
            threshold: Score threshold (if None, returns scores)

        Returns:
            predictions: Binary predictions (1=ID, 0=OOD) or scores
        """
        scores = self.score_samples(features)
        if threshold is None:
            return scores
        return (scores >= threshold).astype(int)


def compute_auroc(id_scores, ood_scores):
    """Compute AUROC for OOD detection."""
    y_true = np.concatenate([
        np.ones(len(id_scores)),
        np.zeros(len(ood_scores))
    ])
    scores = np.concatenate([id_scores, ood_scores])
    return roc_auc_score(y_true, scores)
