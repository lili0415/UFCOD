"""
Scoring utilities for UFCOD.

Implements temperature-scaled proximity scoring with optional
adaptive temperature based on local density estimation.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class AdaptiveTemperatureScorer:
    """
    Adaptive Temperature Scoring for OOD Detection.

    Implements density-aware temperature adjustment:
    - T(f) = T_0 * (1 + alpha * exp(-rho(f) / beta))

    Where rho(f) is local density estimate based on k-NN distances.
    In sparse regions, temperature increases for more robust scoring.
    In dense regions, temperature decreases for higher sensitivity.
    """

    def __init__(self, T_0=0.5, alpha=1.0, k_density=5, k_score=10):
        """
        Args:
            T_0: Base temperature
            alpha: Temperature scaling factor
            k_density: Number of neighbors for density estimation
            k_score: Number of neighbors for OOD scoring
        """
        self.T_0 = T_0
        self.alpha = alpha
        self.k_density = k_density
        self.k_score = k_score

        self.scaler = StandardScaler()
        self.reference_features = None
        self.nn_model = None
        self.beta = None  # Auto-computed from reference set

    def fit(self, features):
        """
        Fit the scorer on reference features.

        Args:
            features: numpy array of shape (N, D) - reference features

        Returns:
            self
        """
        # Standardize
        self.reference_features = self.scaler.fit_transform(features)

        # Build kNN model
        self.nn_model = NearestNeighbors(
            n_neighbors=self.k_score,
            algorithm='auto'
        )
        self.nn_model.fit(self.reference_features)

        # Compute beta from average density of reference set
        densities = self._compute_density(self.reference_features)
        self.beta = np.median(densities)  # Use median for robustness

        return self

    def _compute_density(self, features):
        """Compute local density as inverse of average k-NN distance."""
        k = min(self.k_density + 1, len(self.reference_features))
        distances, _ = self.nn_model.kneighbors(features, n_neighbors=k)

        # Average distance (excluding 0-distance to self if applicable)
        if distances.shape[1] > self.k_density:
            avg_dist = np.mean(distances[:, 1:self.k_density+1], axis=1)
        else:
            avg_dist = np.mean(distances[:, :self.k_density], axis=1)

        # Density = inverse distance (with small epsilon for stability)
        density = 1.0 / (avg_dist + 1e-8)
        return density

    def _adaptive_temperature(self, density):
        """Compute adaptive temperature based on local density."""
        # T(f) = T_0 * (1 + alpha * exp(-rho / beta))
        # Low density (sparse region) -> high T (more smoothing)
        # High density (dense region) -> low T (more sensitive)
        return self.T_0 * (1 + self.alpha * np.exp(-density / self.beta))

    def score_samples(self, features, use_adaptive=True):
        """
        Compute OOD scores using soft-min with adaptive temperature.

        Args:
            features: Test features to score
            use_adaptive: Whether to use adaptive temperature

        Returns:
            scores: OOD scores (higher = more likely ID)
        """
        features_scaled = self.scaler.transform(features)

        # Get distances to all reference points
        distances, _ = self.nn_model.kneighbors(features_scaled, n_neighbors=self.k_score)

        if use_adaptive:
            # Compute density and adaptive temperature for each test sample
            density = self._compute_density(features_scaled)
            T = self._adaptive_temperature(density)

            # Soft-min distance: T * log(sum(exp(-d/T)))
            scores = np.zeros(len(features_scaled))
            for i, (d, t) in enumerate(zip(distances, T)):
                soft_min_dist = -t * np.log(np.sum(np.exp(-d / t)) + 1e-10)
                scores[i] = -soft_min_dist  # Negate: closer = higher score
        else:
            # Fixed temperature soft-min
            T = self.T_0
            exp_neg_d = np.exp(-distances / T)
            soft_min_dist = -T * np.log(np.sum(exp_neg_d, axis=1) + 1e-10)
            scores = -soft_min_dist  # Negate: closer = higher score

        return scores

    def score_samples_fixed_T(self, features, T):
        """
        Score with a specific fixed temperature.

        Args:
            features: Test features to score
            T: Fixed temperature value

        Returns:
            scores: OOD scores
        """
        features_scaled = self.scaler.transform(features)
        distances, _ = self.nn_model.kneighbors(features_scaled, n_neighbors=self.k_score)
        exp_neg_d = np.exp(-distances / T)
        soft_min_dist = -T * np.log(np.sum(exp_neg_d, axis=1) + 1e-10)
        return -soft_min_dist  # Negate: closer = higher score
