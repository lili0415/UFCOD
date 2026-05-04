"""
Energy-based OOD Detector using 2D diffusion geometry features.

This is the main detector class for UFCOD, implementing:
- Facility Location coreset selection for few-shot reference set construction
- Temperature-scaled proximity scoring
- Z-score calibration for cross-domain deployment
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


class EnergyOODDetector:
    """
    UFCOD: Unified Few-shot Cross-domain OOD Detector.

    Uses 2D energy features (Path Energy + Dynamics Energy) from diffusion
    trajectories with temperature-scaled proximity scoring.

    Attributes:
        T: Temperature for soft-min scoring (default: 0.5)
        k: Number of nearest neighbors for scoring (default: 10)
        scaler: StandardScaler for feature normalization
        reference_features: Normalized reference features
        nn_model: Nearest neighbors model
        calibration_mean: Mean of ID scores for calibration
        calibration_std: Std of ID scores for calibration
    """

    def __init__(self, T=0.5, k=10):
        """
        Initialize the OOD detector.

        Args:
            T: Temperature parameter for soft-min scoring.
               Lower T = more sensitive to nearest neighbor (sharper).
               Higher T = more robust, considers more neighbors.
            k: Number of nearest neighbors for scoring.
        """
        self.T = T
        self.k = k
        self.scaler = StandardScaler()
        self.reference_features = None
        self.nn_model = None
        self.calibration_mean = 0.0
        self.calibration_std = 1.0

    def fit(self, features, method='facility_location', n_samples=100, random_state=42):
        """
        Fit the detector on ID reference features.

        Args:
            features: numpy array of shape (N, 2) - 2D energy features
            method: Sample selection method ('facility_location', 'random', 'stratified')
            n_samples: Number of samples to select for reference set
            random_state: Random seed for reproducibility

        Returns:
            self
        """
        # Select reference samples
        if len(features) <= n_samples:
            selected_idx = np.arange(len(features))
        elif method == 'facility_location':
            selected_idx = self._facility_location_sample(features, n_samples, random_state)
        elif method == 'stratified':
            selected_idx = self._stratified_sample(features, n_samples, random_state)
        else:
            np.random.seed(random_state)
            selected_idx = np.random.choice(len(features), n_samples, replace=False)

        selected_features = features[selected_idx]

        # Normalize features
        self.reference_features = self.scaler.fit_transform(selected_features)

        # Build kNN model
        self.nn_model = NearestNeighbors(n_neighbors=self.k, algorithm='auto')
        self.nn_model.fit(self.reference_features)

        # Compute calibration statistics on reference set
        ref_scores = self._score_samples_raw(self.reference_features)
        self.calibration_mean = np.mean(ref_scores)
        self.calibration_std = np.std(ref_scores) + 1e-8

        return self

    def _facility_location_sample(self, features, n_samples, random_state):
        """
        Facility Location coreset selection for maximum coverage.

        This method selects samples that minimize the maximum distance from
        any point to the nearest selected point, providing optimal coverage
        of the feature space.
        """
        np.random.seed(random_state)
        N = len(features)

        # Normalize for distance computation
        normalized = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

        # Greedy facility location
        selected = []
        remaining = list(range(N))

        # Start with random point
        first = np.random.randint(N)
        selected.append(first)
        remaining.remove(first)

        # Distance from each point to nearest selected
        min_dist = np.linalg.norm(normalized - normalized[first], axis=1)

        for _ in range(n_samples - 1):
            if not remaining:
                break

            # Find point that maximizes minimum improvement
            best_idx = None
            best_improvement = -np.inf

            for idx in remaining:
                # Distance from each point to this candidate
                dist_to_candidate = np.linalg.norm(normalized - normalized[idx], axis=1)
                # New min distance if we add this candidate
                new_min_dist = np.minimum(min_dist, dist_to_candidate)
                # Total improvement (reduction in sum of min distances)
                improvement = np.sum(min_dist - new_min_dist)

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
                dist_to_new = np.linalg.norm(normalized - normalized[best_idx], axis=1)
                min_dist = np.minimum(min_dist, dist_to_new)

        return np.array(selected)

    def _stratified_sample(self, features, n_samples, random_state):
        """Stratified sampling based on feature norms."""
        np.random.seed(random_state)
        N = len(features)

        norms = np.linalg.norm(features, axis=1)
        n_layers = 5
        samples_per_layer = n_samples // n_layers
        extra = n_samples % n_layers
        quantiles = np.percentile(norms, np.linspace(0, 100, n_layers + 1))

        selected = []
        for i in range(n_layers):
            lower, upper = quantiles[i], quantiles[i + 1]
            if i == n_layers - 1:
                mask = (norms >= lower) & (norms <= upper)
            else:
                mask = (norms >= lower) & (norms < upper)

            layer_indices = np.where(mask)[0]
            n_select = samples_per_layer + (1 if i < extra else 0)
            n_select = min(n_select, len(layer_indices))

            if len(layer_indices) > 0:
                selected.extend(np.random.choice(layer_indices, n_select, replace=False))

        return np.array(selected[:n_samples])

    def _score_samples_raw(self, features_scaled):
        """Compute raw soft-min scores (before calibration)."""
        distances, _ = self.nn_model.kneighbors(features_scaled, n_neighbors=self.k)

        # Soft-min with temperature: -T * log(sum(exp(-d/T)))
        exp_neg_d = np.exp(-distances / self.T)
        soft_min_dist = -self.T * np.log(np.sum(exp_neg_d, axis=1) + 1e-10)

        # Negate so higher score = closer to ID (more likely ID)
        return -soft_min_dist

    def score_samples(self, features, calibrate=True):
        """
        Compute OOD scores for samples.

        Args:
            features: numpy array of shape (N, 2) - 2D energy features
            calibrate: Whether to apply Z-score calibration

        Returns:
            scores: numpy array of shape (N,)
                    Higher scores indicate more likely IN-distribution.
        """
        features_scaled = self.scaler.transform(features)
        scores = self._score_samples_raw(features_scaled)

        if calibrate:
            scores = (scores - self.calibration_mean) / self.calibration_std

        return scores

    def predict(self, features, threshold=0.0, calibrate=True):
        """
        Predict whether samples are in-distribution or out-of-distribution.

        Args:
            features: numpy array of shape (N, 2)
            threshold: Score threshold (default 0 for calibrated scores)
            calibrate: Whether to apply calibration

        Returns:
            predictions: numpy array of shape (N,)
                        1 = in-distribution, 0 = out-of-distribution
        """
        scores = self.score_samples(features, calibrate=calibrate)
        return (scores >= threshold).astype(int)


def compute_auroc(id_scores, ood_scores):
    """
    Compute AUROC for OOD detection.

    Args:
        id_scores: Scores for in-distribution samples (higher = more ID)
        ood_scores: Scores for out-of-distribution samples

    Returns:
        auroc: Area under ROC curve
    """
    y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    return roc_auc_score(y_true, scores)


def compute_fpr_at_tpr(id_scores, ood_scores, tpr_threshold=0.95):
    """
    Compute FPR at a given TPR threshold.

    Args:
        id_scores: Scores for in-distribution samples
        ood_scores: Scores for out-of-distribution samples
        tpr_threshold: Target TPR (default 0.95)

    Returns:
        fpr: False positive rate at the given TPR
    """
    # Find threshold that achieves target TPR
    threshold = np.percentile(id_scores, 100 * (1 - tpr_threshold))
    # Compute FPR
    fpr = np.mean(ood_scores >= threshold)
    return fpr
