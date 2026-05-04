"""
Coreset selection utilities for UFCOD.

Implements Facility Location and stratified sampling for
constructing representative reference sets from limited samples.
"""

import numpy as np


def facility_location_sample(features, n_samples, random_state=42):
    """
    Facility Location coreset selection for maximum coverage.

    This method selects samples that minimize the sum of distances from
    all points to their nearest selected point, providing optimal coverage
    of the feature space for few-shot detection.

    Args:
        features: numpy array of shape (N, D)
        n_samples: Number of samples to select
        random_state: Random seed for reproducibility

    Returns:
        selected_indices: numpy array of selected indices
    """
    np.random.seed(random_state)
    N = len(features)

    if N <= n_samples:
        return np.arange(N)

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

        # Find point that maximizes improvement (reduces total distance)
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


def stratified_sample(features, n_samples, n_layers=5, random_state=42):
    """
    Stratified sampling based on feature norms.

    Divides the feature space into quantile-based layers and samples
    proportionally from each layer to ensure coverage across the
    distribution.

    Args:
        features: numpy array of shape (N, D)
        n_samples: Number of samples to select
        n_layers: Number of stratification layers
        random_state: Random seed for reproducibility

    Returns:
        selected_indices: numpy array of selected indices
    """
    np.random.seed(random_state)
    N = len(features)

    if N <= n_samples:
        return np.arange(N)

    # Use L2 norm for stratification
    norms = np.linalg.norm(features, axis=1)

    # Compute quantile boundaries
    quantiles = np.percentile(norms, np.linspace(0, 100, n_layers + 1))

    # Sample proportionally from each layer
    samples_per_layer = n_samples // n_layers
    extra = n_samples % n_layers

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


def kmeans_pp_sample(features, n_samples, random_state=42):
    """
    K-means++ style sampling for diversity.

    Iteratively selects points with probability proportional to
    squared distance from nearest already-selected point.

    Args:
        features: numpy array of shape (N, D)
        n_samples: Number of samples to select
        random_state: Random seed for reproducibility

    Returns:
        selected_indices: numpy array of selected indices
    """
    np.random.seed(random_state)
    N = len(features)

    if N <= n_samples:
        return np.arange(N)

    selected = [np.random.randint(N)]

    for _ in range(n_samples - 1):
        # Compute distances to nearest selected point
        dists = np.min([
            np.linalg.norm(features - features[s], axis=1)
            for s in selected
        ], axis=0)

        # Sample with probability proportional to distance squared
        probs = dists ** 2
        probs /= probs.sum()
        next_idx = np.random.choice(N, p=probs)
        selected.append(next_idx)

    return np.array(selected)
