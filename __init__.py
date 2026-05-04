"""
UFCOD: Unified Few-shot Cross-domain OOD Detection

A framework for few-shot out-of-distribution detection using diffusion geometry features.
"""

__version__ = "0.1.0"
__author__ = "Anonymous"

from .detectors.energy_detector import EnergyOODDetector
from .detectors.fewshot_detector import FewShotOODDetector
from .models.feature_extractor import DiffPathExtractor
from .utils.scoring import AdaptiveTemperatureScorer

__all__ = [
    "EnergyOODDetector",
    "FewShotOODDetector",
    "DiffPathExtractor",
    "AdaptiveTemperatureScorer",
]
