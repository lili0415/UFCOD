"""
Detectors module for UFCOD.
"""

from .energy_detector import EnergyOODDetector, compute_auroc, compute_fpr_at_tpr
from .fewshot_detector import FewShotOODDetector

__all__ = [
    "EnergyOODDetector",
    "FewShotOODDetector",
    "compute_auroc",
    "compute_fpr_at_tpr",
]
