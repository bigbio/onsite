"""
OnSite: Mass spectrometry post-translational modification localization tool.

This package provides tools for phosphorylation site localization and scoring
using various algorithms including AScore, PhosphoRS, and LucXor.
"""

__version__ = "0.0.1"
__author__ = "BigBio Stack"
__license__ = "MIT"

# Import main modules
from .ascore import AScore
from .phosphors import calculate_phospho_localization_compomics_style

# Import LucXor components
try:
    from . import lucxor
    LUCXOR_AVAILABLE = True
except ImportError:
    LUCXOR_AVAILABLE = False

__all__ = [
    "AScore",
    "calculate_phospho_localization_compomics_style"
]

if LUCXOR_AVAILABLE:
    __all__.extend([
        "lucxor"
    ])
