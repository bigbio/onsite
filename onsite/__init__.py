"""
OnSite: Mass spectrometry post-translational modification localization tool.
"""

__version__ = "0.0.3"
__author__ = "BigBio Stack"
__license__ = "MIT"

from .ascore import AScore
from .phosphors import calculate_phospho_localization_compomics_style

try:
    from . import lucxor
    LUCXOR_AVAILABLE = True
except ImportError:
    LUCXOR_AVAILABLE = False

__all__ = ["AScore", "calculate_phospho_localization_compomics_style"]
if LUCXOR_AVAILABLE:
    __all__.extend(["lucxor"])
