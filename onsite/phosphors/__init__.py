"""
PhosphoRS package for phosphorylation site localization.

This package provides the PhosphoRS algorithm implementation and scoring pipeline
for mass spectrometry post-translational modification localization.
"""

from .phosphors import calculate_phospho_localization_compomics_style

__all__ = ['calculate_phospho_localization_compomics_style']
