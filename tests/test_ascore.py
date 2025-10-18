"""
Test AScore algorithm.
"""

import pytest
import sys
import os
from pyopenms import AASequence, MSSpectrum, Peak1D, PeptideHit
from onsite import AScore

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_ascore_initialization():
    """Test AScore initialization."""
    ascore = AScore()
    assert ascore is not None
    assert hasattr(ascore, 'fragment_mass_tolerance_')
    assert hasattr(ascore, 'compute')

def test_ascore_parameters():
    """Test AScore parameter setting."""
    ascore = AScore()
    
    # Test parameter setting
    ascore.fragment_mass_tolerance_ = 0.1
    assert ascore.fragment_mass_tolerance_ == 0.1
    
    ascore.fragment_tolerance_ppm_ = True
    assert ascore.fragment_tolerance_ppm_ == True

def test_ascore_with_peptide():
    """Test AScore with a simple peptide."""
    
    ascore = AScore()
    
    # Create a simple peptide
    sequence = AASequence.fromString("PEPTIDE")
    hit = PeptideHit()
    hit.setSequence(sequence)
    
    # Create a simple spectrum
    spectrum = MSSpectrum()
    spectrum.set_peaks([(100.0, 1000.0), (200.0, 2000.0)])
    
    # This should not crash (even if it doesn't find phosphorylation sites)
    try:
        result = ascore.compute(hit, spectrum)
        assert result is not None
    except Exception as e:
        # AScore might fail for non-phosphorylated peptides, which is expected
        assert "phosphorylation" in str(e).lower() or "modification" in str(e).lower()
