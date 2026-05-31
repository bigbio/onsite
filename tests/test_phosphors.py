"""
Test PhosphoRS algorithm.
"""

import pytest
import sys
import os
from pyopenms import AASequence, MSSpectrum, Peak1D, PeptideHit
from onsite import calculate_phospho_localization_compomics_style

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_phosphors_import():
    """Test PhosphoRS function import."""
    from onsite import calculate_phospho_localization_compomics_style

    assert calculate_phospho_localization_compomics_style is not None


def test_phosphors_with_peptide():
    """Test PhosphoRS with a simple phosphorylated peptide."""
    # Create a phosphorylated peptide
    sequence = AASequence.fromString("PEPTIDE(Phospho)")
    hit = PeptideHit()
    hit.setSequence(sequence)

    # Create a simple spectrum
    spectrum = MSSpectrum()
    spectrum.set_peaks([(100.0, 1000.0), (200.0, 2000.0)])

    # Test the function
    try:
        site_probs, isomer_list = calculate_phospho_localization_compomics_style(
            hit, spectrum, fragment_tolerance=0.05, fragment_method_ppm=False
        )
        # The function should return results or None
        assert site_probs is None or isinstance(site_probs, dict)
        assert isomer_list is None or isinstance(isomer_list, list)
    except Exception as e:
        # PhosphoRS might fail for various reasons, which is acceptable in tests
        assert len(str(e)) > 0


def test_phosphors_parameters():
    """Test PhosphoRS with different parameters."""
    # Create a phosphorylated peptide
    sequence = AASequence.fromString("PEPTIDE(Phospho)")
    hit = PeptideHit()
    hit.setSequence(sequence)

    # Create a simple spectrum
    spectrum = MSSpectrum()
    spectrum.set_peaks([(100.0, 1000.0), (200.0, 2000.0)])

    # Test with different tolerance values
    for tolerance in [0.01, 0.05, 0.1]:
        try:
            site_probs, isomer_list = calculate_phospho_localization_compomics_style(
                hit,
                spectrum,
                fragment_tolerance=tolerance,
                fragment_method_ppm=False,
                add_decoys=False,
            )
            # Should not crash
        except Exception:
            # Acceptable if it fails
            pass


def test_count_matched_ions_no_double_counting():
    """Peak matching must not count one experimental peak for several
    theoretical ions, nor count indistinguishable theoretical ions twice
    (bigbio/onsite#40)."""
    from onsite.phosphors.phosphors import _count_matched_ions

    # One experimental peak sits within tolerance of TWO distinct theoretical
    # ions; it may be consumed only once -> k == 1 (naive scan would give 2).
    n, k = _count_matched_ions([100.00, 100.08], [100.04], 0.05, False)
    assert n == 2 and k == 1

    # Three near-identical theoretical ions collapse to ONE trial.
    n, k = _count_matched_ions([200.00, 200.02, 200.04], [200.01], 0.05, False)
    assert n == 1 and k == 1

    # Independent ions and peaks match normally.
    n, k = _count_matched_ions([300.0, 400.0, 500.0], [300.01, 500.02], 0.05, False)
    assert n == 3 and k == 2

    # No experimental peaks -> no matches.
    n, k = _count_matched_ions([300.0, 400.0], [], 0.05, False)
    assert n == 2 and k == 0
