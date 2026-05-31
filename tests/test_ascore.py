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
    assert hasattr(ascore, "fragment_mass_tolerance_")
    assert hasattr(ascore, "compute")


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


def test_getsites_sorted_with_decoys():
    """Decoy (A) sites must be returned in ascending position order.

    Regression for bigbio/onsite#40: getSites_ appended A sites after the
    S/T/Y sites, leaving the list unsorted (e.g. [0, 4, 1, 2, 3] for SAAAYK).
    """
    ascore = AScore()
    ascore.setAddDecoys(True)
    sites = ascore.getSites_("SAAAYK")
    assert sites == sorted(sites), f"site positions not ascending: {sites}"
    assert sites == [0, 1, 2, 3, 4]

    # combinations() over the (now sorted) sites must yield ascending combos
    for perm in ascore.computePermutations_(sites, 2):
        assert perm == sorted(perm), f"unsorted permutation produced: {perm}"


def test_decoy_modification_not_dropped():
    """An out-of-order permutation must not silently drop a modification.

    Regression for bigbio/onsite#40: createTheoreticalSpectra_ walked positions
    ascending and dropped any site whose index was smaller than a preceding one,
    so a descending combo like [4, 1] on SAAAYK lost the decoy A entirely.
    """
    ascore = AScore()
    ascore.setAddDecoys(True)
    seq = AASequence.fromString("SAAAYK")

    # Descending and ascending forms of the same site set must be identical,
    # and must place BOTH modifications (decoy A@1 + phospho Y@4).
    expected = "SA(PhosphoDecoy)AAY(Phospho)K"
    for combo in ([4, 1], [1, 4]):
        name = ascore.createTheoreticalSpectra_([combo], seq)[0].getName()
        assert name == expected, f"{combo} -> {name}"

    # Three sites (2 decoys + 1 target) fed descending must keep all three.
    name3 = ascore.createTheoreticalSpectra_([[4, 3, 1]], seq)[0].getName()
    assert name3.count("(PhosphoDecoy)") == 2
    assert name3.count("(Phospho)") == 1
