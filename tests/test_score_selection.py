"""
Test score selection functionality for lucxor
"""

import pytest
import numpy as np
from pyopenms import PeptideIdentification, PeptideHit, AASequence


def test_score_selection_import():
    """Test that the score selection method can be imported"""
    from onsite.lucxor.cli import PyLuciPHOr2
    
    tool = PyLuciPHOr2()
    assert hasattr(tool, 'get_score_for_psm')


def test_score_normalization_pep():
    """Test score normalization for PEP scores (lower is better, normalized)"""
    from onsite.lucxor.cli import PyLuciPHOr2
    
    tool = PyLuciPHOr2()
    
    # Create test objects
    pep_id = PeptideIdentification()
    pep_id.setScoreType("Posterior Error Probability")
    pep_id.setHigherScoreBetter(False)
    
    hit = PeptideHit()
    hit.setScore(0.1)  # Low PEP = good
    hit.setSequence(AASequence.fromString("PEPTIDE"))
    
    # Test score selection
    normalized_score, score_type, is_normalized = tool.get_score_for_psm(pep_id, hit)
    
    # For PEP, we convert to 1-PEP for higher-is-better
    assert normalized_score == pytest.approx(0.9)
    assert "posterior error probability" in score_type.lower()
    assert is_normalized is True


def test_score_normalization_qvalue():
    """Test score normalization for q-value scores (lower is better, normalized)"""
    from onsite.lucxor.cli import PyLuciPHOr2
    
    tool = PyLuciPHOr2()
    
    # Create test objects with q-value as metavalue
    pep_id = PeptideIdentification()
    pep_id.setScoreType("q-value")
    pep_id.setHigherScoreBetter(False)
    
    hit = PeptideHit()
    hit.setScore(0.05)  # Low q-value = good
    hit.setSequence(AASequence.fromString("PEPTIDE"))
    
    # Test score selection
    normalized_score, score_type, is_normalized = tool.get_score_for_psm(pep_id, hit)
    
    # For q-value, we convert to 1-q-value for higher-is-better
    assert normalized_score == pytest.approx(0.95)
    assert "q-value" in score_type.lower()
    assert is_normalized is True


def test_score_normalization_evalue():
    """Test score normalization for E-value scores (lower is better, unnormalized)"""
    from onsite.lucxor.cli import PyLuciPHOr2
    
    tool = PyLuciPHOr2()
    
    # Create test objects with E-value
    pep_id = PeptideIdentification()
    pep_id.setScoreType("E-value")
    pep_id.setHigherScoreBetter(False)
    
    hit = PeptideHit()
    hit.setScore(1e-5)  # Low E-value = good
    hit.setSequence(AASequence.fromString("PEPTIDE"))
    
    # Test score selection
    normalized_score, score_type, is_normalized = tool.get_score_for_psm(pep_id, hit)
    
    # For E-value, we use -log10 transformation
    expected_score = -np.log10(1e-5)  # Should be 5.0
    assert normalized_score == pytest.approx(expected_score)
    assert "e-value" in score_type.lower()
    assert is_normalized is False  # E-values are not normalized


def test_score_selection_with_preferred_type():
    """Test that preferred score type is used when specified"""
    from onsite.lucxor.cli import PyLuciPHOr2
    
    tool = PyLuciPHOr2()
    
    # Create test objects with both primary score and meta value
    pep_id = PeptideIdentification()
    pep_id.setScoreType("Posterior Error Probability")
    pep_id.setHigherScoreBetter(False)
    
    hit = PeptideHit()
    hit.setScore(0.1)  # PEP score
    hit.setMetaValue("q-value", 0.05)  # q-value as meta value
    hit.setSequence(AASequence.fromString("PEPTIDE"))
    
    # Test with preferred q-value
    normalized_score, score_type, is_normalized = tool.get_score_for_psm(
        pep_id, hit, preferred_score_type="q-value"
    )
    
    # Should use q-value instead of PEP
    assert normalized_score == pytest.approx(0.95)
    assert "q-value" in score_type.lower()


def test_score_higher_better():
    """Test score handling for higher-is-better scores"""
    from onsite.lucxor.cli import PyLuciPHOr2
    
    tool = PyLuciPHOr2()
    
    # Create test objects with higher-is-better score
    pep_id = PeptideIdentification()
    pep_id.setScoreType("XCorr")
    pep_id.setHigherScoreBetter(True)
    
    hit = PeptideHit()
    hit.setScore(5.0)
    hit.setSequence(AASequence.fromString("PEPTIDE"))
    
    # Test score selection
    normalized_score, score_type, is_normalized = tool.get_score_for_psm(pep_id, hit)
    
    # For higher-is-better scores, no transformation needed
    assert normalized_score == pytest.approx(5.0)
    assert "xcorr" in score_type.lower()
    assert is_normalized is False  # XCorr is not normalized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
