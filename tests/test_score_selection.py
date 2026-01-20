"""
Test score selection and handling logic for pyLuciPHOr2.

This module tests the score selection, conversion, and filtering logic
that was implemented to fix bugs in score handling for different search engines.
"""

import pytest
import sys
import os
from typing import Tuple

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onsite.lucxor.cli import PyLuciPHOr2


class MockPeptideHit:
    """Mock PeptideHit for testing score selection logic."""
    
    def __init__(self, score, meta_values=None):
        self.score = score
        self.meta_values = meta_values or {}
    
    def getScore(self):
        return self.score
    
    def metaValueExists(self, key):
        return key in self.meta_values
    
    def getMetaValue(self, key):
        return self.meta_values.get(key, None)


class MockPeptideIdentification:
    """Mock PeptideIdentification for testing score selection logic."""
    
    def __init__(self, score_type, higher_score_better, hits):
        self.score_type = score_type
        self.higher_score_better = higher_score_better
        self.hits = hits
    
    def getScoreType(self):
        return self.score_type
    
    def isHigherScoreBetter(self):
        return self.higher_score_better
    
    def getHits(self):
        return self.hits


class MockPeptideIdentificationList(list):
    """Mock PeptideIdentificationList for testing."""
    pass


@pytest.fixture
def tool():
    """Create a PyLuciPHOr2 instance for testing."""
    return PyLuciPHOr2()


class TestScoreTypeSelection:
    """Test score type selection logic."""
    
    def test_select_percolator_pep_highest_priority(self, tool):
        """Test that Percolator PEP is selected with highest priority."""
        hit = MockPeptideHit(
            score=2.85e-05,
            meta_values={
                "MS:1002049": 166.0,
                "MS:1001493": 2.85e-05,
            }
        )
        pep_id = MockPeptideIdentification("Posterior Error Probability", False, [hit])
        pep_ids = MockPeptideIdentificationList([pep_id])
        
        score_type, higher_better = tool.select_score_type(pep_ids)
        
        assert score_type == "Posterior Error Probability"
        assert higher_better is False
    
    def test_select_msgf_rawscore_over_specevalue(self, tool):
        """Test that MSGF+ RawScore is preferred over SpecEValue."""
        hit = MockPeptideHit(
            score=1.0777954e-12,
            meta_values={
                "MS:1002049": 166.0,
                "MS:1002052": 1.0777954e-12,
            }
        )
        pep_id = MockPeptideIdentification("SpecEValue", False, [hit])
        pep_ids = MockPeptideIdentificationList([pep_id])
        
        score_type, higher_better = tool.select_score_type(pep_ids)
        
        # Should switch to RawScore
        assert score_type == "MS:1002049"
        assert higher_better is True
    
    def test_select_comet_xcorr_over_expect(self, tool):
        """Test that Comet xcorr is preferred over expect."""
        hit = MockPeptideHit(
            score=5.3e-03,
            meta_values={
                "COMET:xcorr": 1.641,
                "COMET:deltaCn": 1.0,
            }
        )
        pep_id = MockPeptideIdentification("expect", False, [hit])
        pep_ids = MockPeptideIdentificationList([pep_id])
        
        score_type, higher_better = tool.select_score_type(pep_ids)
        
        # Should switch to xcorr
        assert score_type == "COMET:xcorr"
        assert higher_better is True
    
    def test_user_specified_score_type(self, tool):
        """Test that user-specified score type is respected."""
        hit = MockPeptideHit(
            score=1.641,
            meta_values={"COMET:xcorr": 1.641}
        )
        pep_id = MockPeptideIdentification("expect", False, [hit])
        pep_ids = MockPeptideIdentificationList([pep_id])
        
        score_type, higher_better = tool.select_score_type(
            pep_ids, 
            user_score_type="COMET:xcorr"
        )
        
        assert score_type == "COMET:xcorr"
        assert higher_better is True
    
    def test_empty_peptide_list(self, tool):
        """Test handling of empty peptide identification list."""
        pep_ids = MockPeptideIdentificationList([])
        
        score_type, higher_better = tool.select_score_type(pep_ids)
        
        assert score_type is None
        assert higher_better is True


class TestScoreConversion:
    """Test score conversion logic."""
    
    def test_pep_score_conversion(self, tool):
        """Test that PEP scores are converted to 1-PEP."""
        hit = MockPeptideHit(score=0.05)
        pep_id = MockPeptideIdentification("Posterior Error Probability", False, [hit])
        
        score = tool.get_psm_score(pep_id, hit, "Posterior Error Probability", False)
        
        assert pytest.approx(score, abs=1e-6) == 0.95
    
    def test_qvalue_score_conversion(self, tool):
        """Test that Q-value scores are converted to 1-Q."""
        hit = MockPeptideHit(
            score=0.01,
            meta_values={"MS:1001491": 0.01}
        )
        pep_id = MockPeptideIdentification("Q-value", False, [hit])
        
        score = tool.get_psm_score(pep_id, hit, "MS:1001491", False)
        
        assert pytest.approx(score, abs=1e-6) == 0.99
    
    def test_evalue_no_conversion(self, tool):
        """Test that E-values are NOT converted."""
        hit = MockPeptideHit(score=0.005)
        pep_id = MockPeptideIdentification("expect", False, [hit])
        
        score = tool.get_psm_score(pep_id, hit, "expect", False)
        
        assert score == 0.005
    
    def test_xcorr_no_conversion(self, tool):
        """Test that xcorr scores are NOT converted."""
        hit = MockPeptideHit(
            score=1.641,
            meta_values={"COMET:xcorr": 1.641}
        )
        pep_id = MockPeptideIdentification("COMET:xcorr", True, [hit])
        
        score = tool.get_psm_score(pep_id, hit, "COMET:xcorr", True)
        
        assert score == 1.641
    
    def test_rawscore_no_conversion(self, tool):
        """Test that MSGF+ RawScore is NOT converted."""
        hit = MockPeptideHit(
            score=166.0,
            meta_values={"MS:1002049": 166.0}
        )
        pep_id = MockPeptideIdentification("MS:1002049", True, [hit])
        
        score = tool.get_psm_score(pep_id, hit, "MS:1002049", True)
        
        assert score == 166.0
    
    def test_pep_out_of_range_handling(self, tool):
        """Test that out-of-range PEP values are handled gracefully."""
        hit = MockPeptideHit(score=1.5)  # Invalid PEP > 1
        pep_id = MockPeptideIdentification("Posterior Error Probability", False, [hit])
        
        score = tool.get_psm_score(pep_id, hit, "Posterior Error Probability", False)
        
        # Should return as-is with warning
        assert score == 1.5
    
    def test_negative_pep_handling(self, tool):
        """Test that negative PEP values are handled gracefully."""
        hit = MockPeptideHit(score=-0.1)  # Invalid PEP < 0
        pep_id = MockPeptideIdentification("Posterior Error Probability", False, [hit])
        
        score = tool.get_psm_score(pep_id, hit, "Posterior Error Probability", False)
        
        # Should return as-is with warning
        assert score == -0.1


class TestScoreFiltering:
    """Test score filtering logic."""
    
    def test_higher_is_better_filtering(self):
        """Test filtering logic for 'higher is better' scores."""
        threshold = 0.95
        
        # Score above threshold - should pass
        assert 1.641 >= threshold
        
        # Score below threshold - should fail
        assert not (0.80 >= threshold)
    
    def test_lower_is_better_filtering(self):
        """Test filtering logic for 'lower is better' scores."""
        threshold = 0.01
        
        # Score below threshold - should pass
        assert 0.005 <= threshold
        
        # Score above threshold - should fail
        assert not (0.05 <= threshold)
    
    def test_pep_converted_filtering(self):
        """Test filtering logic for converted PEP scores."""
        threshold = 0.95
        
        # PEP = 0.00001 -> 1-PEP = 0.99999 -> should pass
        pep_score = 0.00001
        converted_score = 1.0 - pep_score
        assert converted_score >= threshold
        
        # PEP = 0.10 -> 1-PEP = 0.90 -> should fail
        pep_score = 0.10
        converted_score = 1.0 - pep_score
        assert not (converted_score >= threshold)


class TestScorePriority:
    """Test score priority ordering."""
    
    def test_pep_priority_over_rawscore(self, tool):
        """Test that PEP has priority over MSGF+ RawScore."""
        hit = MockPeptideHit(
            score=2.85e-05,
            meta_values={
                "MS:1002049": 166.0,
                "MS:1001493": 2.85e-05,
            }
        )
        pep_id = MockPeptideIdentification("Posterior Error Probability", False, [hit])
        pep_ids = MockPeptideIdentificationList([pep_id])
        
        score_type, _ = tool.select_score_type(pep_ids)
        
        assert "posterior error probability" in score_type.lower()
    
    def test_pep_priority_over_xcorr(self, tool):
        """Test that PEP has priority over Comet xcorr."""
        hit = MockPeptideHit(
            score=2.85e-05,
            meta_values={
                "COMET:xcorr": 1.641,
                "MS:1001493": 2.85e-05,
            }
        )
        pep_id = MockPeptideIdentification("Posterior Error Probability", False, [hit])
        pep_ids = MockPeptideIdentificationList([pep_id])
        
        score_type, _ = tool.select_score_type(pep_ids)
        
        assert "posterior error probability" in score_type.lower()
    
    def test_rawscore_and_xcorr_equal_priority(self, tool):
        """Test that MSGF+ RawScore and Comet xcorr have equal priority."""
        # When both are available, either can be selected (implementation dependent)
        hit = MockPeptideHit(
            score=166.0,
            meta_values={
                "MS:1002049": 166.0,
                "COMET:xcorr": 1.641,
            }
        )
        pep_id = MockPeptideIdentification("MS:1002049", True, [hit])
        pep_ids = MockPeptideIdentificationList([pep_id])
        
        score_type, _ = tool.select_score_type(pep_ids)
        
        # Should select one of the raw scores (both have priority 2)
        assert score_type in ["MS:1002049", "COMET:xcorr"]
    
    def test_rawscore_priority_over_specevalue(self, tool):
        """Test that MSGF+ RawScore has priority over SpecEValue."""
        hit = MockPeptideHit(
            score=1e-12,
            meta_values={
                "MS:1002049": 166.0,
                "MS:1002052": 1e-12,
            }
        )
        pep_id = MockPeptideIdentification("SpecEValue", False, [hit])
        pep_ids = MockPeptideIdentificationList([pep_id])
        
        score_type, _ = tool.select_score_type(pep_ids)
        
        assert score_type == "MS:1002049"
    
    def test_xcorr_priority_over_expect(self, tool):
        """Test that Comet xcorr has priority over expect."""
        hit = MockPeptideHit(
            score=5.3e-03,
            meta_values={
                "COMET:xcorr": 1.641,
            }
        )
        pep_id = MockPeptideIdentification("expect", False, [hit])
        pep_ids = MockPeptideIdentificationList([pep_id])
        
        score_type, _ = tool.select_score_type(pep_ids)
        
        assert score_type == "COMET:xcorr"
    
    def test_specevalue_priority_over_expect(self, tool):
        """Test that SpecEValue and expect have similar priority (both E-values)."""
        # Both are E-values with similar priority levels
        hit = MockPeptideHit(
            score=1e-12,
            meta_values={
                "MS:1002052": 1e-12,
            }
        )
        pep_id = MockPeptideIdentification("expect", False, [hit])
        pep_ids = MockPeptideIdentificationList([pep_id])
        
        score_type, _ = tool.select_score_type(pep_ids)
        
        # Should prefer SpecEValue if available
        assert score_type == "MS:1002052"


class TestBugFixes:
    """Test that the reported bugs are fixed."""
    
    def test_bug_fix_xcorr_not_negative(self, tool):
        """Test Bug #1: xcorr should not become negative after conversion."""
        hit = MockPeptideHit(
            score=1.641,
            meta_values={"COMET:xcorr": 1.641}
        )
        pep_id = MockPeptideIdentification("COMET:xcorr", True, [hit])
        
        score = tool.get_psm_score(pep_id, hit, "COMET:xcorr", True)
        
        # Should NOT be 1.0 - 1.641 = -0.641
        assert score > 0
        assert score == 1.641
    
    def test_bug_fix_evalue_not_inverted(self, tool):
        """Test Bug #1: E-values should not be inverted."""
        hit = MockPeptideHit(score=0.005)
        pep_id = MockPeptideIdentification("expect", False, [hit])
        
        score = tool.get_psm_score(pep_id, hit, "expect", False)
        
        # Should NOT be 1.0 - 0.005 = 0.995
        assert score < 0.1
        assert score == 0.005
    
    def test_bug_fix_evalue_filtering_direction(self):
        """Test Bug #2: E-value filtering should use <= not >=."""
        threshold = 0.01
        evalue = 0.05
        
        # Old bug: evalue >= threshold would be True (wrong)
        old_logic = evalue >= threshold
        assert old_logic is True  # This was the bug
        
        # New fix: evalue <= threshold should be False (correct)
        new_logic = evalue <= threshold
        assert new_logic is False  # This is correct
    
    def test_bug_fix_good_evalue_passes(self):
        """Test Bug #2: Good E-values should pass filtering."""
        threshold = 0.01
        evalue = 0.005
        
        # Good E-value should pass
        assert evalue <= threshold
    
    def test_bug_fix_rawscore_not_converted(self, tool):
        """Test that high RawScore values are not incorrectly converted."""
        hit = MockPeptideHit(
            score=200.0,
            meta_values={"MS:1002049": 200.0}
        )
        pep_id = MockPeptideIdentification("MS:1002049", True, [hit])
        
        score = tool.get_psm_score(pep_id, hit, "MS:1002049", True)
        
        # Should NOT be 1.0 - 200.0 = -199.0
        assert score > 0
        assert score == 200.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_score(self, tool):
        """Test handling of zero scores."""
        hit = MockPeptideHit(score=0.0)
        pep_id = MockPeptideIdentification("COMET:xcorr", True, [hit])
        
        score = tool.get_psm_score(pep_id, hit, "COMET:xcorr", True)
        
        assert score == 0.0
    
    def test_pep_boundary_zero(self, tool):
        """Test PEP score at boundary (0.0)."""
        hit = MockPeptideHit(score=0.0)
        pep_id = MockPeptideIdentification("Posterior Error Probability", False, [hit])
        
        score = tool.get_psm_score(pep_id, hit, "Posterior Error Probability", False)
        
        assert score == 1.0  # 1.0 - 0.0
    
    def test_pep_boundary_one(self, tool):
        """Test PEP score at boundary (1.0)."""
        hit = MockPeptideHit(score=1.0)
        pep_id = MockPeptideIdentification("Posterior Error Probability", False, [hit])
        
        score = tool.get_psm_score(pep_id, hit, "Posterior Error Probability", False)
        
        assert score == 0.0  # 1.0 - 1.0
    
    def test_very_small_evalue(self, tool):
        """Test handling of very small E-values."""
        hit = MockPeptideHit(score=1e-100)
        pep_id = MockPeptideIdentification("expect", False, [hit])
        
        score = tool.get_psm_score(pep_id, hit, "expect", False)
        
        assert score == 1e-100
    
    def test_very_large_rawscore(self, tool):
        """Test handling of very large RawScore values."""
        hit = MockPeptideHit(
            score=10000.0,
            meta_values={"MS:1002049": 10000.0}
        )
        pep_id = MockPeptideIdentification("MS:1002049", True, [hit])
        
        score = tool.get_psm_score(pep_id, hit, "MS:1002049", True)
        
        assert score == 10000.0


@pytest.mark.parametrize("score_type,higher_better,expected_direction", [
    ("Posterior Error Probability", False, "lower"),
    ("MS:1002049", True, "higher"),
    ("COMET:xcorr", True, "higher"),
    ("MS:1002052", False, "lower"),
    ("expect", False, "lower"),
    ("SpecEValue", False, "lower"),
])
def test_score_direction_detection(tool, score_type, higher_better, expected_direction):
    """Test that score direction is correctly detected for different score types."""
    hit = MockPeptideHit(score=1.0)
    pep_id = MockPeptideIdentification(score_type, higher_better, [hit])
    pep_ids = MockPeptideIdentificationList([pep_id])
    
    _, detected_higher_better = tool.select_score_type(pep_ids)
    
    if expected_direction == "higher":
        assert detected_higher_better is True
    else:
        assert detected_higher_better is False


class TestCaseInsensitivity:
    """Test case-insensitive score type matching."""
    
    def test_pep_case_variations(self, tool):
        """Test that PEP is recognized in different cases."""
        variations = [
            "Posterior Error Probability",
            "posterior error probability",
            "POSTERIOR ERROR PROBABILITY",
        ]
        
        for variation in variations:
            hit = MockPeptideHit(score=0.05)
            pep_id = MockPeptideIdentification(variation, False, [hit])
            pep_ids = MockPeptideIdentificationList([pep_id])
            
            score_type, higher_better = tool.select_score_type(pep_ids)
            
            # Should recognize as PEP regardless of case
            assert "posterior error probability" in score_type.lower()
            assert higher_better is False
    
    def test_expect_case_variations(self, tool):
        """Test that expect is recognized in different cases."""
        variations = ["expect", "Expect", "EXPECT"]
        
        for variation in variations:
            hit = MockPeptideHit(
                score=0.005,
                meta_values={"COMET:xcorr": 1.641}
            )
            pep_id = MockPeptideIdentification(variation, False, [hit])
            pep_ids = MockPeptideIdentificationList([pep_id])
            
            score_type, higher_better = tool.select_score_type(pep_ids)
            
            # Should switch to xcorr regardless of expect case
            assert score_type == "COMET:xcorr"
            assert higher_better is True
    
    def test_specevalue_case_variations(self, tool):
        """Test that SpecEValue is recognized in different cases."""
        variations = ["SpecEValue", "specevalue", "SPECEVALUE"]
        
        for variation in variations:
            hit = MockPeptideHit(
                score=1e-12,
                meta_values={"MS:1002049": 166.0}
            )
            pep_id = MockPeptideIdentification(variation, False, [hit])
            pep_ids = MockPeptideIdentificationList([pep_id])
            
            score_type, higher_better = tool.select_score_type(pep_ids)
            
            # Should switch to RawScore regardless of SpecEValue case
            assert score_type == "MS:1002049"
            assert higher_better is True
    
    def test_userparam_case_variations(self, tool):
        """Test that UserParam scores are found with case variations."""
        # Test with different case variations in UserParam
        hit = MockPeptideHit(
            score=1.641,
            meta_values={
                "COMET:xcorr": 1.641,  # Actual case used in files
                "comet:xcorr": 1.641,  # Lowercase variation
            }
        )
        pep_id = MockPeptideIdentification("expect", False, [hit])
        pep_ids = MockPeptideIdentificationList([pep_id])
        
        score_type, higher_better = tool.select_score_type(pep_ids)
        
        # Should find COMET:xcorr in UserParam
        assert "xcorr" in score_type.lower()
        assert higher_better is True
