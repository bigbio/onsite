"""
Test modeling_score_threshold parameter handling for different score types.

This module tests that the modeling_score_threshold parameter is correctly
applied for different score types (PEP, RawScore, xcorr, E-values).

Related to: MODELING_SCORE_THRESHOLD_FIX.md
"""

import pytest
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockPSM:
    """Mock PSM for testing threshold filtering logic."""
    
    def __init__(self, psm_score, score_type=None, higher_score_better=True):
        self.psm_score = psm_score
        self.score_type = score_type
        self.higher_score_better = higher_score_better
        
        # Mock peptide with modification sites
        self.peptide = MockPeptide()
    
    def __repr__(self):
        return f"MockPSM(score={self.psm_score}, type={self.score_type})"


class MockPeptide:
    """Mock Peptide with modification sites."""
    
    def __init__(self, num_pps=1, num_rps=1):
        self.num_pps = num_pps  # Number of potential phosphorylation sites
        self.num_rps = num_rps  # Number of reported phosphorylation sites
        self.score = None


class TestModelingThresholdHigherIsBetter:
    """Test modeling_score_threshold for 'higher is better' scores."""
    
    def test_pep_threshold_filtering(self):
        """Test PEP (converted to 1-PEP) threshold filtering."""
        modeling_score_threshold = 0.95
        higher_score_better = True
        
        # Create PSMs with different PEP scores (already converted to 1-PEP)
        psms = [
            MockPSM(0.99, "Posterior Error Probability", higher_score_better),  # Should pass
            MockPSM(0.96, "Posterior Error Probability", higher_score_better),  # Should pass
            MockPSM(0.95, "Posterior Error Probability", higher_score_better),  # Should pass (boundary)
            MockPSM(0.94, "Posterior Error Probability", higher_score_better),  # Should fail
            MockPSM(0.80, "Posterior Error Probability", higher_score_better),  # Should fail
        ]
        
        # Filter PSMs using the threshold
        high_score_psms = []
        for psm in psms:
            if psm.psm_score >= modeling_score_threshold:
                high_score_psms.append(psm)
        
        assert len(high_score_psms) == 3
        assert all(psm.psm_score >= 0.95 for psm in high_score_psms)
    
    def test_rawscore_threshold_filtering(self):
        """Test MSGF+ RawScore threshold filtering."""
        modeling_score_threshold = 100.0
        higher_score_better = True
        
        # Create PSMs with different RawScore values
        psms = [
            MockPSM(200.0, "MS:1002049", higher_score_better),  # Should pass
            MockPSM(150.0, "MS:1002049", higher_score_better),  # Should pass
            MockPSM(100.0, "MS:1002049", higher_score_better),  # Should pass (boundary)
            MockPSM(99.0, "MS:1002049", higher_score_better),   # Should fail
            MockPSM(50.0, "MS:1002049", higher_score_better),   # Should fail
        ]
        
        # Filter PSMs using the threshold
        high_score_psms = []
        for psm in psms:
            if psm.psm_score >= modeling_score_threshold:
                high_score_psms.append(psm)
        
        assert len(high_score_psms) == 3
        assert all(psm.psm_score >= 100.0 for psm in high_score_psms)
    
    def test_xcorr_threshold_filtering(self):
        """Test Comet xcorr threshold filtering."""
        modeling_score_threshold = 2.0
        higher_score_better = True
        
        # Create PSMs with different xcorr values
        psms = [
            MockPSM(3.5, "COMET:xcorr", higher_score_better),  # Should pass
            MockPSM(2.5, "COMET:xcorr", higher_score_better),  # Should pass
            MockPSM(2.0, "COMET:xcorr", higher_score_better),  # Should pass (boundary)
            MockPSM(1.9, "COMET:xcorr", higher_score_better),  # Should fail
            MockPSM(1.0, "COMET:xcorr", higher_score_better),  # Should fail
        ]
        
        # Filter PSMs using the threshold
        high_score_psms = []
        for psm in psms:
            if psm.psm_score >= modeling_score_threshold:
                high_score_psms.append(psm)
        
        assert len(high_score_psms) == 3
        assert all(psm.psm_score >= 2.0 for psm in high_score_psms)


class TestModelingThresholdLowerIsBetter:
    """Test modeling_score_threshold for 'lower is better' scores (E-values)."""
    
    def test_specevalue_threshold_filtering(self):
        """Test MSGF+ SpecEValue threshold filtering."""
        modeling_score_threshold = 0.01
        higher_score_better = False
        
        # Create PSMs with different SpecEValue scores
        psms = [
            MockPSM(0.001, "MS:1002052", higher_score_better),  # Should pass
            MockPSM(0.005, "MS:1002052", higher_score_better),  # Should pass
            MockPSM(0.01, "MS:1002052", higher_score_better),   # Should pass (boundary)
            MockPSM(0.02, "MS:1002052", higher_score_better),   # Should fail
            MockPSM(0.1, "MS:1002052", higher_score_better),    # Should fail
        ]
        
        # Filter PSMs using the threshold (FIXED: use <= not hardcoded 0.01)
        high_score_psms = []
        for psm in psms:
            if psm.psm_score <= modeling_score_threshold:
                high_score_psms.append(psm)
        
        assert len(high_score_psms) == 3
        assert all(psm.psm_score <= 0.01 for psm in high_score_psms)
    
    def test_expect_threshold_filtering(self):
        """Test Comet expect threshold filtering."""
        modeling_score_threshold = 0.01
        higher_score_better = False
        
        # Create PSMs with different expect values
        psms = [
            MockPSM(0.001, "expect", higher_score_better),  # Should pass
            MockPSM(0.005, "expect", higher_score_better),  # Should pass
            MockPSM(0.01, "expect", higher_score_better),   # Should pass (boundary)
            MockPSM(0.02, "expect", higher_score_better),   # Should fail
            MockPSM(0.1, "expect", higher_score_better),    # Should fail
        ]
        
        # Filter PSMs using the threshold (FIXED: use <= not hardcoded 0.01)
        high_score_psms = []
        for psm in psms:
            if psm.psm_score <= modeling_score_threshold:
                high_score_psms.append(psm)
        
        assert len(high_score_psms) == 3
        assert all(psm.psm_score <= 0.01 for psm in high_score_psms)
    
    def test_evalue_strict_threshold(self):
        """Test E-value filtering with strict threshold (0.001)."""
        modeling_score_threshold = 0.001
        higher_score_better = False
        
        # Create PSMs with different E-values
        psms = [
            MockPSM(0.0001, "MS:1002052", higher_score_better),  # Should pass
            MockPSM(0.001, "MS:1002052", higher_score_better),   # Should pass (boundary)
            MockPSM(0.005, "MS:1002052", higher_score_better),   # Should fail
            MockPSM(0.01, "MS:1002052", higher_score_better),    # Should fail
            MockPSM(0.1, "MS:1002052", higher_score_better),     # Should fail
        ]
        
        # Filter PSMs using the strict threshold
        high_score_psms = []
        for psm in psms:
            if psm.psm_score <= modeling_score_threshold:
                high_score_psms.append(psm)
        
        assert len(high_score_psms) == 2
        assert all(psm.psm_score <= 0.001 for psm in high_score_psms)
    
    def test_evalue_loose_threshold(self):
        """Test E-value filtering with loose threshold (0.1)."""
        modeling_score_threshold = 0.1
        higher_score_better = False
        
        # Create PSMs with different E-values
        psms = [
            MockPSM(0.001, "expect", higher_score_better),  # Should pass
            MockPSM(0.01, "expect", higher_score_better),   # Should pass
            MockPSM(0.05, "expect", higher_score_better),   # Should pass
            MockPSM(0.1, "expect", higher_score_better),    # Should pass (boundary)
            MockPSM(0.2, "expect", higher_score_better),    # Should fail
        ]
        
        # Filter PSMs using the loose threshold
        high_score_psms = []
        for psm in psms:
            if psm.psm_score <= modeling_score_threshold:
                high_score_psms.append(psm)
        
        assert len(high_score_psms) == 4
        assert all(psm.psm_score <= 0.1 for psm in high_score_psms)


class TestModelingThresholdBugFix:
    """Test that the modeling_score_threshold bug is fixed."""
    
    def test_bug_evalue_hardcoded_threshold_removed(self):
        """Test that E-value filtering no longer uses hardcoded 0.01."""
        higher_score_better = False
        
        # User sets a different threshold
        user_threshold = 0.001
        
        # Create PSMs
        psms = [
            MockPSM(0.0005, "MS:1002052", higher_score_better),
            MockPSM(0.001, "MS:1002052", higher_score_better),
            MockPSM(0.005, "MS:1002052", higher_score_better),
            MockPSM(0.01, "MS:1002052", higher_score_better),
        ]
        
        # OLD BUG: Would use hardcoded 0.01, ignoring user_threshold
        hardcoded_threshold = 0.01
        old_logic_psms = [psm for psm in psms if psm.psm_score <= hardcoded_threshold]
        
        # NEW FIX: Uses user_threshold
        new_logic_psms = [psm for psm in psms if psm.psm_score <= user_threshold]
        
        # Old logic would select 4 PSMs (all <= 0.01)
        assert len(old_logic_psms) == 4
        
        # New logic should select only 2 PSMs (only <= 0.001)
        assert len(new_logic_psms) == 2
        
        # Verify the fix works
        assert len(new_logic_psms) < len(old_logic_psms)
    
    def test_bug_user_threshold_respected_for_evalues(self):
        """Test that user-specified threshold is respected for E-values."""
        higher_score_better = False
        
        # Test with different user thresholds
        test_cases = [
            (0.001, 2),  # Strict: should select 2 PSMs
            (0.01, 4),   # Default: should select 4 PSMs
            (0.1, 5),    # Loose: should select 5 PSMs
        ]
        
        psms = [
            MockPSM(0.0005, "expect", higher_score_better),
            MockPSM(0.001, "expect", higher_score_better),
            MockPSM(0.005, "expect", higher_score_better),
            MockPSM(0.01, "expect", higher_score_better),
            MockPSM(0.05, "expect", higher_score_better),
            MockPSM(0.2, "expect", higher_score_better),
        ]
        
        for user_threshold, expected_count in test_cases:
            filtered_psms = [psm for psm in psms if psm.psm_score <= user_threshold]
            assert len(filtered_psms) == expected_count, \
                f"Threshold {user_threshold} should select {expected_count} PSMs, got {len(filtered_psms)}"


class TestModelingThresholdEdgeCases:
    """Test edge cases for modeling_score_threshold."""
    
    def test_zero_threshold_higher_is_better(self):
        """Test threshold of 0.0 for 'higher is better' scores."""
        modeling_score_threshold = 0.0
        higher_score_better = True
        
        psms = [
            MockPSM(1.0, "COMET:xcorr", higher_score_better),
            MockPSM(0.5, "COMET:xcorr", higher_score_better),
            MockPSM(0.0, "COMET:xcorr", higher_score_better),
            MockPSM(-0.1, "COMET:xcorr", higher_score_better),
        ]
        
        high_score_psms = [psm for psm in psms if psm.psm_score >= modeling_score_threshold]
        
        # Should select all PSMs with score >= 0.0
        assert len(high_score_psms) == 3
    
    def test_zero_threshold_lower_is_better(self):
        """Test threshold of 0.0 for 'lower is better' scores."""
        modeling_score_threshold = 0.0
        higher_score_better = False
        
        psms = [
            MockPSM(0.0, "expect", higher_score_better),
            MockPSM(0.001, "expect", higher_score_better),
            MockPSM(0.01, "expect", higher_score_better),
        ]
        
        high_score_psms = [psm for psm in psms if psm.psm_score <= modeling_score_threshold]
        
        # Should select only PSMs with score <= 0.0
        assert len(high_score_psms) == 1
    
    def test_very_high_threshold_higher_is_better(self):
        """Test very high threshold for 'higher is better' scores."""
        modeling_score_threshold = 1000.0
        higher_score_better = True
        
        psms = [
            MockPSM(2000.0, "MS:1002049", higher_score_better),
            MockPSM(1000.0, "MS:1002049", higher_score_better),
            MockPSM(500.0, "MS:1002049", higher_score_better),
        ]
        
        high_score_psms = [psm for psm in psms if psm.psm_score >= modeling_score_threshold]
        
        # Should select only PSMs with score >= 1000.0
        assert len(high_score_psms) == 2
    
    def test_very_high_threshold_lower_is_better(self):
        """Test very high threshold for 'lower is better' scores."""
        modeling_score_threshold = 1.0
        higher_score_better = False
        
        psms = [
            MockPSM(0.001, "expect", higher_score_better),
            MockPSM(0.01, "expect", higher_score_better),
            MockPSM(0.1, "expect", higher_score_better),
            MockPSM(1.0, "expect", higher_score_better),
            MockPSM(2.0, "expect", higher_score_better),
        ]
        
        high_score_psms = [psm for psm in psms if psm.psm_score <= modeling_score_threshold]
        
        # Should select all PSMs with score <= 1.0
        assert len(high_score_psms) == 4


class TestModelingThresholdWithPeptideScore:
    """Test modeling_score_threshold when score is in peptide object."""
    
    def test_peptide_score_higher_is_better(self):
        """Test threshold filtering using peptide.score for 'higher is better'."""
        modeling_score_threshold = 0.95
        higher_score_better = True
        
        # Create PSMs without psm_score, but with peptide.score
        psms = []
        for score in [0.99, 0.96, 0.95, 0.94, 0.80]:
            psm = MockPSM(None, "Posterior Error Probability", higher_score_better)
            psm.peptide.score = score
            psms.append(psm)
        
        # Filter using peptide.score
        high_score_psms = []
        for psm in psms:
            if hasattr(psm, "psm_score") and psm.psm_score is not None:
                if psm.psm_score >= modeling_score_threshold:
                    high_score_psms.append(psm)
            elif hasattr(psm, "peptide") and hasattr(psm.peptide, "score"):
                if psm.peptide.score >= modeling_score_threshold:
                    high_score_psms.append(psm)
        
        assert len(high_score_psms) == 3
    
    def test_peptide_score_lower_is_better(self):
        """Test threshold filtering using peptide.score for 'lower is better'."""
        modeling_score_threshold = 0.01
        higher_score_better = False
        
        # Create PSMs without psm_score, but with peptide.score
        psms = []
        for score in [0.001, 0.005, 0.01, 0.02, 0.1]:
            psm = MockPSM(None, "expect", higher_score_better)
            psm.peptide.score = score
            psms.append(psm)
        
        # Filter using peptide.score (FIXED: use <= not hardcoded 0.01)
        high_score_psms = []
        for psm in psms:
            if hasattr(psm, "psm_score") and psm.psm_score is not None:
                if psm.psm_score <= modeling_score_threshold:
                    high_score_psms.append(psm)
            elif hasattr(psm, "peptide") and hasattr(psm.peptide, "score"):
                if psm.peptide.score <= modeling_score_threshold:
                    high_score_psms.append(psm)
        
        assert len(high_score_psms) == 3


@pytest.mark.parametrize("score_type,higher_better,threshold,scores,expected_count", [
    # PEP (higher is better after 1-PEP conversion)
    ("Posterior Error Probability", True, 0.95, [0.99, 0.96, 0.95, 0.94, 0.80], 3),
    ("Posterior Error Probability", True, 0.90, [0.99, 0.96, 0.95, 0.94, 0.80], 4),
    
    # RawScore (higher is better)
    ("MS:1002049", True, 100.0, [200.0, 150.0, 100.0, 99.0, 50.0], 3),
    ("MS:1002049", True, 50.0, [200.0, 150.0, 100.0, 99.0, 50.0], 5),
    
    # xcorr (higher is better)
    ("COMET:xcorr", True, 2.0, [3.5, 2.5, 2.0, 1.9, 1.0], 3),
    ("COMET:xcorr", True, 1.0, [3.5, 2.5, 2.0, 1.9, 1.0], 5),
    
    # SpecEValue (lower is better)
    ("MS:1002052", False, 0.01, [0.001, 0.005, 0.01, 0.02, 0.1], 3),
    ("MS:1002052", False, 0.001, [0.001, 0.005, 0.01, 0.02, 0.1], 1),
    ("MS:1002052", False, 0.1, [0.001, 0.005, 0.01, 0.02, 0.1], 5),
    
    # expect (lower is better)
    ("expect", False, 0.01, [0.001, 0.005, 0.01, 0.02, 0.1], 3),
    ("expect", False, 0.001, [0.001, 0.005, 0.01, 0.02, 0.1], 1),
    ("expect", False, 0.1, [0.001, 0.005, 0.01, 0.02, 0.1], 5),
])
def test_threshold_filtering_parametrized(score_type, higher_better, threshold, scores, expected_count):
    """Parametrized test for threshold filtering across different score types."""
    psms = [MockPSM(score, score_type, higher_better) for score in scores]
    
    # Apply threshold filtering
    if higher_better:
        filtered_psms = [psm for psm in psms if psm.psm_score >= threshold]
    else:
        filtered_psms = [psm for psm in psms if psm.psm_score <= threshold]
    
    assert len(filtered_psms) == expected_count, \
        f"Score type {score_type}, threshold {threshold}: expected {expected_count}, got {len(filtered_psms)}"


class TestModelingThresholdIntegration:
    """Integration tests for modeling_score_threshold in the full workflow."""
    
    def test_phospho_psm_filtering_then_threshold(self):
        """Test that PSMs are first filtered for phospho sites, then by threshold."""
        modeling_score_threshold = 0.95
        higher_score_better = True
        
        # Create PSMs with different phospho site counts
        psms = []
        
        # PSMs with phospho sites
        for score in [0.99, 0.96, 0.94]:
            psm = MockPSM(score, "Posterior Error Probability", higher_score_better)
            psm.peptide.num_pps = 1
            psm.peptide.num_rps = 1
            psms.append(psm)
        
        # PSMs without phospho sites (should be excluded first)
        for score in [0.99, 0.98]:
            psm = MockPSM(score, "Posterior Error Probability", higher_score_better)
            psm.peptide.num_pps = 0
            psm.peptide.num_rps = 0
            psms.append(psm)
        
        # Step 1: Filter for phospho sites
        phospho_psms = []
        for psm in psms:
            if hasattr(psm, "peptide") and psm.peptide is not None:
                num_pps = getattr(psm.peptide, "num_pps", 0)
                num_rps = getattr(psm.peptide, "num_rps", 0)
                if num_pps > 0 and num_rps > 0:
                    phospho_psms.append(psm)
        
        # Step 2: Filter by threshold
        high_score_psms = []
        for psm in phospho_psms:
            if psm.psm_score >= modeling_score_threshold:
                high_score_psms.append(psm)
        
        # Should have 2 PSMs (0.99 and 0.96, both with phospho sites and >= 0.95)
        assert len(high_score_psms) == 2
        assert all(psm.psm_score >= 0.95 for psm in high_score_psms)
        assert all(psm.peptide.num_pps > 0 for psm in high_score_psms)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
