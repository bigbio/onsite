"""
Regression tests for LucXor algorithm.

These tests ensure that optimizations don't change the algorithm's output.
Random seeds are set for reproducibility.
"""

import pytest
import random
import numpy as np
import os

# Set seeds at module level for consistent test ordering
RANDOM_SEED = 42


def set_random_seeds(seed=RANDOM_SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


class TestPeptideIonLadders:
    """Test that ion ladder building produces consistent results."""

    def test_ion_ladder_consistency(self):
        """Test that ion ladders are identical with and without skip_expensive_init."""
        from onsite.lucxor.peptide import Peptide

        config = {
            "fragment_mass_tolerance": 0.5,
            "ms2_tolerance_units": "Da",
            "target_modifications": ["Phospho (S)", "Phospho (T)", "Phospho (Y)"],
            "neutral_losses": ["sty -H3PO4 -97.97690"],
            "decoy_neutral_losses": ["X -H3PO4 -97.97690"],
            "min_mz": 150.0,
        }

        # Test with normal initialization
        pep1 = Peptide("PEPTIDEs(Phospho)EQ", config, charge=2)

        # Test with skip_expensive_init
        pep2 = Peptide(
            "PEPTIDEs(Phospho)EQ", config, charge=2, skip_expensive_init=True
        )
        pep2.build_ion_ladders()

        assert pep1.b_ions == pep2.b_ions, "B-ions differ with skip_expensive_init"
        assert pep1.y_ions == pep2.y_ions, "Y-ions differ with skip_expensive_init"

    def test_ion_ladder_values(self):
        """Test that ion ladder values match expected reference values."""
        from onsite.lucxor.peptide import Peptide

        config = {
            "fragment_mass_tolerance": 0.5,
            "ms2_tolerance_units": "Da",
            "target_modifications": ["Phospho (S)", "Phospho (T)", "Phospho (Y)"],
            "neutral_losses": ["sty -H3PO4 -97.97690"],
            "decoy_neutral_losses": ["X -H3PO4 -97.97690"],
            "min_mz": 150.0,
        }

        pep = Peptide("PEPTIDEs(Phospho)EQ", config, charge=2)

        # Reference values (computed once and stored)
        expected_b_ions = {
            "b2:PE": 227.10262646688,
            "b3:PEP": 324.15538646688,
            "b4:PEPT": 425.20306646688,
        }

        expected_y_ions = {
            "y9:EPTIDEsEQ": 1127.41399215088,
            "y8:PTIDEsEQ": 998.3714021508799,
            "y7:TIDEsEQ": 901.3186421508799,
        }

        for ion_name, expected_mz in expected_b_ions.items():
            assert ion_name in pep.b_ions, f"Missing b-ion: {ion_name}"
            assert abs(pep.b_ions[ion_name] - expected_mz) < 1e-6, (
                f"B-ion {ion_name} mismatch: {pep.b_ions[ion_name]} vs {expected_mz}"
            )

        for ion_name, expected_mz in expected_y_ions.items():
            assert ion_name in pep.y_ions, f"Missing y-ion: {ion_name}"
            assert abs(pep.y_ions[ion_name] - expected_mz) < 1e-6, (
                f"Y-ion {ion_name} mismatch: {pep.y_ions[ion_name]} vs {expected_mz}"
            )


class TestPeakMatching:
    """Test that peak matching produces consistent results."""

    def test_match_peaks_basic(self):
        """Test basic peak matching functionality."""
        from onsite.lucxor.peptide import Peptide
        from onsite.lucxor.spectrum import Spectrum

        set_random_seeds()

        config = {
            "fragment_mass_tolerance": 0.5,
            "ms2_tolerance_units": "Da",
            "target_modifications": ["Phospho (S)", "Phospho (T)", "Phospho (Y)"],
            "neutral_losses": ["sty -H3PO4 -97.97690"],
            "decoy_neutral_losses": ["X -H3PO4 -97.97690"],
            "min_mz": 150.0,
        }

        pep = Peptide("PEPTIDEs(Phospho)EQ", config, charge=2)

        # Create spectrum with peaks at theoretical m/z positions
        theo_mz = list(pep.b_ions.values()) + list(pep.y_ions.values())
        mz_array = np.array(theo_mz) + np.random.uniform(-0.1, 0.1, len(theo_mz))
        mz_array = np.sort(mz_array)
        intensity_array = np.random.uniform(1000, 10000, len(mz_array))

        spectrum = Spectrum(mz_array, intensity_array)

        matched = pep.match_peaks(spectrum, config)

        # All matched peaks should be within tolerance
        for peak in matched:
            assert abs(peak["mass_diff"]) <= 0.25, (
                f"Peak {peak['matched_ion_str']} outside tolerance: {peak['mass_diff']}"
            )

    def test_match_peaks_deterministic(self):
        """Test that peak matching gives identical results with same seed."""
        from onsite.lucxor.peptide import Peptide
        from onsite.lucxor.spectrum import Spectrum

        config = {
            "fragment_mass_tolerance": 0.5,
            "ms2_tolerance_units": "Da",
            "target_modifications": ["Phospho (S)", "Phospho (T)", "Phospho (Y)"],
            "neutral_losses": ["sty -H3PO4 -97.97690"],
            "decoy_neutral_losses": ["X -H3PO4 -97.97690"],
            "min_mz": 150.0,
        }

        results = []
        for _ in range(2):
            set_random_seeds()

            pep = Peptide("PEPTIDEs(Phospho)EQ", config, charge=2)

            theo_mz = list(pep.b_ions.values()) + list(pep.y_ions.values())
            mz_array = np.array(theo_mz) + np.random.uniform(-0.1, 0.1, len(theo_mz))
            mz_array = np.sort(mz_array)
            intensity_array = np.random.uniform(1000, 10000, len(mz_array))

            spectrum = Spectrum(mz_array, intensity_array)
            matched = pep.match_peaks(spectrum, config)

            results.append(matched)

        # Compare results from two runs
        assert len(results[0]) == len(results[1]), "Different number of matched peaks"
        for p1, p2 in zip(
            sorted(results[0], key=lambda x: x["mz"]),
            sorted(results[1], key=lambda x: x["mz"]),
        ):
            assert p1["mz"] == p2["mz"], "m/z values differ"
            assert p1["matched_ion_str"] == p2["matched_ion_str"], "Ion strings differ"
            assert abs(p1["mass_diff"] - p2["mass_diff"]) < 1e-10, "Mass diff values differ"


class TestPSMScoring:
    """Test PSM scoring consistency."""

    def test_score_permutations_deterministic(self):
        """Test that PSM scoring is deterministic with same seed."""
        from onsite.lucxor.psm import PSM
        from onsite.lucxor.peptide import Peptide
        from onsite.lucxor.spectrum import Spectrum

        config = {
            "fragment_mass_tolerance": 0.5,
            "ms2_tolerance_units": "Da",
            "target_modifications": ["Phospho (S)", "Phospho (T)", "Phospho (Y)"],
            "neutral_losses": ["sty -H3PO4 -97.97690"],
            "decoy_neutral_losses": ["X -H3PO4 -97.97690"],
            "decoy_mass": 79.966331,
            "min_mz": 150.0,
            "max_charge_state": 3,
            "modeling_score_threshold": 0.95,
        }

        # Create a simple spectrum
        set_random_seeds()
        mz_array = np.linspace(200, 1200, 500)
        intensity_array = np.random.uniform(100, 10000, len(mz_array))
        spectrum = Spectrum(mz_array, intensity_array)

        results = []
        for _ in range(2):
            set_random_seeds()

            # Create Peptide object first
            peptide = Peptide("PEPTIDES(Phospho)TEQ", config, charge=2)

            # Create PSM with Peptide object and Spectrum
            psm = PSM(
                peptide=peptide,
                spectrum_source=spectrum,
                config=config,
            )

            # Generate permutations (run_number=0 for first round with decoys)
            psm.generate_permutations(run_number=0)

            results.append({
                "pos_perms": list(psm.pos_permutation_score_map.keys()),
                "neg_perms": list(psm.neg_permutation_score_map.keys()),
            })

        # Real permutations should be identical (no randomness)
        assert results[0]["pos_perms"] == results[1]["pos_perms"], (
            "Positive permutations differ"
        )

        # Decoy permutations should be identical with same seed
        assert results[0]["neg_perms"] == results[1]["neg_perms"], (
            "Negative permutations differ with same seed"
        )


class TestEndToEndRegression:
    """End-to-end regression tests using real data."""

    @pytest.fixture
    def data_files(self):
        """Check if test data files exist."""
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        idxml = os.path.join(data_dir, "1_consensus_fdr_filter_pep.idXML")
        mzml = os.path.join(data_dir, "1.mzML")

        if not os.path.exists(idxml) or not os.path.exists(mzml):
            pytest.skip("Test data files not available")

        return idxml, mzml

    def test_lucxor_deterministic_with_seed(self, data_files, tmp_path):
        """Test that LucXor produces identical results with same seed."""
        import subprocess

        idxml, mzml = data_files
        output1 = tmp_path / "output1.idXML"
        output2 = tmp_path / "output2.idXML"

        # Run twice with seed wrapper
        for output in [output1, output2]:
            # Create a wrapper script that sets the seed
            result = subprocess.run(
                [
                    "python3",
                    "-c",
                    f"""
import random
import numpy as np
random.seed(42)
np.random.seed(42)

import sys
sys.argv = ['lucxor', '--input-id', '{idxml}', '--input-spectrum', '{mzml}', '--output', '{output}']

from onsite.lucxor.cli import main
main()
""",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                pytest.fail(f"LucXor failed: {result.stderr}")

        # Compare outputs
        import re

        def extract_scores(filename):
            scores = []
            with open(filename) as f:
                for line in f:
                    match = re.search(r'PeptideHit score="([^"]+)"', line)
                    if match:
                        scores.append(float(match.group(1)))
            return np.array(scores)

        scores1 = extract_scores(output1)
        scores2 = extract_scores(output2)

        assert len(scores1) == len(scores2), "Different number of scores"
        np.testing.assert_allclose(
            scores1,
            scores2,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Scores differ between runs with same seed",
        )

    def test_score_statistics_stability(self, data_files, tmp_path):
        """Test that score statistics remain stable across runs (even without seed)."""
        import subprocess
        import re

        idxml, mzml = data_files

        scores_list = []
        for i in range(2):
            output = tmp_path / f"output_{i}.idXML"

            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "onsite.lucxor.cli",
                    "--input-id",
                    idxml,
                    "--input-spectrum",
                    mzml,
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                pytest.fail(f"LucXor run {i} failed: {result.stderr}")

            scores = []
            with open(output) as f:
                for line in f:
                    match = re.search(r'PeptideHit score="([^"]+)"', line)
                    if match:
                        scores.append(float(match.group(1)))
            scores_list.append(np.array(scores))

        # Statistical comparison - runs should be very similar
        s1, s2 = scores_list

        # Same number of scores
        assert len(s1) == len(s2), "Different number of identifications"

        # Correlation should be very high (>0.999)
        mask = (s1 > 0) | (s2 > 0)
        if np.sum(mask) > 10:
            corr = np.corrcoef(s1[mask], s2[mask])[0, 1]
            assert corr > 0.999, f"Score correlation too low: {corr}"

            # Mean should be very similar (<1% difference)
            mean1, mean2 = np.mean(s1[mask]), np.mean(s2[mask])
            rel_diff = abs(mean1 - mean2) / max(mean1, mean2)
            assert rel_diff < 0.01, f"Mean scores differ by {rel_diff*100:.2f}%"
