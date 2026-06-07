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

        # Reference values from PyOpenMS TheoreticalSpectrumGenerator
        # Ion naming format: b2+, y3-H2O1+, etc.
        expected_b_ions = {
            "b2+": 227.102634913371,
            "b3+": 324.155399136671,
            "b4+": 425.203078359971,
        }

        expected_y_ions = {
            "y9+": 1127.414015094071,
            "y8+": 998.371420870771,
            "y7+": 901.318656646471,
        }

        mass_tolerance = 1e-6

        for ion_name, expected_mz in expected_b_ions.items():
            assert ion_name in pep.b_ions, f"Missing b-ion: {ion_name}"
            assert abs(pep.b_ions[ion_name] - expected_mz) < mass_tolerance, (
                f"B-ion {ion_name} mismatch: {pep.b_ions[ion_name]} vs {expected_mz}"
            )

        for ion_name, expected_mz in expected_y_ions.items():
            assert ion_name in pep.y_ions, f"Missing y-ion: {ion_name}"
            assert abs(pep.y_ions[ion_name] - expected_mz) < mass_tolerance, (
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

    def test_match_peaks_with_non_target_mods(self):
        """Test that _match_peaks correctly handles non-target modifications like Oxidation.

        Regression test for bug where non_target_mods weren't copied to temporary
        Peptide objects in PSM._match_peaks(), causing PyOpenMS to fail with:
        'unexpected character m' when converting internal format to PyOpenMS format.

        The internal format uses lowercase letters (e.g., 'm' for oxidized M), which
        must be converted to PyOpenMS format (e.g., 'M(Oxidation)') using the
        non_target_mods dictionary.
        """
        from onsite.lucxor.psm import PSM
        from onsite.lucxor.peptide import Peptide
        from onsite.lucxor.spectrum import Spectrum

        set_random_seeds()

        config = {
            "fragment_mass_tolerance": 0.5,
            "ms2_tolerance_units": "Da",
            "target_modifications": ["Phospho (S)", "Phospho (T)", "Phospho (Y)"],
            "neutral_losses": ["sty -H3PO4 -97.97690"],
            "decoy_neutral_losses": ["X -H3PO4 -97.97690"],
            "decoy_mass": 79.966331,
            "min_mz": 150.0,
            "max_charge_state": 3,
        }

        # Peptide with both Phospho (target) and Oxidation (non-target) modifications
        # This is the case that triggered the bug: M(Oxidation) becomes 'm' in internal
        # format, and without non_target_mods being passed, it couldn't be converted back
        peptide_seq = "ALLSLHT(Phospho)M(Oxidation)K"
        peptide = Peptide(peptide_seq, config, charge=2)

        # Verify that non_target_mods was populated correctly
        assert len(peptide.non_target_mods) > 0, "non_target_mods should contain Oxidation"
        assert any("Oxidation" in str(v) for v in peptide.non_target_mods.values()), (
            "non_target_mods should contain Oxidation modification"
        )

        # Create a simple spectrum
        mz_array = np.linspace(200, 1200, 500)
        intensity_array = np.random.uniform(100, 10000, len(mz_array))
        spectrum = Spectrum(mz_array, intensity_array)

        # Create PSM
        psm = PSM(
            peptide=peptide,
            spectrum_source=spectrum,
            config=config,
        )

        # Verify PSM has the non_target_mods
        assert len(psm.non_target_mods) > 0, "PSM.non_target_mods should contain Oxidation"

        # This is the critical test: _match_peaks should work without raising
        # "unexpected character 'm'" error. Before the fix, this would fail because
        # non_target_mods wasn't passed to the temporary Peptide object.
        internal_format_perm = peptide.mod_peptide  # e.g., "ALLSLHtmK"
        try:
            matched_peaks = psm._match_peaks(internal_format_perm, 0.5)
        except ValueError as e:
            if "unexpected character" in str(e):
                pytest.fail(
                    f"_match_peaks failed to handle non-target mod in internal format: {e}\n"
                    f"This indicates non_target_mods wasn't copied to temp Peptide."
                )
            raise

        # Verify we got some matched peaks (exact count depends on spectrum)
        # The important thing is that it didn't crash
        assert isinstance(matched_peaks, list), "_match_peaks should return a list"

    def test_match_peaks_preserves_non_target_mods_in_permutations(self):
        """Test that non-target mods are preserved across all permutation scorings.

        When scoring different phospho site permutations, non-target modifications
        (like Oxidation) should remain in the same position for all permutations.
        """
        from onsite.lucxor.psm import PSM
        from onsite.lucxor.peptide import Peptide
        from onsite.lucxor.spectrum import Spectrum

        set_random_seeds()

        config = {
            "fragment_mass_tolerance": 0.5,
            "ms2_tolerance_units": "Da",
            "target_modifications": ["Phospho (S)", "Phospho (T)", "Phospho (Y)"],
            "neutral_losses": ["sty -H3PO4 -97.97690"],
            "decoy_neutral_losses": ["X -H3PO4 -97.97690"],
            "decoy_mass": 79.966331,
            "min_mz": 150.0,
            "max_charge_state": 3,
        }

        # Peptide with multiple potential phospho sites and an oxidation
        # S at pos 4, T at pos 6 - both potential phospho sites
        # M at pos 8 - oxidation (non-target)
        peptide_seq = "AAALST(Phospho)SM(Oxidation)K"
        peptide = Peptide(peptide_seq, config, charge=2)

        # Create spectrum
        mz_array = np.linspace(200, 1200, 500)
        intensity_array = np.random.uniform(100, 10000, len(mz_array))
        spectrum = Spectrum(mz_array, intensity_array)

        psm = PSM(
            peptide=peptide,
            spectrum_source=spectrum,
            config=config,
        )

        # Generate real permutations - these move the Phospho to different S/T positions
        # but Oxidation on M should stay the same
        real_perms = psm.generate_real_permutations()

        assert len(real_perms) > 1, "Should have multiple permutations for ambiguous sites"

        # Each permutation should successfully build ion ladders without error
        for perm_str, sites in real_perms:
            try:
                matched = psm._match_peaks(perm_str, 0.5)
                assert isinstance(matched, list), f"_match_peaks failed for {perm_str}"
            except ValueError as e:
                pytest.fail(
                    f"_match_peaks failed for permutation '{perm_str}': {e}\n"
                    f"Non-target mods may not be correctly preserved."
                )


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


class TestFLRCalculation:
    """Test FLR calculation accuracy."""

    def test_flr_interpolation_accuracy(self):
        """Test that vectorized density interpolation matches expected values."""
        from onsite.lucxor.flr import FLRCalculator

        # Create a simple FLR calculator with known density
        calc = FLRCalculator()
        calc.NMARKS = 101
        calc.tick_marks = np.linspace(0, 10, calc.NMARKS)

        # Simple triangular density for testing
        calc.f0 = np.maximum(0, 1 - np.abs(calc.tick_marks - 3) / 3)
        calc.f1 = np.maximum(0, 1 - np.abs(calc.tick_marks - 7) / 3)

        # Test interpolation at various points
        test_points = np.array([0.0, 1.5, 3.0, 5.0, 7.0, 8.5, 10.0])

        for x in test_points:
            # Find expected value by linear interpolation
            idx = np.searchsorted(calc.tick_marks, x, side='right') - 1
            idx = np.clip(idx, 0, calc.NMARKS - 2)
            a, b = calc.tick_marks[idx], calc.tick_marks[idx + 1]
            t = (x - a) / (b - a)
            expected_f0 = calc.f0[idx] * (1 - t) + calc.f0[idx + 1] * t
            expected_f1 = calc.f1[idx] * (1 - t) + calc.f1[idx + 1] * t

            # Get vectorized result
            result_f0 = calc._interpolate_density_vectorized(np.array([x]), calc.f0)[0]
            result_f1 = calc._interpolate_density_vectorized(np.array([x]), calc.f1)[0]

            assert abs(result_f0 - expected_f0) < 1e-10, f"f0 interpolation failed at x={x}"
            assert abs(result_f1 - expected_f1) < 1e-10, f"f1 interpolation failed at x={x}"

    def test_flr_boundary_conditions(self):
        """Test FLR handling of boundary conditions."""
        from onsite.lucxor.flr import FLRCalculator

        calc = FLRCalculator()
        calc.NMARKS = 101
        calc.tick_marks = np.linspace(0, 10, calc.NMARKS)
        calc.f0 = np.ones(calc.NMARKS) * 0.1
        calc.f1 = np.ones(calc.NMARKS) * 0.2

        # Test below minimum
        result = calc._interpolate_density_vectorized(np.array([-5.0]), calc.f0)[0]
        assert result == calc.f0[0], "Should return first value for x < min"

        # Test above maximum
        result = calc._interpolate_density_vectorized(np.array([15.0]), calc.f0)[0]
        assert result == calc.f0[-1], "Should return last value for x > max"

        # Test global AUC at boundaries
        cumulative_auc = calc._compute_cumulative_auc_from_end(calc.f0)

        # At x=0, should get total area
        result = calc._global_auc_vectorized(np.array([0.0]), calc.f0, cumulative_auc)[0]
        expected_total = np.sum(np.diff(calc.tick_marks) * 0.5 * (calc.f0[:-1] + calc.f0[1:]))
        assert abs(result - expected_total) < 1e-10, "Should return total area at x=0"

        # At x=max, should get 0
        result = calc._global_auc_vectorized(np.array([10.0]), calc.f0, cumulative_auc)[0]
        assert result == 0.0, "Should return 0 at x=max"

    def test_flr_cumulative_auc_correctness(self):
        """Test that cumulative AUC is computed correctly."""
        from onsite.lucxor.flr import FLRCalculator

        calc = FLRCalculator()
        calc.NMARKS = 11
        calc.tick_marks = np.linspace(0, 10, calc.NMARKS)
        calc.f0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

        cumulative = calc._compute_cumulative_auc_from_end(calc.f0)

        # Verify cumulative[i] = sum of trapezoid areas from i to end
        for i in range(calc.NMARKS - 1):
            expected = 0.0
            for j in range(i, calc.NMARKS - 1):
                dx = calc.tick_marks[j + 1] - calc.tick_marks[j]
                area = dx * 0.5 * (calc.f0[j] + calc.f0[j + 1])
                expected += area
            assert abs(cumulative[i] - expected) < 1e-10, f"Cumulative AUC wrong at index {i}"

        # Last element should be 0
        assert cumulative[-1] == 0.0, "Cumulative AUC at end should be 0"


class TestEndToEndRegression:
    """End-to-end regression tests using real data."""

    @pytest.fixture
    def data_files(self):
        """Check if test data files exist."""
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        idparquet = os.path.join(data_dir, "SF_200217_pPeptideLibrary_pool1_HCDnlETcaD_OT_rep1_comet_perc.idparquet")
        mzml = os.path.join(data_dir, "SF_200217_pPeptideLibrary_pool1_HCDnlETcaD_OT_rep1.mzML")

        if not os.path.isdir(idparquet) or not os.path.isfile(os.path.join(idparquet, "psms.parquet")) or not os.path.exists(mzml):
            pytest.skip("Test data files not available")

        return idparquet, mzml

    def test_lucxor_deterministic_with_seed(self, data_files, tmp_path):
        """Test that LucXor produces identical results with same seed."""
        from click.testing import CliRunner
        from onsite.onsitec import cli

        idparquet, mzml = data_files
        output1 = str(tmp_path / "output1.idparquet")
        output2 = str(tmp_path / "output2.idparquet")
        runner = CliRunner()

        args_base = [
            "lucxor",
            "--input-spectrum", mzml,
            "--input-id", idparquet,
            "--target-modifications", "Phospho(S),Phospho(T),Phospho(Y)",
        ]

        for out in [output1, output2]:
            r = runner.invoke(cli, args_base + ["--output", out])
            if r.exit_code != 0:
                pytest.fail(f"LucXor failed: {r.output[:500]}")

        import pandas as pd
        scores1 = pd.read_parquet(os.path.join(output1, "psms.parquet"))["score"].values
        scores2 = pd.read_parquet(os.path.join(output2, "psms.parquet"))["score"].values

        assert len(scores1) == len(scores2), "Different number of scores"
        np.testing.assert_allclose(
            scores1, scores2, rtol=1e-10, atol=1e-10,
            err_msg="Scores differ between runs with same seed",
        )

    def test_score_statistics_stability(self, data_files, tmp_path):
        """Test that score statistics remain stable across runs (even without seed)."""
        from click.testing import CliRunner
        from onsite.onsitec import cli

        idparquet, mzml = data_files
        runner = CliRunner()

        args_base = [
            "lucxor",
            "--input-spectrum", mzml,
            "--input-id", idparquet,
            "--target-modifications", "Phospho(S),Phospho(T),Phospho(Y)",
        ]

        scores_list = []
        for i in range(2):
            output = str(tmp_path / f"output_{i}.idparquet")
            r = runner.invoke(cli, args_base + ["--output", output])
            if r.exit_code != 0:
                pytest.fail(f"LucXor run {i} failed: {r.output[:500]}")

            import pandas as pd
            scores = pd.read_parquet(os.path.join(output, "psms.parquet"))["score"].values
            scores_list.append(scores)

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


class TestPhosphoDecoyAlanine:
    """Decoy-amino-acid (PhosphoDecoy on Alanine) must compete fairly.

    Regression for bigbio/onsite#40: in the production scoring path a
    PhosphoDecoy-A site is encoded as lowercase 'a', which was recognized by
    neither the phospho branch (S/T/Y) nor the native-decoy branch
    (DECOY_AA_MAP). It was therefore scored as the unmodified backbone (no
    +79.966) and could not be serialized to (PhosphoDecoy) in the output.
    """

    def _bare_psm(self):
        from onsite.lucxor.psm import PSM

        return PSM.__new__(PSM)  # the parsers/formatter only use the `perm` arg

    def test_decoy_a_is_a_real_phospho_site(self):
        psm = self._bare_psm()
        # PhosphoDecoy A at index 6
        phospho_pos, decoy_pos, is_decoy = psm._get_mod_positions_from_perm(
            "ALLSSSaVLYK"
        )
        assert phospho_pos == [6], "decoy-A must be a phospho-mass-bearing position"
        assert decoy_pos == []
        assert is_decoy is False, "decoy-A is a target permutation, not a native decoy"

    def test_decoy_a_carries_phospho_mass(self):
        from onsite.lucxor.constants import PHOSPHO_MOD_MASS

        psm = self._bare_psm()
        assert psm._get_mod_map("ALLSSSaVLYK") == {6: PHOSPHO_MOD_MASS}

    def test_decoy_a_serializes_and_parses(self):
        from pyopenms import AASequence

        psm = self._bare_psm()
        std = psm.convert_sequence_to_standard_format("ALLSSSaVLYK")
        assert std == "ALLSSSA(PhosphoDecoy)VLYK"
        # Previously this threw and the output silently kept the input sequence.
        assert AASequence.fromString(std).toString() == "ALLSSSA(PhosphoDecoy)VLYK"

    def test_sty_and_mixed_paths_unchanged(self):
        psm = self._bare_psm()
        # Pure S-phospho path is unaffected
        assert psm._get_mod_positions_from_perm("ALLsSSAVLYK") == ([3], [], False)
        assert (
            psm.convert_sequence_to_standard_format("ALLsSSAVLYK")
            == "ALLS(Phospho)SSAVLYK"
        )
        # S-phospho + A-decoy together
        phospho_pos, _, is_decoy = psm._get_mod_positions_from_perm("aLLsSSAVLYK")
        assert phospho_pos == [0, 3] and is_decoy is False


class TestLucXorSiteScores:
    """Per-site localization scores derived from permutation scores.

    For bigbio/onsite#40: LuciPHOr2 is natively per-PSM. get_site_scores()
    derives a per-site delta (best-with minus best-without) from the already
    computed real-permutation scores so a site-level decoy-AA FLR can rank
    individual sites. Higher = more confident.
    """

    def _psm_with_perms(self, perm_scores):
        from onsite.lucxor.psm import PSM

        psm = PSM.__new__(PSM)
        psm.pos_permutation_score_map = dict(perm_scores)
        return psm

    def test_ambiguous_sites_ranked_by_delta(self):
        # 3 candidate sites S@3 / T@4 / Y@5, one phospho; S clearly preferred.
        psm = self._psm_with_perms(
            {"PEPsTYK": 10.0, "PEPStYK": 4.0, "PEPSTyK": 1.0}
        )
        ss = psm.get_site_scores()
        assert ss[4] == 6.0  # 10 - max(4, 1)
        assert ss[5] == -6.0  # 4 - 10
        assert ss[6] == -9.0  # 1 - 10
        assert max(ss, key=ss.get) == 4  # winning site is most confident

    def test_decoy_a_site_is_rankable(self):
        # Decoy A@0 competes with S@3; both sites get a score.
        psm = self._psm_with_perms({"aEPSK": 2.0, "AEPsK": 9.0})
        ss = psm.get_site_scores()
        assert set(ss) == {1, 4}
        assert ss[1] == -7.0 and ss[4] == 7.0

    def test_unambiguous_site_gets_top_score(self):
        # Single candidate / single phospho: no alternative -> top score.
        psm = self._psm_with_perms({"PEPsK": 8.0})
        assert psm.get_site_scores() == {4: 8.0}

    def test_no_permutations_returns_empty(self):
        assert self._psm_with_perms({}).get_site_scores() == {}
