"""
PSM (Peptide-Spectrum Match) module.

This module contains the PSM class for handling peptide-spectrum matches.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Set, Any, Union
import os
from itertools import combinations, islice
import random
import re

import numpy as np
import pyopenms
from pyopenms import AASequence, ResidueModification

from .constants import (
    NTERM_MOD,
    CTERM_MOD,
    AA_MASSES,
    DECOY_AA_MAP,
    AA_DECOY_MAP,
    NEUTRAL_LOSSES,
    MIN_DELTA_SCORE,
    PROTON_MASS,
    WATER_MASS,
    DECOY_AMINO_ACIDS,
    PHOSPHO_MOD_MASS,
    OXIDATION_MASS,
)
from .peak import Peak
from .peptide import Peptide, extract_target_amino_acids
from .spectrum import Spectrum
from .flr import FLRCalculator
from .globals import get_decoy_symbol

logger = logging.getLogger(__name__)


class PSM:
    """
    Class representing a peptide-spectrum match.
    """

    def __init__(
        self,
        peptide: Peptide,
        spectrum_source: Union[str, Dict, Spectrum],
        scan_num: Optional[int] = None,
        config: Dict = None,
    ):
        """
        Initialize PSM.

        Args:
            peptide: Peptide object
            spectrum_source: Either a path to spectrum file (str), a spectrum dictionary, or a Spectrum object
            scan_num: Scan number (required if spectrum_source is a file path)
            config: Configuration dictionary
        """
        self.peptide = peptide
        self.config = config or {}
        self.scan_num = scan_num

        # Score related
        self.delta_score = 0.0
        self.psm_score = 0.0  # Original search engine score
        self.score = 0.0  # Luciphor score
        self.global_flr = 1.0  # Global false localization rate
        self.local_flr = 1.0  # Local false localization rate
        self.is_decoy = False

        # Permutation related
        self.pos_permutation_score_map = {}
        self.neg_permutation_score_map = {}
        self.score1_pep = None
        self.score2_pep = None

        # Flags
        self.is_keeper = False
        self.use_for_model = False
        self.is_unambiguous = False  # Add is_unambiguous attribute

        # Modification related
        self.mod_coord_map = {}  # Modification site -> mass
        self.non_target_mods = {}  # Non-target modifications
        self.mod_sites = []
        self.mod_masses = []
        self.target_mod_map = {}  # Add target_mod_map
        self.mod_pos_map = {}  # Initialize mod_pos_map
        self._init_modifications()

        # Process spectrum data
        if isinstance(spectrum_source, str):
            self.spectrum_file = spectrum_source
            self.spectrum = None
            if scan_num is None:
                raise ValueError(
                    "scan_num is required when spectrum_source is a file path"
                )
            self._load_spectrum()
        elif isinstance(spectrum_source, dict):
            self.spectrum_file = None
            # Create Spectrum object from dictionary
            self.spectrum = Spectrum(
                mz_array=spectrum_source.get("mz"),
                intensity_array=spectrum_source.get("intensities"),
            )
            self.spectrum_native_id = spectrum_source.get("native_id", None)
            self.spectrum_rt = spectrum_source.get("rt", None)
            if not self.spectrum.is_empty():
                # Process spectrum data
                if self.config.get("reduce_nl", False):
                    self._reduce_nl_peak()
                # Normalize spectrum intensity
                self.spectrum.median_normalize_spectra()
        else:
            self.spectrum_file = None
            self.spectrum = spectrum_source
            if not self.spectrum.is_empty():
                # Process spectrum data
                if self.config.get("reduce_nl", False):
                    self._reduce_nl_peak()
                # Normalize spectrum intensity
                self.spectrum.median_normalize_spectra()

        # FLR calculator will be passed from external, not initialized here
        self.flr_calculator = None

    @classmethod
    def from_peptide_id(cls, peptide_id, peptide_hit, spectrum, config: Dict = None):
        """
        Create PSM object from peptide_id and peptide_hit

        Args:
            peptide_id: PeptideIdentification object
            peptide_hit: PeptideHit object
            spectrum: Spectrum object
            config: Configuration dictionary

        Returns:
            PSM object
        """
        # Get sequence and charge from peptide_hit
        sequence = peptide_hit.getSequence().toString()
        charge = peptide_hit.getCharge()
        score = peptide_hit.getScore()

        # Create Peptide object
        peptide = Peptide(sequence, config, charge=charge)

        # Create PSM object
        psm = cls(peptide, spectrum, config=config)
        psm.psm_score = score

        # Save original peptide identification data
        psm.peptide_id = peptide_id
        psm.peptide_hit = peptide_hit

        # Store search_engine_sequence for later use in idXML output
        psm.search_engine_sequence = sequence

        return psm

    def _load_spectrum(self) -> None:
        """Load spectrum data from file"""
        if not os.path.exists(self.spectrum_file):
            logger.error(f"Spectrum file not found: {self.spectrum_file}")
            return

        try:
            # Determine format based on file extension
            file_format = self.spectrum_file.lower().split(".")[-1]

            # Read spectrum
            self.spectrum = Spectrum.from_file(
                self.spectrum_file, self.scan_num, file_format
            )

            if self.spectrum.is_empty():
                logger.warning(f"No spectrum data found for scan {self.scan_num}")
            else:
                logger.debug(f"Loaded spectrum with {self.spectrum.N} peaks")

                # Process spectrum data
                if self.config.get("reduce_nl", False):
                    self._reduce_nl_peak()

                # Normalize spectrum intensity
                self.spectrum.median_normalize_spectra()

        except Exception as e:
            logger.error(f"Error loading spectrum: {str(e)}")
            self.spectrum = Spectrum()

    def _reduce_nl_peak(self) -> None:
        """Reduce neutral loss peak intensity"""
        if self.spectrum is None or self.spectrum.is_empty():
            return

        try:
            # Calculate neutral loss mass of precursor ion
            precursor_mass = self.peptide.get_precursor_mass()
            nl_mass = self.config.get("neutral_loss_mass", 0.0)

            if nl_mass > 0:
                # Find neutral loss peak in spectrum
                nl_mz = precursor_mass - nl_mass
                for i in range(self.spectrum.N):
                    if abs(self.spectrum.mz[i] - nl_mz) < 0.5:  # Use 0.5 Da window
                        # Reduce intensity to 10% of original
                        self.spectrum.raw_intensity[i] *= 0.1

                # Recalculate relative intensity
                self.spectrum.calc_relative_intensity()

        except Exception as e:
            logger.error(f"Error reducing neutral loss peak: {str(e)}")

    def _init_modifications(self):
        """Initialize modification information."""
        try:
            # Get the peptide sequence
            seq = self.peptide.peptide

            # Initialize modification maps
            self.mod_coord_map = {}
            self.non_target_mods = {}
            self.target_mod_map = {}  # Initialize target_mod_map
            self.mod_pos_map = {}  # Initialize mod_pos_map

            # Initialize target_mod_map, add S, T, Y as target modification sites
            self.target_mod_map = {"S": True, "T": True, "Y": True}

            # Process N-terminal modifications
            if seq.startswith("["):
                self.mod_coord_map[NTERM_MOD] = 0.0
                self.mod_pos_map[NTERM_MOD] = (
                    0.0  # Add N-terminal modification position
                )
                seq = seq[1:]

            # Process C-terminal modifications
            if seq.endswith("]"):
                self.mod_coord_map[CTERM_MOD] = 0.0
                self.mod_pos_map[CTERM_MOD] = len(
                    seq
                )  # Add C-terminal modification position
                seq = seq[:-1]

            # Process internal modifications - use same logic as Peptide class
            mod_pep_pos = ""  # Used to track modified peptide length
            i = 0
            while i < len(seq):
                if i + 9 <= len(seq) and seq[i : i + 9] == "(Phospho)":
                    if len(mod_pep_pos) > 0:
                        pos = len(mod_pep_pos) - 1
                        self.mod_coord_map[pos] = PHOSPHO_MOD_MASS
                        self.mod_pos_map[pos] = PHOSPHO_MOD_MASS
                    i += 9
                elif i + 14 <= len(seq) and seq[i : i + 14] == "(PhosphoDecoy)":
                    # Treat PhosphoDecoy as phosphorylation modification
                    if len(mod_pep_pos) > 0:
                        pos = len(mod_pep_pos) - 1
                        self.mod_coord_map[pos] = PHOSPHO_MOD_MASS
                        self.mod_pos_map[pos] = PHOSPHO_MOD_MASS
                    i += 14
                elif i + 11 <= len(seq) and seq[i : i + 11] == "(Oxidation)":
                    if len(mod_pep_pos) > 0:
                        pos = len(mod_pep_pos) - 1
                        self.mod_coord_map[pos] = OXIDATION_MASS
                        self.mod_pos_map[pos] = OXIDATION_MASS
                        self.non_target_mods[pos] = mod_pep_pos[-1].lower()
                    i += 11
                else:
                    mod_pep_pos += seq[i]
                    i += 1

            # Update the peptide sequence
            self.peptide_sequence = self.peptide.get_unmodified_sequence()

        except Exception as e:
            logger.error(f"Error initializing modification information: {str(e)}")
            raise

    def _extract_scan_number(self, spectrum_data: Dict) -> int:
        """
        Extract scan number from spectrum data.

        Args:
            spectrum_data: Spectrum data dictionary

        Returns:
            Scan number
        """
        try:
            # Try to extract scan number from spectrum ID
            spectrum_id = spectrum_data.get("native_id", "")
            if "scan=" in spectrum_id:
                scan_str = spectrum_id.split("scan=")[1].split()[0]
                return int(scan_str)
            elif "index=" in spectrum_id:
                scan_str = spectrum_id.split("index=")[1].split()[0]
                return int(scan_str)
            else:
                logger.warning(
                    f"Could not extract scan number from spectrum ID: {spectrum_id}"
                )
                return 0
        except Exception as e:
            logger.error(f"Error extracting scan number: {str(e)}")
            return 0

    def _get_modified_peptide(self) -> str:
        """
        Get modified peptide string.

        Returns:
            Modified peptide string
        """
        try:
            # Create AASequence
            seq = AASequence()

            # Get peptide sequence
            peptide_seq = self.peptide.peptide

            # Add amino acids and modifications
            i = 0
            while i < len(peptide_seq):
                if peptide_seq[i : i + 9] == "(Phospho)":
                    # Add phosphorylation modification
                    mod = ResidueModification()
                    mod.setMonoMass(79.966331)
                    seq.setModification(i - 1, mod)
                    i += 9
                elif peptide_seq[i : i + 14] == "(PhosphoDecoy)":
                    # Add decoy phosphorylation modification (treat as phosphorylation)
                    mod = ResidueModification()
                    mod.setMonoMass(79.966331)
                    seq.setModification(i - 1, mod)
                    i += 14
                elif peptide_seq[i : i + 9] == "(Oxidation)":
                    # Add oxidation modification
                    mod = ResidueModification()
                    mod.setMonoMass(15.994915)
                    seq.setModification(i - 1, mod)
                    i += 9
                else:
                    # Add amino acid
                    seq += peptide_seq[i]
                    i += 1

            return seq.toString()
        except Exception as e:
            logger.error(f"Error creating modified peptide: {str(e)}")
            return self.peptide.peptide

    def process(self, config: Optional[Dict] = None, round_number: int = 0) -> None:
        """
        Process PSM and calculate scores.

        Args:
            config: Optional configuration dictionary
            round_number: Calculation round (0: first round includes decoys, 1: second round only real permutations)
        """
        # Update config if provided
        if config:
            self.config.update(config)

        # Skip if no phosphorylation (including PhosphoDecoy)
        if (
            "(Phospho)" not in self.peptide.peptide
            and "(PhosphoDecoy)" not in self.peptide.peptide
        ):
            self.is_keeper = False
            self.use_for_model = False
            self.psm_score = -1.0
            self.delta_score = -1.0
            logger.debug(f"Skipping non-phosphorylated peptide: {self.peptide.peptide}")
            return

        # Check spectrum data
        if self.spectrum is None or self.spectrum.is_empty():
            logger.warning("No valid spectrum data available")
            self.psm_score = 0.0
            self.delta_score = 0.0
            return

        # Calculate number of potential PTM sites
        # Extract target amino acids from target_modifications
        target_modifications = self.config.get("target_modifications", [])
        target_amino_acids = extract_target_amino_acids(target_modifications)

        potential_ptm_sites = sum(
            1 for aa in self.peptide.peptide if aa in target_amino_acids
        )
        reported_ptm_sites = self.peptide.peptide.count(
            "(Phospho)"
        ) + self.peptide.peptide.count("(PhosphoDecoy)")

        # Set is_unambiguous flag
        self.is_unambiguous = potential_ptm_sites == reported_ptm_sites

        # Set is_keeper and use_for_model
        keeping_score = 0

        # Check potential PTM sites > 0
        if potential_ptm_sites > 0:
            keeping_score += 1

        # Check reported PTM sites > 0
        if reported_ptm_sites > 0:
            keeping_score += 1

        # Check score >= scoring threshold
        score_threshold = self.config.get("scoring_threshold", 0.9)
        if self.psm_score >= score_threshold:
            keeping_score += 1

        # Check charge <= max charge state
        max_charge_state = self.config.get("max_charge_state", 5)
        if self.charge <= max_charge_state:
            keeping_score += 1

        # Check peptide length <= max peptide length
        max_pep_len = self.config.get("max_pep_len", 50)
        if len(self.peptide.peptide) <= max_pep_len:
            keeping_score += 1

        # Set is_keeper
        if keeping_score == 5:
            self.is_keeper = True

        # Set use_for_model
        modeling_threshold = self.config.get("modeling_score_threshold", 0.95)
        if self.is_keeper and (self.psm_score >= modeling_threshold):
            self.use_for_model = True
        else:
            self.use_for_model = False

        # Generate permutations based on round number
        self.generate_permutations(run_number=round_number)

        # Score permutations
        model = self.config.get("model", None)
        if model is None and round_number == 0:
            logger.warning("No model provided for scoring")
            self.psm_score = 0.0
            self.delta_score = 0.0
            return

        # Check model status
        logger.debug(f"Model type: {type(model)}")
        if hasattr(model, "charge_models"):
            logger.debug(f"Available charge models: {list(model.charge_models.keys())}")

        # Add more debug information
        logger.debug(f"Processing PSM with charge: {self.charge}")
        logger.debug(f"Peptide sequence: {self.peptide.peptide}")
        logger.debug(f"Number of permutations: {len(self.pos_permutation_score_map)}")

        self.score_permutations(model)

        # Calculate delta score, decide whether to include decoys based on round
        include_decoys = round_number == 0
        self.calculate_delta_score(include_decoys=include_decoys)

        # Calculate FLR, only using PSM's deltaScore
        if (
            hasattr(self, "flr_calculator")
            and self.flr_calculator is not None
            and round_number == 0
        ):
            # Only add current PSM's deltaScore
            if self.delta_score > self.flr_calculator.min_delta_score:
                self.flr_calculator.add_psm(self.delta_score, self.is_decoy)
                logger.debug(
                    f"Added PSM to FLR calculator - delta_score: {self.delta_score:.6f}, is_decoy: {self.is_decoy}"
                )
            else:
                logger.debug(
                    f"PSM delta_score {self.delta_score:.6f} <= {self.flr_calculator.min_delta_score}, skipping FLR calculation"
                )

            # Note: FLR calculation should be performed uniformly after all PSMs are processed, not calculated separately for each PSM
            # Here we just add PSM to FLR calculator, actual FLR calculation is done in core.py
        else:
            if round_number == 1:
                # Second round calculation: FLR values will be assigned later through mapping relationship
                pass
            else:
                self.global_flr = 1.0
                self.local_flr = 1.0

        # Save final scores - use values already calculated in score_permutations
        # Only recalculate if score_permutations didn"t calculate successfully
        if self.delta_score == 0.0 and self.psm_score == 0.0:
            real_scores = list(self.pos_permutation_score_map.values())
            if real_scores:
                sorted_real = sorted(real_scores, reverse=True)
                self.psm_score = sorted_real[0]
                self.delta_score = (
                    sorted_real[0] - sorted_real[1]
                    if len(sorted_real) > 1
                    else sorted_real[0]
                )
            else:
                self.psm_score = 0.0
                self.delta_score = 0.0
        self.score = self.delta_score
        logger.debug(
            f"PSM processing completed (round {round_number}) - Delta Score: {self.delta_score}, PepScore: {self.psm_score}"
        )

    def process_round2(self, flr_calculator) -> None:
        """
        Second round processing: recalculate delta score excluding decoys, and use first round FLR mapping relationship

        Args:
            flr_calculator: FLR calculator obtained from first round calculation, containing delta score to FLR mapping relationship
        """
        logger.debug(f"Second round processing PSM: {self.peptide.peptide}")

        # Regenerate permutations (only including real permutations)
        self.generate_permutations(run_number=1)

        # Rescore permutations
        model = self.config.get("model", None)
        self.score_permutations(model)

        # Recalculate delta score (excluding decoys)
        self.calculate_delta_score(include_decoys=False)

        # Use first round FLR mapping relationship to assign FLR values
        if flr_calculator and hasattr(flr_calculator, "find_closest_flr"):
            if self.delta_score > flr_calculator.min_delta_score:
                global_flr, local_flr = flr_calculator.find_closest_flr(
                    self.delta_score
                )
                self.global_flr = global_flr
                self.local_flr = local_flr
            else:
                # For PSMs with delta_score <= min_delta_score, set default values
                self.global_flr = 1.0
                self.local_flr = 1.0
        else:
            # If no FLR calculator, set default values
            self.global_flr = 1.0
            self.local_flr = 1.0

        logger.debug(
            f"Second round processing completed - Delta Score: {self.delta_score:.6f}, Global FLR: {self.global_flr:.6f}, Local FLR: {self.local_flr:.6f}"
        )

    def generate_permutations_stage2(self) -> None:
        """
        Second stage permutation generation: only generate real permutations, not decoy permutations
        """
        logger.debug(f"Second stage generating permutations for PSM {self.scan_num}")

        # Clear previous permutations
        self.pos_permutation_score_map.clear()
        self.neg_permutation_score_map.clear()

        # Only generate real permutations
        real_permutations = self.generate_real_permutations()

        logger.debug(
            f"Second stage generated {len(real_permutations)} real permutations"
        )

        # Store real permutations (don't generate decoy permutations)
        for perm, sites in real_permutations:
            self.pos_permutation_score_map[perm] = (
                0.0  # Score will be calculated in scoring stage
            )

    def _validate_permutation(self, perm: str) -> bool:
        """
        Validate permutation validity

        Args:
            perm: Permutation string

        Returns:
            bool: Whether the permutation is valid
        """
        try:
            # Get unmodified sequence length
            def get_unmodified_length(seq: str) -> int:
                length = 0
                i = 0
                while i < len(seq):
                    if seq[i : i + 9] == "(Phospho)":
                        i += 9
                    elif seq[i : i + 14] == "(PhosphoDecoy)":
                        i += 14
                    elif seq[i : i + 11] == "(Oxidation)":
                        i += 11
                    else:
                        length += 1
                        i += 1
                return length

            # Check if unmodified sequence length is correct
            perm_unmod_len = get_unmodified_length(perm)
            orig_unmod_len = get_unmodified_length(self.peptide.peptide)

            if perm_unmod_len != orig_unmod_len:
                logger.debug(
                    f"Unmodified sequence length mismatch: {perm_unmod_len} != {orig_unmod_len}"
                )
                return False

            # Check if modification site count is correct
            phospho_count = perm.count("(Phospho)")
            decoy_count = sum(1 for c in perm if c in DECOY_AA_MAP)
            # Expected total phospho-like sites include both target and decoy phospho markers
            expected_count = self.peptide.peptide.count(
                "(Phospho)"
            ) + self.peptide.peptide.count("(PhosphoDecoy)")

            if phospho_count + decoy_count != expected_count:
                logger.debug(
                    f"Phosphorylation count mismatch: {phospho_count + decoy_count} != {expected_count}"
                )
                return False

            return True
        except Exception as e:
            logger.error(f"Error validating permutation {perm}: {str(e)}")
            return False

    def score_permutations(self, model: Optional[object]) -> None:
        """
        Score all permutations of the peptide sequence

        Args:
            model: Optional scoring model
        """
        try:
            # Get all permutations from pos_permutation_score_map and neg_permutation_score_map
            all_perms = list(self.pos_permutation_score_map.keys()) + list(
                self.neg_permutation_score_map.keys()
            )
            base_tolerance = self.config.get(
                "fragment_mass_tolerance", 0.1
            )  # Read from config, default 0.1

            # Score each permutation
            for perm in all_perms:
                # Check if it"s a decoy sequence
                is_decoy = self._is_decoy_sequence(perm)

                if is_decoy:
                    tolerance = base_tolerance * 2.0  # decoyPadding = 2.0
                else:
                    tolerance = base_tolerance

                # Calculate score
                final_score = 0.0
                matched_peaks = self._match_peaks(perm, tolerance)

                # Get charge model
                charge_model = model.get_charge_model(self.charge) if model else None

                if charge_model is None:
                    logger.debug(f"No charge model found for charge {self.charge}")
                    continue

                # Calculate score for each peak
                for peak in matched_peaks:
                    # Get ion type
                    ion_type = peak.get("matched_ion_str", "")

                    # Calculate intensity score
                    intensity_m = 0.0
                    intensity_u = 0.0

                    if ion_type.startswith("b"):
                        intensity_m = model.get_log_np_density_int(
                            "b", peak["norm_intensity"], self.charge
                        )
                    elif ion_type.startswith("y"):
                        intensity_m = model.get_log_np_density_int(
                            "y", peak["norm_intensity"], self.charge
                        )

                    intensity_u = model.get_log_np_density_int(
                        "n", peak["norm_intensity"], self.charge
                    )

                    # Calculate distance score
                    dist_m = 0.0
                    dist_u = 0.0

                    if ion_type.startswith("b"):
                        dist_m = model.get_log_np_density_dist_pos(
                            peak["mass_diff"], self.charge
                        )
                    elif ion_type.startswith("y"):
                        dist_m = model.get_log_np_density_dist_pos(
                            peak["mass_diff"], self.charge
                        )

                    # For unmatched ions, use uniform distribution
                    dist_u = 0.0

                    # Calculate score
                    intensity_score = intensity_m - intensity_u
                    distance_score = dist_m - dist_u

                    # Handle invalid values
                    if np.isnan(intensity_score) or np.isinf(intensity_score):
                        intensity_score = 0.0
                    if np.isnan(distance_score) or np.isinf(distance_score):
                        distance_score = 0.0

                    # Calculate final score: direct addition
                    peak_score = intensity_score + distance_score

                    # If score is negative, set to 0
                    if peak_score < 0:
                        peak_score = 0.0

                    # Add to total score
                    final_score += peak_score

                logger.debug(
                    f"Scoring permutation: {perm}, Score: {final_score:.6f}, Is decoy: {is_decoy}"
                )

                # Store scores
                if is_decoy:
                    self.neg_permutation_score_map[perm] = final_score
                    logger.debug(f"Stored decoy score: {perm} -> {final_score:.6f}")
                else:
                    self.pos_permutation_score_map[perm] = final_score
                    logger.debug(f"Stored real score: {perm} -> {final_score:.6f}")

            # Collect all scores into a list
            all_scores = []
            for score in self.pos_permutation_score_map.values():
                all_scores.append(score)

            # Only add decoy scores in non-unambiguous cases
            if not self.is_unambiguous:
                for score in self.neg_permutation_score_map.values():
                    all_scores.append(score)

            # Sort (from low to high, then reverse)
            all_scores.sort()  # From low to high
            all_scores.reverse()  # From high to low

            # Get first two scores, using 6 decimal precision
            score1 = self._round_dbl(all_scores[0], 6) if all_scores else 0.0
            score2 = 0.0
            if not self.is_unambiguous:
                score2 = (
                    self._round_dbl(all_scores[1], 6) if len(all_scores) > 1 else 0.0
                )

            # Find corresponding peptide sequence
            pep1 = ""
            pep2 = ""
            num_assigned = 0

            if not self.is_unambiguous:
                # First find highest and second highest scores in non-decoy permutations
                for p, s in self.pos_permutation_score_map.items():
                    x = s
                    d = self._round_dbl(x, 6)

                    if (d == score1) and (pep1 == ""):
                        pep1 = p
                        num_assigned += 1
                    elif (d == score2) and (pep2 == ""):
                        pep2 = p
                        num_assigned += 1

                    if num_assigned == 2:
                        break

                # If not found, search in decoy permutations
                if num_assigned != 2:
                    for p, s in self.neg_permutation_score_map.items():
                        x = s
                        d = self._round_dbl(x, 6)

                        if (d == score1) and (pep1 == ""):
                            pep1 = p
                            num_assigned += 1
                        elif (d == score2) and (pep2 == ""):
                            pep2 = p
                            num_assigned += 1

                        if num_assigned == 2:
                            break
            else:
                # Special handling for unambiguous cases
                for p, s in self.pos_permutation_score_map.items():
                    pep1 = p
                    break

            # Calculate delta score
            if self.is_unambiguous:
                # Unambiguous case: deltaScore = score1
                self.delta_score = score1
                logger.debug(f"Unambiguous case, delta score: {self.delta_score}")
            else:
                # Ambiguous case: deltaScore = score1 - score2
                self.delta_score = score1 - score2
                logger.debug(
                    f"Non-unambiguous case, delta score: {score1} - {score2} = {self.delta_score}"
                )

            # Set psm_score to score1
            self.psm_score = score1

            self.score1_pep = pep1
            self.score2_pep = pep2

            # Set is_decoy flag based on highest scoring peptide
            if pep1:
                self.is_decoy = self._is_decoy_sequence(pep1)

        except Exception as e:
            logger.error(f"Error scoring permutations: {str(e)}")
            self.delta_score = 0.0
            self.psm_score = 0.0

    def _kill_thread_results(self):
        """Handle scoring failure cases"""
        self.delta_score = 0.0
        self.score = -1.0
        self.psm_score = -1.0
        self.score1_pep = self.peptide
        self.score2_pep = self.peptide

    def _round_dbl(self, value: float, num_places: int) -> float:
        """
        Round to specified decimal places

        Args:
            value: Value to round
            num_places: Number of decimal places

        Returns:
            Rounded value
        """
        n = math.pow(10, num_places)
        ret = round(value * n) / n
        return ret

    def _is_decoy_sequence(self, sequence: str) -> bool:
        """
        Check if a sequence is a decoy sequence.

        Args:
            sequence: Peptide sequence to check

        Returns:
            bool: True if sequence is a decoy sequence
        """
        try:
            # First remove modification markers, then check for decoy amino acids
            unmod_seq = ""
            i = 0
            while i < len(sequence):
                if sequence[i : i + 9] == "(Phospho)":
                    i += 9
                elif sequence[i : i + 14] == "(PhosphoDecoy)":
                    i += 14
                elif sequence[i : i + 11] == "(Oxidation)":
                    i += 11
                else:
                    unmod_seq += sequence[i]
                    i += 1

            # Check if sequence contains decoy amino acid symbols after removing modifications
            for aa in unmod_seq:
                if aa in DECOY_AA_MAP:
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking decoy sequence: {str(e)}")
            return False

    def _get_mod_map(self, perm: str) -> Dict[int, float]:
        """
        Get peptide modification site mapping

        Args:
            perm: Peptide permutation

        Returns:
            Dict[int, float]: Position -> modification mass
        """
        mod_map = {}

        # Handle lowercase letter format modifications (new format)
        for i, aa in enumerate(perm):
            if aa.islower() and aa.upper() in ["S", "T", "Y"]:
                # Lowercase letters indicate phosphorylation modification sites
                mod_map[i] = PHOSPHO_MOD_MASS
            elif aa.islower() and aa.upper() in ["M", "W", "F", "Y"]:
                # Lowercase letters indicate oxidation modification sites
                mod_map[i] = OXIDATION_MASS
            elif aa in DECOY_AA_MAP:
                # Handle decoy amino acids
                # In Java, decoy amino acids already have extra mass, no need to add modification mass
                # Because decoy amino acid mass already includes DECOY_MASS in AA_MASSES
                pass  # Don't add extra modification mass

        # Handle bracket format modifications (compatibility with old format)
        i = 0
        while i < len(perm):
            if perm[i : i + 9] == "(Phospho)":
                # Get original amino acid at modification site
                orig_aa = perm[i - 1].upper()  # Convert to uppercase
                if orig_aa in AA_MASSES:
                    mod_map[i - 1] = PHOSPHO_MOD_MASS
                i += 9
            elif perm[i : i + 11] == "(Oxidation)":
                # Get original amino acid at modification site
                orig_aa = perm[i - 1].upper()
                if orig_aa in AA_MASSES:
                    mod_map[i - 1] = OXIDATION_MASS
                i += 11
            else:
                i += 1

        return mod_map

    def _get_aa_mass(self, aa: str) -> float:
        """
        Get amino acid mass

        Args:
            aa: Amino acid

        Returns:
            float: Amino acid mass
        """
        # Convert to uppercase and check
        aa_upper = aa.upper()
        if aa_upper in AA_MASSES:
            return AA_MASSES[aa_upper]
        return 110.0  # Default mass

    def _calc_theoretical_masses(self, perm: str) -> List[float]:
        """
        Calculate theoretical masses

        Args:
            perm: Peptide permutation

        Returns:
            List of theoretical masses
        """
        if not perm:
            return []

        # Get modification site mapping
        mod_map = self._get_mod_map(perm)

        masses = []

        # Calculate b ions
        current_mass = 0.0
        for i in range(len(perm)):
            if perm[i] not in ["(", ")"]:  # Skip modification markers
                # Handle amino acids, including lowercase letters (modification sites)
                if perm[i].upper() in AA_MASSES:
                    current_mass += self._get_aa_mass(perm[i].upper())
                elif perm[i] in DECOY_AA_MAP:
                    # Handle decoy amino acids - directly use decoy amino acid mass
                    # Decoy amino acid mass already includes DECOY_MASS in AA_MASSES
                    current_mass += self._get_aa_mass(perm[i])
            if i in mod_map:
                current_mass += mod_map[i]
            # Only add ions longer than minimum length
            if i >= 1:  # Minimum length is 2
                # Only generate ions with charge less than peptide charge
                for z in range(
                    1, self.charge
                ):  # Only generate charge states 1 to charge-1
                    ion_mass = (current_mass + PROTON_MASS) / z
                    if ion_mass < self.peptide.get_precursor_mass():
                        masses.append(ion_mass)

        # Calculate y ions
        current_mass = 0.0
        for i in range(len(perm) - 1, -1, -1):
            if perm[i] not in ["(", ")"]:  # Skip modification markers
                # Handle amino acids, including lowercase letters (modification sites)
                if perm[i].upper() in AA_MASSES:
                    current_mass += self._get_aa_mass(perm[i].upper())
                elif perm[i] in DECOY_AA_MAP:
                    # Handle decoy amino acids - directly use decoy amino acid mass
                    # Decoy amino acid mass already includes DECOY_MASS in AA_MASSES
                    current_mass += self._get_aa_mass(perm[i])
            if i in mod_map:
                current_mass += mod_map[i]
            # Only add ions longer than minimum length
            if i <= len(perm) - 2:  # Minimum length is 2
                # Only generate ions with charge less than peptide charge
                for z in range(
                    1, self.charge
                ):  # Only generate charge states 1 to charge-1
                    ion_mass = (current_mass + WATER_MASS + PROTON_MASS) / z
                    if ion_mass < self.peptide.get_precursor_mass():
                        masses.append(ion_mass)

        return masses

    def get_results(self) -> str:
        """Get result string"""
        results = []
        results.append(str(self.scan_num))
        # Use the best sequence if available, otherwise use original peptide sequence
        best_seq = getattr(self, "_best_sequence", None)
        if best_seq is None:
            best_seq = self.get_best_sequence(include_decoys=False)
            self._best_sequence = best_seq

        results.append(best_seq)
        results.append(f"{self.psm_score:.4f}")  # Use psm_score as peptide match score
        results.append(f"{self.delta_score:.4f}")  # delta_score for site localization
        results.append(f"{self.global_flr:.4f}")
        results.append(f"{self.local_flr:.4f}")
        results.append("1" if self.is_decoy else "0")
        return "\t".join(results)

    def normalize_spectrum(self) -> None:
        """Normalize spectrum intensities."""
        if self.spectrum is None:
            self.logger.warning("No spectrum data available for normalization")
            return

        # Get peaks
        mz_values, intensities = self.spectrum.get_peaks()

        if len(intensities) == 0:
            self.logger.warning("No intensity values available for normalization")
            return

        # Check data validity
        valid_mask = np.isfinite(intensities) & (intensities > 0)
        if not np.any(valid_mask):
            self.logger.warning("No valid intensity values found for normalization")
            return

        # Only use valid intensity values
        valid_intensities = intensities[valid_mask]
        valid_mz_values = mz_values[valid_mask]

        # Log intensity statistics
        self.logger.debug(
            f"Intensity stats before normalization: min={np.min(valid_intensities):.2f}, "
            f"max={np.max(valid_intensities):.2f}, mean={np.mean(valid_intensities):.2f}"
        )

        # Normalize to sum of 1
        total = np.sum(valid_intensities)
        if total > 0:
            normalized_intensities = valid_intensities / total

            # Update spectrum with normalized values
            self.spectrum.set_peaks(valid_mz_values, normalized_intensities)

            self.logger.debug(
                f"Normalized {len(normalized_intensities)} intensity values"
            )
        else:
            self.logger.warning("Total intensity is zero, skipping normalization")

    def reduce_nl_peak(self, precursor_nl_mass: float = 0.0) -> None:
        """Reduce neutral loss peaks.

        Args:
            precursor_nl_mass: Precursor neutral loss mass
        """
        if self.spectrum is None:
            self.logger.warning("No spectrum data available for neutral loss reduction")
            return

        # Get peaks
        mz_values, intensities = self.spectrum.get_peaks()

        if len(intensities) == 0:
            self.logger.warning(
                "No intensity values available for neutral loss reduction"
            )
            return

        # Find peaks to reduce
        nl_indices = []
        for i, mz in enumerate(mz_values):
            # Check if peak is within neutral loss mass window
            if abs(mz - precursor_nl_mass) < 0.5:  # Using 0.5 Da window
                nl_indices.append(i)

        if nl_indices:
            # Reduce intensity of neutral loss peaks
            for idx in nl_indices:
                intensities[idx] *= 0.1  # Reduce to 10% of original intensity

            # Update spectrum
            self.spectrum.set_peaks(mz_values, intensities)

            self.logger.debug(f"Reduced {len(nl_indices)} neutral loss peaks")
        else:
            self.logger.debug("No neutral loss peaks found")

    def get_spectrum_peaks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get spectrum peaks.

        Returns:
            Tuple of (m/z array, intensity array)
        """
        if self.spectrum is None:
            self.logger.warning("No spectrum data available")
            return np.array([]), np.array([])

        return self.spectrum.get_peaks()

    def clear_scores(self) -> None:
        """
        Clear PSM scores
        """
        logger.debug(f"Clearing scores for PSM {self.scan_num}")

        # Clear permutation score mappings
        self.pos_permutation_score_map.clear()
        self.neg_permutation_score_map.clear()

        # Reset score-related attributes
        self.delta_score = np.nan
        self.pep_score = np.nan
        self.global_flr = np.nan
        self.local_flr = np.nan

        # Clear peak data
        if hasattr(self, "pos_peaks"):
            self.pos_peaks.clear()
        if hasattr(self, "neg_peaks"):
            self.neg_peaks.clear()

        logger.debug(f"PSM {self.scan_num} scores cleared")

    def remove_modifications(self, seq):
        # Remove all bracket modifications like (Phospho), (PhosphoDecoy), (Oxidation), etc.
        return re.sub(r"\([A-Za-z0-9;@#%]+?\)", "", seq)

    def generate_real_permutations(self) -> List[Tuple[str, List[int]]]:
        """
        Generate real sequence permutations
        Returns:
            List[Tuple[str, List[int]]]: List of real sequences and modification sites
        """
        try:
            # Get unmodified peptide sequence
            peptide_seq = self.remove_modifications(
                self.peptide.get_unmodified_sequence()
            )

            # Find all possible modification sites (target modification sites)
            # Extract target amino acids from target_modifications
            target_modifications = self.config.get("target_modifications", [])
            target_amino_acids = extract_target_amino_acids(target_modifications)

            cand_mod_sites = []
            for i, aa in enumerate(peptide_seq):
                if aa in target_amino_acids:  # Target modification sites
                    cand_mod_sites.append(i)

            if not cand_mod_sites:
                logger.debug("No potential modification sites found")
                return []

            # DEBUG: Output mod_coord_map content
            logger.debug(f"mod_coord_map: {self.mod_coord_map}")
            logger.debug(f"Original peptide: {self.peptide.peptide}")
            logger.debug(f"Unmodified peptide_seq: {peptide_seq}")
            logger.debug(f"Candidate mod sites: {cand_mod_sites}")

            # Get the number of modification sites in the original peptide
            num_rps = len(
                [
                    m
                    for m in self.mod_coord_map.values()
                    if abs(m - PHOSPHO_MOD_MASS) < 0.01
                ]
            )
            logger.debug(f"num_rps calculated: {num_rps}")

            if num_rps == 0:
                logger.debug("No phosphorylation modifications found in peptide")
                return []

            # Generate all possible modification site combinations
            real_perms = []

            # Generate all possible modification site combinations
            for sites in combinations(cand_mod_sites, num_rps):
                # Create real sequence: modification sites represented by lowercase letters
                mod_pep = ""
                if NTERM_MOD in self.mod_coord_map:
                    mod_pep = "["

                for i, aa in enumerate(peptide_seq):
                    aa_lower = aa.lower()
                    if i in sites:  # Sites that need modification
                        mod_pep += aa_lower
                    elif i in self.non_target_mods:
                        mod_pep += aa_lower
                    else:
                        mod_pep += aa_lower.upper()

                if CTERM_MOD in self.mod_coord_map:
                    mod_pep += "]"

                # Avoid duplicates
                if mod_pep not in [x[0] for x in real_perms]:
                    real_perms.append((mod_pep, list(sites)))

            logger.debug(f"Generated {len(real_perms)} real permutations")
            for perm, sites in real_perms:
                logger.debug(f"  Real permutation: {perm} with sites: {sites}")
            return real_perms
        except Exception as e:
            logger.error(f"Error generating real permutations: {str(e)}")
            return []

    def generate_decoy_permutations(self) -> List[Tuple[str, List[int]]]:
        """
        Generate decoy sequence permutations at non-STY sites
        Returns:
            List[Tuple[str, List[int]]]: List of decoy sequences and modification sites
        """
        try:
            # Get unmodified peptide sequence
            peptide_seq = self.remove_modifications(
                self.peptide.get_unmodified_sequence()
            )

            # Extract target amino acids from target_modifications
            target_modifications = self.config.get("target_modifications", [])
            target_amino_acids = extract_target_amino_acids(target_modifications)

            # Find non-target amino acid sites as decoy modification sites
            cand_mod_sites = []
            for i, aa in enumerate(peptide_seq):
                if aa not in target_amino_acids:  # Non-target amino acid sites
                    cand_mod_sites.append(i)

            if not cand_mod_sites:
                logger.debug(
                    "No non-target amino acid sites available for decoy generation"
                )
                return []

            # Calculate the number of modifications in the original peptide
            num_mods = len(
                [
                    m
                    for m in self.mod_coord_map.values()
                    if abs(m - PHOSPHO_MOD_MASS) < 0.01
                ]
            )
            if num_mods == 0:
                logger.debug("No phosphorylation modifications found in peptide")
                return []

            # Generate all possible modification site combinations
            decoy_perms = []

            combos = list(combinations(cand_mod_sites, num_mods))
            random.shuffle(combos)

            # No limit on decoy count, use all possible combinations
            max_decoys = len(combos)

            for sites in islice(combos, max_decoys):
                # Create decoy sequence: decoy sites represented by decoy symbols
                mod_pep = ""
                for i, aa in enumerate(peptide_seq):
                    if i in sites:  # Sites that need modification
                        decoy_char = get_decoy_symbol(aa)  # Use decoy symbol
                        mod_pep += decoy_char
                    else:
                        mod_pep += (
                            aa.upper()
                        )  # Non-modification sites use uppercase letters

                # Check if it's a valid decoy sequence
                if self._is_valid_decoy_sequence(mod_pep):
                    decoy_perms.append((mod_pep, list(sites)))

            logger.debug(f"Generated {len(decoy_perms)} decoy permutations")
            for perm, sites in decoy_perms:
                logger.debug(f"  Decoy permutation: {perm} with sites: {sites}")
            return decoy_perms
        except Exception as e:
            logger.error(f"Error generating decoy permutations: {str(e)}")
            return []

    def _is_valid_decoy_sequence(self, sequence: str) -> bool:
        """
        Check if the sequence is a valid decoy sequence

        Args:
            sequence: Sequence to check

        Returns:
            bool: Returns True if it"s a valid decoy sequence
        """
        try:
            # Check if it contains decoy amino acid symbols
            for aa in sequence:
                if aa in DECOY_AA_MAP:
                    return True

            return False

        except Exception as e:
            logger.error(f"Error validating decoy sequence: {str(e)}")
            return False

    def _match_peaks(self, perm: str, tolerance: float) -> List[Dict[str, Any]]:
        """
        Match peaks in the spectrum to theoretical fragment ions.
        Now delegates to Peptide.match_peaks for consistency.

        Args:
            perm: Peptide permutation string
            tolerance: Mass tolerance for matching

        Returns:
            List of matched peaks
        """
        # Create temporary Peptide object

        # Get modification site mapping
        mod_map = self._get_mod_map(perm)

        # Use the input sequence directly, no format conversion needed
        # Because generate_real_permutations now generates correct lowercase format
        processed_perm = perm

        # Create temporary Peptide object and initialize
        temp_peptide = Peptide(processed_perm, self.config, charge=self.charge)
        temp_peptide.mod_pos_map = mod_map
        temp_peptide.build_ion_ladders()  # Build ion ladders

        # Temporarily modify configuration to use input tolerance
        original_tolerance = self.config.get("fragment_mass_tolerance", 0.1)
        temp_config = self.config.copy()
        temp_config["fragment_mass_tolerance"] = tolerance

        # Call Peptide's match_peaks method with modified configuration
        matched_peaks = temp_peptide.match_peaks(self.spectrum, temp_config)

        # Clean up temporary object
        temp_peptide = None

        return matched_peaks

    def is_decoy_permutation(self, sequence: str) -> bool:
        """
        Determine if the sequence is a decoy sequence

        Args:
            sequence: Peptide sequence

        Returns:
            bool: Whether it"s a decoy sequence
        """
        # Check if the sequence contains decoy marker characters
        decoy_chars = set("@#$%^&*()_+{}|:\"<>?~`-=[]\\;',./")
        return any(char in decoy_chars for char in sequence)

    @property
    def charge(self):
        """
        Get charge state
        """
        return getattr(self.peptide, "charge", None)

    def generate_permutations(self, run_number: int) -> None:
        """
        Generate permutations, control whether to include decoy based on run number

        Args:
            run_number: Run number (0: include decoy, 1: only include real sequences)
        """
        logger.debug(
            f"Generating permutations for PSM {self.scan_num}, run number: {run_number}"
        )

        # Clear previous permutations
        self.pos_permutation_score_map = {}
        self.neg_permutation_score_map = {}

        # Generate real sequence permutations
        real_perms = self.generate_real_permutations()
        for perm, mod_positions in real_perms:
            self.pos_permutation_score_map[perm] = 0.0

        # Decide whether to generate decoy permutations based on run number
        if not self.is_unambiguous and run_number == 0:
            # First iteration: generate decoy permutations for FLR estimation
            decoy_perms = self.generate_decoy_permutations()
            for perm, mod_positions in decoy_perms:
                self.neg_permutation_score_map[perm] = 0.0
            logger.debug(
                f"PSM {self.scan_num} generated {len(self.pos_permutation_score_map)} real permutations and {len(self.neg_permutation_score_map)} decoy permutations"
            )
        else:
            # Second iteration or unambiguous PSM: only generate real permutations
            logger.debug(
                f"PSM {self.scan_num} generated {len(self.pos_permutation_score_map)} real permutations (no decoy)"
            )

    def update_peptide_id(self) -> None:
        """
        Update peptide identification data, write FLR calculation results to original peptide_hit object

        This method will:
        1. Update peptide_hit score to delta_score
        2. Add FLR-related metadata
        3. Update hits in peptide_id
        """
        if not hasattr(self, "peptide_hit") or not hasattr(self, "peptide_id"):
            logger.warning(
                "PSM does not have original peptide_hit or peptide_id data, cannot update"
            )
            return

        try:
            # Update peptide_hit score to delta_score
            self.peptide_hit.setScore(self.delta_score)

            # Add FLR-related metadata
            self.peptide_hit.setMetaValue("Luciphor_delta_score", self.delta_score)
            self.peptide_hit.setMetaValue(
                "target_decoy", "decoy" if self.is_decoy else "target"
            )

            # Add search_engine_sequence information
            # This should be the original sequence from the search engine
            if hasattr(self, "search_engine_sequence"):
                self.peptide_hit.setMetaValue(
                    "search_engine_sequence", self.search_engine_sequence
                )
            else:
                # If not available, use the original peptide sequence
                self.peptide_hit.setMetaValue(
                    "search_engine_sequence", self.peptide.peptide
                )

            self.peptide_hit.setMetaValue(
                "Luciphor_pep_score", self.psm_score
            )  # Use psm_score as pep_score
            self.peptide_hit.setMetaValue("Luciphor_global_flr", self.global_flr)
            self.peptide_hit.setMetaValue("Luciphor_local_flr", self.local_flr)

            # Update peptide_id score type and attributes
            self.peptide_id.setScoreType("Luciphor_delta_score")
            self.peptide_id.setHigherScoreBetter(True)
            self.peptide_id.setSignificanceThreshold(0.0)

            # Update hits in peptide_id
            hits = self.peptide_id.getHits()
            for i in range(len(hits)):
                if hits[i].getSequence().toString() == self.peptide.peptide:
                    hits[i] = self.peptide_hit
                    break

            self.peptide_id.setHits(hits)

            logger.debug(
                f"Updated peptide identification data for PSM {self.scan_num}, delta_score: {self.delta_score:.6f}, global_flr: {self.global_flr:.6f}, local_flr: {self.local_flr:.6f}"
            )

        except Exception as e:
            logger.error(
                f"Error updating peptide identification data for PSM {self.scan_num}: {str(e)}"
            )
            raise

    def convert_sequence_to_standard_format(self, sequence: str) -> str:
        """
        Convert sequence from lowercase modification format to standard (Phospho) format

        Args:
            sequence: Sequence with lowercase letters indicating modifications

        Returns:
            str: Sequence in standard format with (Phospho) modifications
        """
        try:
            if not sequence:
                return sequence

            result = ""
            i = 0
            while i < len(sequence):
                if sequence[i].islower() and sequence[i].upper() in ["S", "T", "Y"]:
                    # Convert lowercase to uppercase and add (Phospho)
                    result += sequence[i].upper() + "(Phospho)"
                elif sequence[i].islower() and sequence[i].upper() in [
                    "M",
                    "W",
                    "F",
                    "Y",
                ]:
                    # Convert lowercase to uppercase and add (Oxidation)
                    result += sequence[i].upper() + "(Oxidation)"
                else:
                    result += sequence[i]
                i += 1

            return result
        except Exception as e:
            logger.error(f"Error converting sequence format: {str(e)}")
            return sequence

    def get_best_sequence(self, include_decoys: bool = True) -> str:
        """
        Get the best scoring sequence from permutations

        Args:
            include_decoys: Whether to include decoy permutations (True: first round calculation, False: second round calculation)

        Returns:
            str: The best scoring sequence, or original sequence if no permutations available
        """
        try:
            # Get scores for all real permutations
            real_scores = list(self.pos_permutation_score_map.values())

            if len(real_scores) == 0:
                logger.debug(
                    f"PSM {self.scan_num} has no real permutation scores, returning original sequence"
                )
                return self.peptide.peptide

            # Find the best scoring real permutation
            best_real_perm = max(
                self.pos_permutation_score_map.items(), key=lambda x: x[1]
            )
            best_real_score = best_real_perm[1]

            if include_decoys and len(self.neg_permutation_score_map) > 0:
                # Check if any decoy permutation has higher score
                best_decoy_perm = max(
                    self.neg_permutation_score_map.items(), key=lambda x: x[1]
                )
                best_decoy_score = best_decoy_perm[1]

                if best_decoy_score > best_real_score:
                    logger.debug(
                        f"PSM {self.scan_num} best sequence is decoy: {best_decoy_perm[0]} (score: {best_decoy_score:.6f})"
                    )
                    return self.convert_sequence_to_standard_format(best_decoy_perm[0])
                else:
                    logger.debug(
                        f"PSM {self.scan_num} best sequence is real: {best_real_perm[0]} (score: {best_real_score:.6f})"
                    )
                    return self.convert_sequence_to_standard_format(best_real_perm[0])
            else:
                logger.debug(
                    f"PSM {self.scan_num} best sequence is real: {best_real_perm[0]} (score: {best_real_score:.6f})"
                )
                return self.convert_sequence_to_standard_format(best_real_perm[0])

        except Exception as e:
            logger.error(
                f"Error getting best sequence for PSM {self.scan_num}: {str(e)}"
            )
            return self.peptide.peptide

    def calculate_delta_score(self, include_decoys: bool = True) -> None:
        """
        Calculate delta score based on permutation scores

        Args:
            include_decoys: Whether to include decoy permutations (True: first round calculation, False: second round calculation)


        """
        try:
            # Get scores for all real permutations
            real_scores = list(self.pos_permutation_score_map.values())

            if len(real_scores) == 0:
                logger.debug(
                    f"PSM {self.scan_num} has no real permutation scores, setting delta_score to 0"
                )
                self.delta_score = 0.0
                return

            real_scores.sort(reverse=True)

            # For unambiguous cases, delta_score should equal the top real score
            if self.is_unambiguous:
                top_score = real_scores[0]
                self.delta_score = top_score
                self.psm_score = top_score
                logger.debug(
                    f"PSM {self.scan_num} unambiguous: setting delta_score = top real score {top_score:.6f}"
                )
                return

            if include_decoys and len(self.neg_permutation_score_map) > 0:
                all_scores = real_scores + list(self.neg_permutation_score_map.values())
                all_scores.sort(reverse=True)

                if len(all_scores) == 1:
                    # Only one permutation overall: treat as unambiguous
                    self.delta_score = all_scores[0]
                    self.psm_score = all_scores[0]
                    logger.debug(
                        f"PSM {self.scan_num} single permutation overall: delta_score = {self.delta_score:.6f}"
                    )
                else:
                    top_score = all_scores[0]
                    second_score = all_scores[1]
                    self.delta_score = top_score - second_score
                    logger.debug(
                        f"PSM {self.scan_num} first round delta_score (including decoys): {top_score:.6f} - {second_score:.6f} = {self.delta_score:.6f}"
                    )
            else:
                if len(real_scores) == 1:
                    # Only one real permutation: treat as unambiguous
                    self.delta_score = real_scores[0]
                    self.psm_score = real_scores[0]
                    logger.debug(
                        f"PSM {self.scan_num} single real permutation: delta_score = {self.delta_score:.6f}"
                    )
                else:
                    top_score = real_scores[0]
                    second_score = real_scores[1]
                    self.delta_score = top_score - second_score
                    logger.debug(
                        f"PSM {self.scan_num} second round delta_score (real permutations only): {top_score:.6f} - {second_score:.6f} = {self.delta_score:.6f}"
                    )

        except Exception as e:
            logger.error(
                f"Error calculating delta_score for PSM {self.scan_num}: {str(e)}"
            )
            self.delta_score = 0.0
