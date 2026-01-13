"""
Peptide module.

This module contains the Peptide class, which represents a peptide sequence.
"""

import logging
import itertools
import re
from typing import Dict, List, Set, Tuple, Optional, Any, Union

import numpy as np
import pyopenms

from .constants import (
    NTERM_MOD,
    CTERM_MOD,
    AA_MASSES,
    DECOY_AA_MAP,
    AA_DECOY_MAP,
    WATER_MASS,
    PROTON_MASS,
)
from .globals import get_decoy_symbol
from .peak import Peak

logger = logging.getLogger(__name__)

# Regex pattern to match any modification in format (ModName)
# Captures the modification name inside parentheses
_MOD_PATTERN = re.compile(r"\(([^)]+)\)")

# Target modification names that are handled specially for phospho site localization
_TARGET_MOD_NAMES = {"Phospho", "PhosphoDecoy"}


def extract_target_amino_acids(target_modifications: List[str]) -> Set[str]:
    """
    Extract amino acid letters from target modifications format.

    Args:
        target_modifications: List of modification strings like ["Phospho (S)", "Phospho (T)", "Phospho (Y)", "PhosphoDecoy (A)"]

    Returns:
        Set of amino acid letters that can be modified
    """
    amino_acids = set()
    for mod in target_modifications:
        # Extract amino acid from format like "Phospho (S)" or "PhosphoDecoy (A)"
        match = re.search(r"\(([A-Z])\)", mod)
        if match:
            amino_acids.add(match.group(1))
    return amino_acids


class Peptide:
    """
    Class representing a peptide sequence.

    This class contains information about a peptide sequence, including
    its amino acid sequence, modifications, and fragment ions.
    """

    def __init__(
        self,
        peptide: str,
        config: Dict,
        mod_pep: Optional[str] = None,
        charge: Optional[int] = None,
        skip_expensive_init: bool = False,
    ):
        """
        Initialize a new Peptide instance.

        Args:
            peptide: Original peptide sequence
            config: Configuration dictionary
            mod_pep: Modified peptide sequence (optional)
            charge: Charge state (optional)
            skip_expensive_init: If True, skip permutation generation and ion ladder
                building during initialization. Use this when creating temporary
                Peptide objects for peak matching where mod_pos_map will be set
                externally and ion ladders built afterward.
        """
        self.peptide = peptide  # Original sequence
        self.mod_peptide = mod_pep if mod_pep else peptide  # Modified sequence
        self.charge = (
            charge if charge else config.get("max_charge_state", 2)
        )  # Charge state
        self.config = config  # Configuration

        # Modification related
        self.mod_pos_map = {}  # Position -> Modification mass (target mods)
        self.non_target_mods = {}  # Position -> Modification name (non-target mods)

        # Fragment ion information
        self.b_ions = {}  # Ion string -> m/z
        self.y_ions = {}  # Ion string -> m/z

        # Peptide properties
        self.pep_len = len(peptide)
        self.num_rps = 0  # Number of reported phospho sites
        self.num_pps = 0  # Number of potential phospho sites

        self.is_unambiguous = True

        self.num_permutations = 1
        self.num_decoy_permutations = 0
        self.score = 0.0

        self.matched_peaks = []  # List of matched peaks

        # Add HCD model related attributes
        self.hcd_score = 0.0
        self.hcd_matched_peaks = []

        # Add permutations attribute
        self.permutations = []

        # Initialize the peptide
        self._initialize(skip_expensive_init=skip_expensive_init)

    def _initialize(self, skip_expensive_init: bool = False) -> None:
        """Initialize the peptide.

        Args:
            skip_expensive_init: If True, skip permutation generation and ion ladder
                building. Used for temporary Peptide objects in peak matching.
        """
        from .mass_provider import get_modification_mass

        # Initialize modification mapping
        self.mod_pos_map = {}  # Target modifications (phosphorylation) - position -> mass
        self.non_target_mods = {}  # Non-target modifications - position -> mod_name

        # Check if mod_peptide was provided externally (different from raw peptide)
        mod_peptide_provided = self.mod_peptide and self.mod_peptide != self.peptide

        # Parse the original peptide to extract modifications
        # This populates mod_pos_map and non_target_mods
        # Also builds temp_mod_peptide which we use if mod_peptide wasn't provided
        temp_mod_peptide = ""
        i = 0
        while i < len(self.peptide):
            if self.peptide[i] == "(":
                # Found start of modification - extract the full modification name
                match = _MOD_PATTERN.match(self.peptide, i)
                if match:
                    mod_name = match.group(1)
                    mod_end = match.end()

                    # Get the position of the modified residue (previous character)
                    if len(temp_mod_peptide) > 0:
                        pos = len(temp_mod_peptide) - 1

                        if mod_name in _TARGET_MOD_NAMES:
                            # Target modification (Phospho, PhosphoDecoy)
                            self.mod_pos_map[pos] = get_modification_mass(mod_name)
                        else:
                            # Non-target modification - store the modification name
                            self.non_target_mods[pos] = mod_name

                        # Convert the residue to lowercase to mark it as modified
                        temp_mod_peptide = (
                            temp_mod_peptide[:-1] + temp_mod_peptide[-1].lower()
                        )

                    i = mod_end
                    continue

            # Regular amino acid character
            temp_mod_peptide += self.peptide[i]
            i += 1

        # Use provided mod_peptide if available, otherwise use the one we built
        if not mod_peptide_provided:
            self.mod_peptide = temp_mod_peptide

        # Calculate potential phosphorylation sites and reported phosphorylation sites
        # Extract target amino acids from target_modifications
        target_modifications = self.config.get("target_modifications", [])
        target_amino_acids = extract_target_amino_acids(target_modifications)

        # Calculate potential phosphorylation sites from original sequence (including all target amino acid sites)
        self.num_pps = 0
        i = 0
        while i < len(self.peptide):
            if self.peptide[i : i + 9] == "(Phospho)":
                # Skip modification markers
                i += 9
            elif self.peptide[i : i + 14] == "(PhosphoDecoy)":
                # Skip modification markers
                i += 14
            elif self.peptide[i] in target_amino_acids:
                # This is a potential phosphorylation site
                self.num_pps += 1
                i += 1
            else:
                i += 1

        # Calculate reported phosphorylation sites (number of (Phospho) + (PhosphoDecoy))
        self.num_rps = self.peptide.count("(Phospho)") + self.peptide.count(
            "(PhosphoDecoy)"
        )

        # Check if unambiguous (number of potential sites equals number of reported sites)
        self.is_unambiguous = self.num_pps == self.num_rps

        # Skip expensive operations for temporary Peptide objects used in peak matching
        if skip_expensive_init:
            return

        # Generate all possible permutations
        perms = self.get_permutations()
        if isinstance(perms, dict):
            self.permutations = list(perms.keys())
        else:
            self.permutations = perms if isinstance(perms, list) else []

        # Build ion ladders
        self.build_ion_ladders()

    def get_precursor_mass(self) -> float:
        """
        Calculate peptide precursor mass

        Returns:
            float: Precursor mass
        """
        # Initialize mass
        mass = 0.0

        # Add amino acid masses
        for aa in self.peptide:
            if aa in AA_MASSES:
                mass += AA_MASSES[aa]

        # Add modification masses
        for pos, mod_mass in self.mod_pos_map.items():
            if pos not in (
                NTERM_MOD,
                CTERM_MOD,
            ):  # Exclude N-term and C-term modifications
                mass += mod_mass

        # Add N-terminal modification
        if NTERM_MOD in self.mod_pos_map:
            mass += self.mod_pos_map[NTERM_MOD]

        # Add C-terminal modification
        if CTERM_MOD in self.mod_pos_map:
            mass += self.mod_pos_map[CTERM_MOD]

        # Add H2O and protons
        mass += WATER_MASS + (PROTON_MASS * self.charge)

        return mass

    def get_precursor_mz(self) -> float:
        """
        Calculate peptide precursor m/z value

        Returns:
            float: Precursor m/z value
        """
        return self.get_precursor_mass() / self.charge

    def _has_decoy_symbols(self) -> bool:
        """
        Check if the peptide contains decoy symbols (special characters).

        These are symbols from DECOY_AA_MAP representing decoy modifications.
        Now handled by PyOpenMS using registered PhosphoDecoy modifications.
        """
        return any(aa in DECOY_AA_MAP for aa in self.mod_peptide)

    def _to_pyopenms_format(self) -> str:
        """
        Convert internal sequence representation to PyOpenMS AASequence format.

        Internal format uses:
        - Uppercase letters: unmodified amino acids
        - Lowercase letters: modified amino acids
        - Special characters from DECOY_AA_MAP: PhosphoDecoy on mapped amino acids

        Modification disambiguation:
        - Uses self.non_target_mods to retrieve the actual modification name
        - Positions in non_target_mods have their stored modification name emitted
        - Other lowercase positions are target modifications (Phospho on S/T/Y,
          PhosphoDecoy on A or decoy residues)

        PyOpenMS format uses:
        - Uppercase letters with bracketed modifications: S(Phospho), M(Oxidation), etc.
        - PhosphoDecoy modifications: A(PhosphoDecoy), K(PhosphoDecoy (K)), etc.

        Returns:
            PyOpenMS-compatible sequence string
        """
        from .mass_provider import get_phospho_decoy_mod_name

        result = []
        for pos, aa in enumerate(self.mod_peptide):
            # Check if this position has a non-target modification
            if pos in self.non_target_mods:
                # Emit uppercase letter + stored modification name
                mod_name = self.non_target_mods[pos]
                result.append(f"{aa.upper()}({mod_name})")
            elif aa == "s":
                result.append("S(Phospho)")
            elif aa == "t":
                result.append("T(Phospho)")
            elif aa == "y":
                result.append("Y(Phospho)")
            elif aa == "a":
                # PhosphoDecoy on Alanine
                mod_name = get_phospho_decoy_mod_name("A")
                result.append(f"A({mod_name})")
            elif aa in DECOY_AA_MAP:
                # Special character represents decoy modification on mapped amino acid
                real_aa = DECOY_AA_MAP[aa]
                mod_name = get_phospho_decoy_mod_name(real_aa)
                result.append(f"{real_aa}({mod_name})")
            else:
                result.append(aa)

        return "".join(result)

    def get_precursor_mass_pyopenms(self) -> Optional[float]:
        """
        Calculate peptide precursor mass using PyOpenMS AASequence.

        This method provides validation against the custom mass calculation.

        Returns:
            Precursor mass [M+zH]^z+ or None on failure
        """
        seq_str = self._to_pyopenms_format()
        if not seq_str:
            return None

        try:
            seq = pyopenms.AASequence.fromString(seq_str)
            # getMonoWeight returns [M+H]+ mass by default for Full type
            mono_weight = seq.getMonoWeight(pyopenms.Residue.ResidueType.Full, 1)
            # Adjust for charge: add (charge-1) protons
            return mono_weight + (self.charge - 1) * PROTON_MASS
        except Exception as e:
            logger.warning(f"PyOpenMS precursor mass calculation failed: {e}")
            return None

    def build_ion_ladders(self) -> bool:
        """
        Build b-ion and y-ion ladders using PyOpenMS TheoreticalSpectrumGenerator.

        This method uses PyOpenMS for theoretical spectrum generation, providing
        consistency with other algorithms (AScore, PhosphoRS) that use PyOpenMS.
        Handles both regular and decoy sequences using registered PhosphoDecoy
        modifications.

        Returns:
            True if successful

        Raises:
            ValueError: If sequence cannot be converted to PyOpenMS format
            RuntimeError: If PyOpenMS spectrum generation fails
        """
        if not self.mod_peptide:
            raise ValueError("Cannot build ion ladders: mod_peptide is empty")

        self.b_ions = {}
        self.y_ions = {}

        seq_str = self._to_pyopenms_format()
        if not seq_str:
            raise ValueError(
                f"Cannot convert peptide to PyOpenMS format: {self.mod_peptide}"
            )

        try:
            seq = pyopenms.AASequence.fromString(seq_str)
        except Exception as e:
            raise ValueError(
                f"PyOpenMS cannot parse sequence '{seq_str}': {e}"
            ) from e

        # Configure spectrum generator
        spec_gen = pyopenms.TheoreticalSpectrumGenerator()
        params = spec_gen.getParameters()
        params.setValue("add_b_ions", "true")
        params.setValue("add_y_ions", "true")
        params.setValue("add_first_prefix_ion", "true")
        params.setValue("add_losses", "true")
        params.setValue("add_metainfo", "true")
        params.setValue("add_precursor_peaks", "false")
        params.setValue("add_abundant_immonium_ions", "false")
        params.setValue("isotope_model", "none")
        spec_gen.setParameters(params)

        min_mz = self.config.get("min_mz", 0.0)

        # Generate spectrum for each charge state
        for z in range(1, self.charge):
            theo_spectrum = pyopenms.MSSpectrum()
            spec_gen.getSpectrum(theo_spectrum, seq, z, z)

            # Extract ions from spectrum
            for i in range(theo_spectrum.size()):
                peak = theo_spectrum[i]
                mz = peak.getMZ()

                if mz <= min_mz:
                    continue

                # Get ion annotation from metainfo
                ion_name = ""
                if theo_spectrum.getStringDataArrays():
                    for sda in theo_spectrum.getStringDataArrays():
                        if sda.getName() == "IonNames" and i < len(sda):
                            raw_name = sda[i]
                            ion_name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
                            break

                if not ion_name:
                    continue

                # Parse ion name and categorize (PyOpenMS format: b2-H2O1++, y3+, etc.)
                if ion_name.startswith("b"):
                    self.b_ions[ion_name] = mz
                elif ion_name.startswith("y"):
                    self.y_ions[ion_name] = mz

        # Build theoretical masses list
        self.theoretical_masses = list(self.b_ions.values()) + list(
            self.y_ions.values()
        )
        self.theoretical_masses.sort()
        return True

    def calc_theoretical_masses(self, perm: str) -> List[float]:
        """
        Calculate theoretical masses for given permutation

        Args:
            perm: Peptide permutation

        Returns:
            List of theoretical masses
        """
        if not perm:
            return []

        # Parse modification sites
        mod_map = {}
        i = 0
        while i < len(perm):
            if perm[i : i + 9] == "(Phospho)":
                mod_map[i - 1] = 79.966331
                i += 9
            elif perm[i : i + 9] == "(Oxidation)":
                mod_map[i - 1] = 15.994915
                i += 9
            else:
                i += 1

        masses = []
        current_mass = 0.0

        for i in range(len(perm)):
            if perm[i] in AA_MASSES:
                current_mass += AA_MASSES[perm[i]]
            if i in mod_map:
                current_mass += mod_map[i]
            masses.append(current_mass + 1.007825)

        current_mass = 0.0
        for i in range(len(perm) - 1, -1, -1):
            if perm[i] in AA_MASSES:
                current_mass += AA_MASSES[perm[i]]
            if i in mod_map:
                current_mass += mod_map[i]
            masses.append(current_mass + 19.01839)

        if self.config.get("neutral_losses"):
            nl_masses = []
            for nl in self.config["neutral_losses"]:
                if nl.startswith("sty"):
                    nl_mass = float(nl.split()[-1])
                    nl_masses.append(nl_mass)

            masses_with_nl = []
            for mass in masses:
                masses_with_nl.append(mass)
                for nl_mass in nl_masses:
                    masses_with_nl.append(mass - nl_mass)

            masses = masses_with_nl

        masses = sorted(list(set(masses)))

        return masses

    def get_ion_ladder(self, ion_type: Optional[str] = None) -> Dict[str, float]:
        """Get ion ladder for specified ion type or all ions

        Args:
            ion_type: Optional ion type ("b" or "y"). If None, returns all ions.

        Returns:
            Dict[str, float]: Ion ladder dictionary mapping ion string to m/z
        """
        if ion_type == "b":
            return self.b_ions
        elif ion_type == "y":
            return self.y_ions
        else:
            combined = {}
            combined.update(self.b_ions)
            combined.update(self.y_ions)
            return combined

    def get_permutations(self, decoy: bool = False) -> Dict[str, float]:
        """
        Get all possible permutations of the peptide

        Args:
            decoy: Whether to generate decoy sequences

        Returns:
            Dict[str, float]: Dictionary of sequences and their scores
        """
        ret = {}

        if not decoy:
            if self.is_unambiguous:
                ret[self.mod_peptide] = 0.0
            else:
                sites = []
                for i, aa in enumerate(self.peptide):
                    if aa in ["S", "T", "Y"]:
                        sites.append(i)

                for site in sites:
                    seq = list(self.mod_peptide)
                    if site < len(seq):
                        seq[site] = seq[site].lower()
                    ret["".join(seq)] = 0.0
        else:
            cand_mod_sites = []

            # Extract target amino acids from target_modifications
            target_modifications = self.config.get("target_modifications", [])
            target_amino_acids = extract_target_amino_acids(target_modifications)

            for i in range(self.pep_len):
                aa = self.peptide[i]
                score = 0

                if aa not in target_amino_acids:
                    score += 1

                if i not in self.non_target_mods:
                    score += 1

                if score == 2:
                    cand_mod_sites.append(i)

            for combo in itertools.combinations(cand_mod_sites, self.num_rps):
                mod_pep = ""

                if NTERM_MOD in self.mod_pos_map:
                    mod_pep = "["

                for i in range(self.pep_len):
                    aa = self.peptide[i].lower()

                    if i in combo:  # Sites that need modification
                        decoy_char = get_decoy_symbol(self.peptide[i])
                        mod_pep += decoy_char
                    elif i in self.non_target_mods:  # Sites with existing modifications
                        mod_pep += aa
                    else:  # Regular sites
                        mod_pep += aa.upper()

                if CTERM_MOD in self.mod_pos_map:
                    mod_pep += "]"

                ret[mod_pep] = 0.0

        return ret

    def match_peaks(
        self, spectrum, config: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Match peaks in the spectrum to theoretical fragment ions.

        Uses vectorized NumPy operations with searchsorted for O(log n)
        binary search instead of O(n) linear search per ion.

        Args:
            spectrum: Spectrum object
            config: Optional configuration dictionary (if None, uses self.config)

        Returns:
            List of matched peaks
        """
        use_config = config if config is not None else self.config

        # Get spectrum data once (avoid repeated calls)
        mz_values, intensities = spectrum.get_peaks()
        if len(mz_values) == 0:
            return []

        # Pre-compute spectrum arrays for vectorized access
        norm_intensity = spectrum.norm_intensity
        max_i = spectrum.max_i

        # Ensure m/z values are sorted and get sort indices
        # Most spectra are already sorted, but we need indices for lookups
        if spectrum._mz_sorted is None:
            spectrum._update_sorted_indices()
        mz_sorted = spectrum._mz_sorted
        sort_indices = spectrum._mz_sorted_indices

        tolerance = use_config.get("fragment_mass_tolerance", 0.5)
        is_ppm = use_config.get("ms2_tolerance_units", "Da") == "ppm"

        # Combine y and b ions for batch processing
        y_ions = self.get_ion_ladder("y")
        b_ions = self.get_ion_ladder("b")

        # Build arrays of theoretical m/z values and metadata
        all_ion_strs = []
        all_theo_mz = []
        all_ion_types = []

        for ion_str, theo_mz in y_ions.items():
            all_ion_strs.append(ion_str)
            all_theo_mz.append(theo_mz)
            all_ion_types.append("y")

        for ion_str, theo_mz in b_ions.items():
            all_ion_strs.append(ion_str)
            all_theo_mz.append(theo_mz)
            all_ion_types.append("b")

        if not all_theo_mz:
            return []

        # Convert to numpy array for vectorized operations
        theo_mz_array = np.array(all_theo_mz)

        # Vectorized tolerance calculation
        if is_ppm:
            ppm_err = tolerance / 1000000.0
            match_errs = theo_mz_array * ppm_err * 0.5
        else:
            match_errs = np.full(len(theo_mz_array), tolerance * 0.5)

        # Calculate search bounds for all ions at once
        lower_bounds = theo_mz_array - match_errs
        upper_bounds = theo_mz_array + match_errs

        # Use searchsorted for O(log n) binary search to find candidate ranges
        left_indices = np.searchsorted(mz_sorted, lower_bounds, side='left')
        right_indices = np.searchsorted(mz_sorted, upper_bounds, side='right')

        # Track best match for each spectrum peak (by original index)
        # Store: original_idx -> (ion_idx, mass_diff, intensity)
        best_match_by_peak = {}

        # Process each ion's candidate matches
        for ion_idx in range(len(all_theo_mz)):
            left = left_indices[ion_idx]
            right = right_indices[ion_idx]

            if left >= right:
                continue  # No candidates in range

            # Get candidate indices in sorted array
            sorted_candidate_indices = np.arange(left, right)

            # Map back to original indices
            original_indices = sort_indices[sorted_candidate_indices]

            # Get intensities for candidates
            candidate_intensities = intensities[original_indices]

            # Find the most intense peak among candidates
            best_local_idx = np.argmax(candidate_intensities)
            best_orig_idx = original_indices[best_local_idx]
            best_intensity = candidate_intensities[best_local_idx]

            theo_mz = all_theo_mz[ion_idx]
            mass_diff = mz_values[best_orig_idx] - theo_mz

            # Check if this peak was already matched to another ion
            if best_orig_idx in best_match_by_peak:
                _, old_mass_diff, _ = best_match_by_peak[best_orig_idx]
                # Keep the match with smaller mass error
                if abs(mass_diff) < abs(old_mass_diff):
                    best_match_by_peak[best_orig_idx] = (ion_idx, mass_diff, best_intensity)
            else:
                best_match_by_peak[best_orig_idx] = (ion_idx, mass_diff, best_intensity)

        # Build final matched peaks list (only create dicts for final matches)
        matched_peaks = []
        for orig_idx, (ion_idx, mass_diff, _) in best_match_by_peak.items():
            peak = {
                "mz": mz_values[orig_idx],
                "intensity": intensities[orig_idx],
                "rel_intensity": (intensities[orig_idx] / max_i) * 100.0,
                "norm_intensity": norm_intensity[orig_idx],
                "mass_diff": mass_diff,
                "matched": True,
                "matched_ion_str": all_ion_strs[ion_idx],
                "matched_ion_mz": all_theo_mz[ion_idx],
                "ion_type": all_ion_types[ion_idx],
            }
            matched_peaks.append(peak)

        return matched_peaks

    def _find_closest_peak(self, theo_mz: float, spectrum) -> Optional[Peak]:
        """
        Find the closest peak in the spectrum to the theoretical m/z.

        Args:
            theo_mz: Theoretical m/z value
            spectrum: PyOpenMS MSSpectrum object

        Returns:
            Closest peak or None if no peak is within the tolerance
        """
        # Get the spectrum peaks
        mzs, intensities = spectrum.get_peaks()

        # Calculate the fragment error tolerance
        match_err = self.config.get("ms2_tolerance", 0.5)  # Default in Daltons

        if self.config.get("ms2_tolerance_units", "Da") == "ppm":
            ppm_err = match_err / 1000000.0
            match_err = theo_mz * ppm_err

        match_err *= 0.5  # Split in half

        a = theo_mz - match_err
        b = theo_mz + match_err

        # Find all peaks within the tolerance window
        cand_matches = []
        for i in range(len(mzs)):
            if a <= mzs[i] <= b:
                peak = Peak(mzs[i], intensities[i])
                peak.matched = True
                peak.dist = peak.mz - theo_mz  # obs - expected
                cand_matches.append(peak)

        # If at least one match was found
        if cand_matches:
            # Sort by intensity (highest first)
            cand_matches.sort(key=lambda pk: pk.raw_intensity, reverse=True)

            # Return the most intense peak
            return cand_matches[0]

        return None

    def calc_score_cid(self, model) -> float:
        """
        Calculate peptide score using CID model

        Args:
            model: CID scoring model

        Returns:
            float: Scoring result
        """
        if not self.matched_peaks:
            return 0.0

        charge_model = model.get_charge_model(self.charge)
        if not charge_model:
            return 0.0

        total_score = 0.0

        for peak in self.matched_peaks:
            ion_type = None
            if peak.matched_ion_str.startswith("b"):
                ion_type = "b"
            elif peak.matched_ion_str.startswith("y"):
                ion_type = "y"

            intensity_score = model.calc_intensity_score(
                peak.norm_intensity, ion_type, charge_model
            )

            dist_score = model.calc_distance_score(peak.dist, ion_type, charge_model)

            intense_wt = 1.0 / (1.0 + np.exp(-intensity_score))

            if np.isnan(dist_score) or np.isinf(dist_score):
                x = 0.0
            else:
                x = intense_wt * dist_score

            if x < 0:
                x = 0.0  # Prevent negative scores

            peak.score = x
            peak.intensity_score = intensity_score
            peak.dist_score = dist_score

            total_score += x

        return total_score

    def calc_score_hcd(self, model) -> float:
        """
        Calculate peptide score using HCD model

        Args:
            model: HCD scoring model

        Returns:
            float: Scoring result
        """
        if not hasattr(self, "matched_peaks") or not self.matched_peaks:
            return 0.0

        total_score = 0.0

        for peak in self.matched_peaks:
            intensity_u = model.get_log_np_density_int("n", peak["norm_intensity"])
            dist_u = 0.0  # Log of uniform distribution is 0

            ion_type = peak["matched_ion_str"][0]  # Take first character ('b' or 'y')

            intensity_m = model.get_log_np_density_int(ion_type, peak["norm_intensity"])
            dist_m = model.get_log_np_density_dist_pos(peak["mass_diff"])

            intensity_score = intensity_m - intensity_u
            dist_score = dist_m - dist_u

            if np.isnan(intensity_score) or np.isinf(intensity_score):
                intensity_score = 0.0
            if np.isnan(dist_score) or np.isinf(dist_score):
                dist_score = 0.0

            peak_score = intensity_score + dist_score
            if peak_score < 0:
                peak_score = 0.0

            peak["score"] = peak_score
            peak["intensity_score"] = intensity_score
            peak["dist_score"] = dist_score

            total_score += peak_score

        return total_score

    def _log_gaussian_prob(self, mu: float, var: float, x: float) -> float:
        """
        Calculate the log probability of x under a Gaussian distribution.

        Args:
            mu: Mean
            var: Variance
            x: Value

        Returns:
            Log probability
        """
        if var <= 0:
            return float("-inf")

        log_prob = -0.5 * np.log(2 * np.pi * var) - 0.5 * ((x - mu) ** 2) / var

        return log_prob

    def is_decoy_pep(self) -> bool:
        """
        Check if this peptide is a decoy.

        Returns:
            True if the peptide is a decoy, False otherwise
        """
        # Check if any residue is a decoy residue
        for i in range(len(self.mod_peptide)):
            aa = self.mod_peptide[i]
            if aa in DECOY_AA_MAP:
                return True

        return False

    def get_unmodified_sequence(self) -> str:
        """
        Get unmodified peptide sequence.

        Returns:
            str: Unmodified peptide sequence
        """
        try:
            # Remove all modification markers
            unmod_seq = ""
            i = 0
            while i < len(self.peptide):
                if self.peptide[i : i + 9] == "(Phospho)":
                    i += 9
                elif self.peptide[i : i + 9] == "(Oxidation)":
                    i += 9
                elif self.peptide[i : i + 13] == "(PhosphoDecoy)":
                    i += 13
                else:
                    unmod_seq += self.peptide[i]
                    i += 1
            return unmod_seq
        except Exception as e:
            logger.error(f"Error getting unmodified sequence: {str(e)}")
            return self.peptide
