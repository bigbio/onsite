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
    NTERM_MOD, CTERM_MOD, AA_MASSES, DECOY_AA_MAP, AA_DECOY_MAP,
    WATER_MASS, PROTON_MASS, ION_TYPES, NEUTRAL_LOSSES
)
from .globals import get_decoy_symbol
from .peak import Peak

logger = logging.getLogger(__name__)


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
        match = re.search(r'\(([A-Z])\)', mod)
        if match:
            amino_acids.add(match.group(1))
    return amino_acids


class Peptide:
    """
    Class representing a peptide sequence.
    
    This class contains information about a peptide sequence, including
    its amino acid sequence, modifications, and fragment ions.
    """
    
    def __init__(self, peptide: str, config: Dict, mod_pep: Optional[str] = None, charge: Optional[int] = None):
        """
        Initialize a new Peptide instance.
        
        Args:
            peptide: Original peptide sequence
            config: Configuration dictionary
            mod_pep: Modified peptide sequence (optional)
            charge: Charge state (optional)
        """
        self.peptide = peptide  # Original sequence
        self.mod_peptide = mod_pep if mod_pep else peptide  # Modified sequence
        self.charge = charge if charge else config.get('max_charge_state', 2)  # Charge state
        self.config = config  # Configuration
        
        # Modification related
        self.mod_pos_map = {}  # Position -> Modification mass
        self.non_target_mods = {}  # Position -> AA
        
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
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the peptide."""
        # Convert modified sequence to Java style (lowercase letters represent modifications)
        if self.mod_peptide and self.mod_peptide != self.peptide:
            # If modified peptide is provided, use it directly
            pass
        else:
            # Parse modifications from original peptide
            mod_pep = ""
            i = 0
            while i < len(self.peptide):
                # Check phosphorylation modification (9 characters)
                if i + 9 <= len(self.peptide) and self.peptide[i:i+9] == "(Phospho)":
                    # Convert modified amino acid to lowercase
                    if len(mod_pep) > 0:
                        mod_pep = mod_pep[:-1] + mod_pep[-1].lower()
                    i += 9
                    continue
                # Check decoy phosphorylation modification (14 characters)
                elif i + 14 <= len(self.peptide) and self.peptide[i:i+14] == "(PhosphoDecoy)":
                    # Convert modified amino acid to lowercase (treat as phosphorylation)
                    if len(mod_pep) > 0:
                        mod_pep = mod_pep[:-1] + mod_pep[-1].lower()
                    i += 14
                    continue
                # Check oxidation modification (11 characters)
                elif i + 11 <= len(self.peptide) and self.peptide[i:i+11] == "(Oxidation)":
                    # Convert modified amino acid to lowercase
                    if len(mod_pep) > 0:
                        mod_pep = mod_pep[:-1] + mod_pep[-1].lower()
                    i += 11
                    continue
                
                # If not a modification marker, add the character
                mod_pep += self.peptide[i]
                i += 1
            
            self.mod_peptide = mod_pep

        # Initialize modification mapping
        self.mod_pos_map = {}  # Target modifications (phosphorylation)
        self.non_target_mods = {}  # Non-target modifications (oxidation, etc.)
        
        # Parse modification positions - rebuild mod_pep to calculate correct positions
        mod_pep_pos = ""
        i = 0
        while i < len(self.peptide):
            if i + 9 <= len(self.peptide) and self.peptide[i:i+9] == "(Phospho)":
                # Phosphorylation modification - target modification
                if len(mod_pep_pos) > 0:
                    pos = len(mod_pep_pos) - 1
                    self.mod_pos_map[pos] = 79.966  # Phosphorylation mass
                i += 9
            elif i + 14 <= len(self.peptide) and self.peptide[i:i+14] == "(PhosphoDecoy)":
                # Decoy phosphorylation modification - also treat as target modification
                if len(mod_pep_pos) > 0:
                    pos = len(mod_pep_pos) - 1
                    self.mod_pos_map[pos] = 79.966  # Phosphorylation mass
                i += 14
            elif i + 11 <= len(self.peptide) and self.peptide[i:i+11] == "(Oxidation)":
                # Oxidation modification - non-target modification
                if len(mod_pep_pos) > 0:
                    pos = len(mod_pep_pos) - 1
                    self.non_target_mods[pos] = mod_pep_pos[-1].lower()
                i += 11
            else:
                mod_pep_pos += self.peptide[i]
                i += 1

        # Calculate potential phosphorylation sites and reported phosphorylation sites
        # Extract target amino acids from target_modifications
        target_modifications = self.config.get('target_modifications', [])
        target_amino_acids = extract_target_amino_acids(target_modifications)
        
        # Calculate potential phosphorylation sites from original sequence (including all target amino acid sites)
        self.num_pps = 0
        i = 0
        while i < len(self.peptide):
            if self.peptide[i:i+9] == "(Phospho)":
                # Skip modification markers
                i += 9
            elif self.peptide[i:i+14] == "(PhosphoDecoy)":
                # Skip modification markers
                i += 14
            elif self.peptide[i] in target_amino_acids:
                # This is a potential phosphorylation site
                self.num_pps += 1
                i += 1
            else:
                i += 1
        
        # Calculate reported phosphorylation sites (number of (Phospho) + (PhosphoDecoy))
        self.num_rps = self.peptide.count("(Phospho)") + self.peptide.count("(PhosphoDecoy)")
        
        # Check if unambiguous (number of potential sites equals number of reported sites)
        self.is_unambiguous = self.num_pps == self.num_rps
        
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
            if pos not in (NTERM_MOD, CTERM_MOD):  # Exclude N-term and C-term modifications
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

    def build_ion_ladders(self) -> None:
        """
        Build b-ion and y-ion ladders, naming and generation method fully aligned with Java Peptide.java.
        """
        if not self.mod_peptide:
            return

        self.b_ions = {}
        self.y_ions = {}

        peptide = self.mod_peptide
        pep_len = len(peptide)
        charge = self.charge
        NTERM_MOD = self.config.get('NTERM_MOD', -100)
        CTERM_MOD = self.config.get('CTERM_MOD', 100)
        nterm_mass = self.mod_pos_map.get(NTERM_MOD, 0.0) if NTERM_MOD in self.mod_pos_map else 0.0
        cterm_mass = self.mod_pos_map.get(CTERM_MOD, 0.0) if CTERM_MOD in self.mod_pos_map else 0.0
        min_len = 2
        min_mz = self.config.get('min_mz', 0.0)

        # Only generate ions with charge less than peptide charge
        for z in range(1, charge):  # Only generate charge states 1 to charge-1
            for i in range(1, pep_len):  # Fragment length at least 2
                prefix = peptide[:i]
                suffix = peptide[i:]
                prefix_len = str(len(prefix))
                suffix_len = str(len(suffix))

                # b ions
                if len(prefix) >= min_len:
                    b = f"b{prefix_len}:{prefix}"
                    
                    bm = self._fragment_ion_mass(b, z, 0.0, ion_type='b') + nterm_mass
                    bmz = bm / z
                    
                    if z > 1:
                        b += f"/+{z}"
                    
                    if bmz > min_mz:
                        self.b_ions[b] = bmz
                        
                        # Handle neutral losses - extend neutral losses for all charge states
                        self._record_nl_ions(b, z, bm, ion_type='b')

                # y ions
                if len(suffix) >= min_len and suffix.lower() != peptide.lower():
                    y = f"y{suffix_len}:{suffix}"
                    ym = self._fragment_ion_mass(y, z, 18.01056, ion_type='y') + cterm_mass
                    ymz = ym / z
                    
                    if z > 1:
                        y += f"/+{z}"
                    
                    if ymz > min_mz:
                        self.y_ions[y] = ymz
                        
                        # Handle neutral losses - extend neutral losses for all charge states
                        self._record_nl_ions(y, z, ym, ion_type='y')

        # Theoretical mass list
        self.theoretical_masses = list(self.b_ions.values()) + list(self.y_ions.values())
        self.theoretical_masses.sort()

    def _fragment_ion_mass(self, ion_str, z, addl_mass, ion_type='b'):
        """
        Calculate fragment ion mass
        Receives complete ion string, such as "b4:FLLE" or "y2:sK"
        """
        mass = 0.0
        # Parse amino acid sequence starting after colon
        start = ion_str.find(":") + 1
        stop = len(ion_str)
        seq = ion_str[start:stop]
        
        for idx, aa in enumerate(seq):
            # Directly use mass mapping table, lowercase letters represent modified amino acids
            if aa in AA_MASSES:
                mass += AA_MASSES[aa]
        
        # Both b/y ions add z proton masses, y ions also add 1 water molecule mass
        mass += PROTON_MASS * z
        if ion_type == 'y':
            mass += WATER_MASS
        
        return mass

    def _record_nl_ions(self, ion, z, orig_ion_mass, ion_type='b'):
        """
        Handle neutral loss ions
        """
        # Extract sequence part from ion string
        start = ion.find(":") + 1
        end = len(ion)
        if "/" in ion:
            end = ion.find("/")
        
        seq = ion[start:end]
        
        # Check if it's a decoy sequence
        if self._is_decoy_seq(seq):
            # Decoy sequence neutral loss handling
            decoy_nl_map = self.config.get('decoy_neutral_losses', [])
            if isinstance(decoy_nl_map, list):
                decoy_nl_map = self._parse_neutral_losses_list(decoy_nl_map)
            
            for nl_key, nl_mass in decoy_nl_map.items():
                # Extract neutral loss tag from key, format: X-H3PO4 -> -H3PO4
                nl_tag = "-" + nl_key.split("-", 1)[1]  # Split once, take second part
                nl_str = ion + nl_tag
                mass = orig_ion_mass + nl_mass
                mz = round(mass / z, 4)
                
                if mz > self.config.get('min_mz', 0.0):
                    if ion_type == 'b':
                        self.b_ions[nl_str] = mz
                    else:
                        self.y_ions[nl_str] = mz
        else:
            # Normal sequence neutral loss handling
            nl_map = self.config.get('neutral_losses', [])
            if isinstance(nl_map, list):
                nl_map = self._parse_neutral_losses_list(nl_map)
            
            for nl_key, nl_mass in nl_map.items():
                # Parse format: sty-H3PO4 -> check if sequence contains any amino acid in sty
                residues = nl_key.split('-')[0]
                # Extract neutral loss tag from key, format: sty-H3PO4 -> -H3PO4
                nl_tag = "-" + nl_key.split("-", 1)[1]  # Split once, take second part
                
                # Check if sequence contains amino acids that can produce neutral losses
                num_cand_res = sum(1 for aa in seq if aa.upper() in residues.upper())
                if num_cand_res > 0:
                    nl_str = ion + nl_tag
                    mass = orig_ion_mass + nl_mass
                    mz = round(mass / z, 4)
                    
                    if mz > self.config.get('min_mz', 0.0):
                        if ion_type == 'b':
                            self.b_ions[nl_str] = mz
                        else:
                            self.y_ions[nl_str] = mz
    
    def _get_neutral_losses(self, ion_str, ion_type='b', only_main_ion=False):
        """
        Return neutral loss list
        Only extend main ion names, avoid extending ion names with '-'.
        """
        nl_list = []
        # Only extend main ion names (without '-')
        # Main ion name definition: after ':' to '/+' or end, without '-'
        colon_idx = ion_str.find(":")
        if colon_idx == -1:
            return nl_list
        # Find /+ position
        slash_idx = ion_str.find("/+", colon_idx)
        dash_idx = ion_str.find("-", colon_idx)
        # If there's '-' after ':' to '/+' or before '-', it's not a main ion name
        if only_main_ion:
            # Only allow no '-' between after ':' to '/+' or end
            end_idx = len(ion_str)
            if slash_idx != -1 and (dash_idx == -1 or slash_idx < dash_idx):
                end_idx = slash_idx
            elif dash_idx != -1:
                end_idx = dash_idx
            seq_part = ion_str[colon_idx+1:end_idx]
            if '-' in seq_part:
                return nl_list
        
        # Extract sequence part from ion string
        start = ion_str.find(":") + 1
        end = len(ion_str)
        if "-" in ion_str:
            end = ion_str.find("-")
        
        seq = ion_str[start:end]
        
        # Check if it's a decoy sequence
        if self._is_decoy_seq(seq):
            # Decoy sequence neutral loss handling
            decoy_nl_map = self.config.get('decoy_neutral_losses', {})
            # Handle list format decoy_neutral_losses
            if isinstance(decoy_nl_map, list):
                decoy_nl_map = self._parse_neutral_losses_list(decoy_nl_map)
            
            for nl_key, nl_mass in decoy_nl_map.items():
                residues = nl_key.split('-')[0]
                nl_tag = '-' + nl_key.split('-')[1]
                
                num_cand_res = sum(1 for aa in seq if aa.upper() in residues.upper())
                if num_cand_res > 0:
                    nl_list.append((nl_tag, nl_mass))
        else:
            nl_map = self.config.get('neutral_losses', {})
            if isinstance(nl_map, list):
                nl_map = self._parse_neutral_losses_list(nl_map)
            
            for nl_key, nl_mass in nl_map.items():
                residues = nl_key.split('-')[0]
                nl_tag = '-' + nl_key.split('-')[1]
                
                num_cand_res = sum(1 for aa in seq if aa.upper() in residues.upper())
                if num_cand_res > 0:
                    nl_list.append((nl_tag, nl_mass))
        
        return nl_list
    
    def _parse_neutral_losses_list(self, nl_list):
        """
        Parse list format neutral loss configuration
        Format: ["sty -H3PO4 -97.97690"] -> {"sty-H3PO4": -97.97690}
        """
        nl_dict = {}
        for item in nl_list:
            parts = item.strip().split()
            if len(parts) >= 3:
                residues = parts[0]
                nl_name = parts[1]
                nl_mass = float(parts[2])
                # Remove leading - from nl_name to avoid double hyphens
                if nl_name.startswith('-'):
                    nl_key = f"{residues}{nl_name}"  # sty-H3PO4
                else:
                    nl_key = f"{residues}-{nl_name}"
                nl_dict[nl_key] = nl_mass
        return nl_dict
    
    def _is_decoy_seq(self, seq):
        """
        Check if sequence is a decoy sequence
        """
        return any(aa.islower() for aa in seq)
    
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
            if perm[i:i+9] == "(Phospho)":
                mod_map[i-1] = 79.966331
                i += 9
            elif perm[i:i+9] == "(Oxidation)":
                mod_map[i-1] = 15.994915
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
        for i in range(len(perm)-1, -1, -1):
            if perm[i] in AA_MASSES:
                current_mass += AA_MASSES[perm[i]]
            if i in mod_map:
                current_mass += mod_map[i]
            masses.append(current_mass + 19.01839)
        
        if self.config.get('neutral_losses'):
            nl_masses = []
            for nl in self.config['neutral_losses']:
                if nl.startswith('sty'):
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
            ion_type: Optional ion type ('b' or 'y'). If None, returns all ions.
            
        Returns:
            Dict[str, float]: Ion ladder dictionary mapping ion string to m/z
        """
        if ion_type == 'b':
            return self.b_ions
        elif ion_type == 'y':
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
                    if aa in ['S', 'T', 'Y']:
                        sites.append(i)
                        
                for site in sites:
                    seq = list(self.mod_peptide)
                    if site < len(seq):
                        seq[site] = seq[site].lower()
                    ret[''.join(seq)] = 0.0
        else:
            cand_mod_sites = []
            
            # Extract target amino acids from target_modifications
            target_modifications = self.config.get('target_modifications', [])
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
    
    def match_peaks(self, spectrum, config: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Match peaks in the spectrum to theoretical fragment ions.
        
        Args:
            spectrum: Spectrum object
            config: Optional configuration dictionary (if None, uses self.config)
            
        Returns:
            List of matched peaks
        """
        use_config = config if config is not None else self.config
        
       
        matched_peaks = []
        best_match_map = {}  # mz -> peak
        
        tolerance = use_config.get('fragment_mass_tolerance', 0.5)
        
        y_ions = self.get_ion_ladder('y')
        for ion_str, theo_mz in y_ions.items():
            if use_config.get('ms2_tolerance_units', 'Da') == 'ppm':
                ppm_err = tolerance / 1000000.0
                match_err = theo_mz * ppm_err
            else:
                match_err = tolerance
            
            match_err *= 0.5  # Split tolerance window
            a = theo_mz - match_err
            b = theo_mz + match_err
            
            cand_matches = []
            mz_values, intensities = spectrum.get_peaks()
            for i in range(len(mz_values)):
                if a <= mz_values[i] <= b:
                    peak = {
                        'mz': mz_values[i],
                        'intensity': intensities[i],
                        'rel_intensity': (intensities[i] / spectrum.max_i) * 100.0,
                        'norm_intensity': spectrum.norm_intensity[i],
                        'mass_diff': mz_values[i] - theo_mz,  # obs - expected
                        'matched': True,
                        'matched_ion_str': ion_str,
                        'matched_ion_mz': theo_mz,
                        'ion_type': 'y'
                    }
                    cand_matches.append(peak)
            
            if cand_matches:
                cand_matches.sort(key=lambda x: x['intensity'], reverse=True)
                best_peak = cand_matches[0]
                
                if best_peak['mz'] in best_match_map:
                    old_peak = best_match_map[best_peak['mz']]
                    if abs(old_peak['mass_diff']) > abs(best_peak['mass_diff']):
                        best_match_map[best_peak['mz']] = best_peak
                else:
                    best_match_map[best_peak['mz']] = best_peak
        
        b_ions = self.get_ion_ladder('b')
        for ion_str, theo_mz in b_ions.items():
            if use_config.get('ms2_tolerance_units', 'Da') == 'ppm':
                ppm_err = tolerance / 1000000.0
                match_err = theo_mz * ppm_err
            else:
                match_err = tolerance
            
            match_err *= 0.5  # Split tolerance window
            a = theo_mz - match_err
            b = theo_mz + match_err
            
            cand_matches = []
            mz_values, intensities = spectrum.get_peaks()
            for i in range(len(mz_values)):
                if a <= mz_values[i] <= b:
                    peak = {
                        'mz': mz_values[i],
                        'intensity': intensities[i],
                        'rel_intensity': (intensities[i] / spectrum.max_i) * 100.0,
                        'norm_intensity': spectrum.norm_intensity[i],
                        'mass_diff': mz_values[i] - theo_mz,  # obs - expected
                        'matched': True,
                        'matched_ion_str': ion_str,
                        'matched_ion_mz': theo_mz,
                        'ion_type': 'b'
                    }
                    cand_matches.append(peak)
            
            if cand_matches:
                cand_matches.sort(key=lambda x: x['intensity'], reverse=True)
                best_peak = cand_matches[0]
                
                if best_peak['mz'] in best_match_map:
                    old_peak = best_match_map[best_peak['mz']]
                    if abs(old_peak['mass_diff']) > abs(best_peak['mass_diff']):
                        best_match_map[best_peak['mz']] = best_peak
                else:
                    best_match_map[best_peak['mz']] = best_peak
        
        matched_peaks = list(best_match_map.values())
        
        if use_config.get('debug', False):
            output_data = {
                'matched_peaks': matched_peaks,
                'theoretical_ions': {
                    'y_ions': y_ions,
                    'b_ions': b_ions
                },
                'peptide': self.peptide,
                'modified_peptide': self.mod_peptide,
                'charge': self.charge
            }
            
            
            
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
        match_err = self.config.get('ms2_tolerance', 0.5)  # Default in Daltons
        
        if self.config.get('ms2_tolerance_units', 'Da') == 'ppm':
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
            if peak.matched_ion_str.startswith('b'):
                ion_type = 'b'
            elif peak.matched_ion_str.startswith('y'):
                ion_type = 'y'
                
            intensity_score = model.calc_intensity_score(
                peak.norm_intensity,
                ion_type,
                charge_model
            )
            
            dist_score = model.calc_distance_score(
                peak.dist,
                ion_type,
                charge_model
            )
            
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
        if not hasattr(self, 'matched_peaks') or not self.matched_peaks:
            return 0.0
        
        total_score = 0.0
        
        for peak in self.matched_peaks:
            intensity_u = model.get_log_np_density_int('n', peak['norm_intensity'])
            dist_u = 0.0  # Log of uniform distribution is 0
            
            ion_type = peak['matched_ion_str'][0]  # Take first character ('b' or 'y')
            
            intensity_m = model.get_log_np_density_int(ion_type, peak['norm_intensity'])
            dist_m = model.get_log_np_density_dist_pos(peak['mass_diff'])
            
            intensity_score = intensity_m - intensity_u
            dist_score = dist_m - dist_u
            
            if np.isnan(intensity_score) or np.isinf(intensity_score):
                intensity_score = 0.0
            if np.isnan(dist_score) or np.isinf(dist_score):
                dist_score = 0.0
            
            peak_score = intensity_score + dist_score
            if peak_score < 0:
                peak_score = 0.0
            
            peak['score'] = peak_score
            peak['intensity_score'] = intensity_score
            peak['dist_score'] = dist_score
            
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
            return float('-inf')
        
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
                if self.peptide[i:i+9] == "(Phospho)":
                    i += 9
                elif self.peptide[i:i+9] == "(Oxidation)":
                    i += 9
                elif self.peptide[i:i+13] == "(PhosphoDecoy)":
                    i += 13
                else:
                    unmod_seq += self.peptide[i]
                    i += 1
            return unmod_seq
        except Exception as e:
            logger.error(f"Error getting unmodified sequence: {str(e)}")
            return self.peptide