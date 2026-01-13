"""
PyOpenMS-based mass provider for pyLuciPHOr2.

Extracts amino acid and modification masses from PyOpenMS at module load time,
providing fast O(1) lookup during processing.
"""

import numpy as np
from pyopenms import ResidueDB, ModificationsDB, ResidueModification, Residue, Constants, EmpiricalFormula


# Module-level caches populated at import time
_AA_MASSES: dict = {}
_MASS_ARRAY: np.ndarray = None  # Indexed by ord(char) for fast lookup
_INITIALIZED: bool = False
_PHOSPHO_DECOY_REGISTERED: bool = False

# Residues that already have PhosphoDecoy defined in PyOpenMS
_BUILTIN_PHOSPHO_DECOY_RESIDUES = {'A', 'G', 'L'}

# All standard amino acids
_STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"


def _register_phospho_decoy_modifications():
    """
    Register PhosphoDecoy modification for all amino acid residues.

    PyOpenMS only has PhosphoDecoy defined for A, G, L by default.
    This function registers it for all other residues so that decoy
    sequences can be handled uniformly by TheoreticalSpectrumGenerator.
    """
    global _PHOSPHO_DECOY_REGISTERED

    if _PHOSPHO_DECOY_REGISTERED:
        return

    mod_db = ModificationsDB()
    phospho_decoy_mass = 79.966331  # Same as Phospho

    # Register PhosphoDecoy for residues that don't have it
    for aa in _STANDARD_AAS:
        if aa in _BUILTIN_PHOSPHO_DECOY_RESIDUES:
            continue  # Already defined in PyOpenMS

        try:
            mod = ResidueModification()
            mod.setId(f'PhosphoDecoy ({aa})')
            mod.setFullId(f'PhosphoDecoy ({aa})')
            mod.setName('PhosphoDecoy')
            mod.setDiffMonoMass(phospho_decoy_mass)
            mod.setOrigin(aa.encode())
            mod_db.addModification(mod)
        except Exception:
            pass  # Ignore if already registered or other errors

    _PHOSPHO_DECOY_REGISTERED = True


def get_phospho_decoy_mod_name(residue: str) -> str:
    """
    Get the correct PhosphoDecoy modification name for a residue.

    Args:
        residue: Single-letter amino acid code

    Returns:
        Modification name to use with AASequence.setModification()
    """
    if residue in _BUILTIN_PHOSPHO_DECOY_RESIDUES:
        return "PhosphoDecoy"
    return f"PhosphoDecoy ({residue})"


def _initialize():
    """Initialize mass caches from PyOpenMS databases."""
    global _MASS_ARRAY, _INITIALIZED  # _AA_MASSES is modified in-place, not reassigned

    if _INITIALIZED:
        return

    # Register PhosphoDecoy for all residues first
    _register_phospho_decoy_modifications()

    # Get residue database
    residue_db = ResidueDB()

    # Build mass dictionary from PyOpenMS
    for aa in _STANDARD_AAS:
        residue = residue_db.getResidue(aa)
        _AA_MASSES[aa] = residue.getMonoWeight(Residue.ResidueType.Internal)

    # Build NumPy array for fast indexed lookup (covers ASCII range)
    _MASS_ARRAY = np.zeros(256, dtype=np.float64)
    for aa, mass in _AA_MASSES.items():
        _MASS_ARRAY[ord(aa)] = mass

    _INITIALIZED = True


def get_residue_mass(aa: str) -> float:
    """
    Get monoisotopic mass of a single amino acid residue.

    Args:
        aa: Single-letter amino acid code (uppercase)

    Returns:
        Monoisotopic mass (internal residue, no termini)
    """
    if not _INITIALIZED:
        _initialize()
    return _AA_MASSES.get(aa, 0.0)


def get_residue_mass_fast(aa_ord: int) -> float:
    """
    Get residue mass by ASCII ordinal for fast lookup.

    Args:
        aa_ord: ord() value of amino acid character

    Returns:
        Monoisotopic mass
    """
    if not _INITIALIZED:
        _initialize()
    return _MASS_ARRAY[aa_ord]


def get_mass_array() -> np.ndarray:
    """
    Get the full mass lookup array for vectorized operations.

    Returns:
        NumPy array where index = ord(char), value = mass
    """
    if not _INITIALIZED:
        _initialize()
    return _MASS_ARRAY


def get_all_aa_masses() -> dict:
    """
    Get dictionary of all standard amino acid masses.

    Returns:
        Dict mapping single-letter codes to monoisotopic masses
    """
    if not _INITIALIZED:
        _initialize()
    return _AA_MASSES.copy()


# Physical constants from PyOpenMS
def get_proton_mass() -> float:
    """Get proton mass from PyOpenMS Constants."""
    return Constants.PROTON_MASS_U


def get_water_mass() -> float:
    """Get water mass (H2O) from PyOpenMS."""
    return EmpiricalFormula("H2O").getMonoWeight()


# Cache for modification masses to avoid repeated PyOpenMS lookups
# (which can produce warnings about multiple matches)
_MOD_MASS_CACHE: dict = {}


def get_modification_mass(mod_name: str) -> float:
    """
    Get modification delta mass from PyOpenMS ModificationsDB.

    Results are cached to avoid repeated lookups and suppress duplicate
    warnings from PyOpenMS about multiple modifications with the same name.

    Args:
        mod_name: Modification name (e.g., "Phospho", "Oxidation")

    Returns:
        Delta mass of the modification

    Raises:
        ValueError: If modification is not found in PyOpenMS ModificationsDB
    """
    # Check cache first
    if mod_name in _MOD_MASS_CACHE:
        return _MOD_MASS_CACHE[mod_name]

    mod_db = ModificationsDB()

    # Try different naming conventions that PyOpenMS uses
    # Common residue-specific suffixes
    residue_suffixes = ["", " (S)", " (T)", " (Y)", " (M)", " (C)", " (K)", " (R)",
                        " (N)", " (Q)", " (W)", " (H)", " (D)", " (E)", " (F)"]

    for suffix in residue_suffixes:
        try:
            mod = mod_db.getModification(mod_name + suffix)
            mass = mod.getDiffMonoMass()
            _MOD_MASS_CACHE[mod_name] = mass  # Cache the result
            return mass
        except Exception:
            continue

    # Also try without any suffix for terminal modifications
    # and common alternative names
    alternative_names = [
        mod_name.replace(" ", ""),  # Try without spaces
        mod_name.lower(),  # Try lowercase
        mod_name.upper(),  # Try uppercase
    ]

    for name in alternative_names:
        try:
            mod = mod_db.getModification(name)
            mass = mod.getDiffMonoMass()
            _MOD_MASS_CACHE[mod_name] = mass  # Cache the result
            return mass
        except Exception:
            continue

    raise ValueError(
        f"Modification '{mod_name}' not found in PyOpenMS ModificationsDB. "
        f"Please check the modification name is correct."
    )


# Pre-compute common modification masses
PHOSPHO_MASS = None
OXIDATION_MASS = None


def get_phospho_mass() -> float:
    """Get phosphorylation modification mass."""
    global PHOSPHO_MASS
    if PHOSPHO_MASS is None:
        PHOSPHO_MASS = get_modification_mass("Phospho")
    return PHOSPHO_MASS


def get_oxidation_mass() -> float:
    """Get oxidation modification mass."""
    global OXIDATION_MASS
    if OXIDATION_MASS is None:
        OXIDATION_MASS = get_modification_mass("Oxidation")
    return OXIDATION_MASS


# Initialize on module load for fastest access
_initialize()
