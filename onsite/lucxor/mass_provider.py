"""
PyOpenMS-based mass provider for pyLuciPHOr2.

Extracts amino acid and modification masses from PyOpenMS at module load time,
providing fast O(1) lookup during processing.
"""

import numpy as np
from pyopenms import ResidueDB, ModificationsDB, Residue, Constants


# Module-level caches populated at import time
_AA_MASSES: dict = {}
_MASS_ARRAY: np.ndarray = None  # Indexed by ord(char) for fast lookup
_INITIALIZED: bool = False


def _initialize():
    """Initialize mass caches from PyOpenMS databases."""
    global _AA_MASSES, _MASS_ARRAY, _INITIALIZED

    if _INITIALIZED:
        return

    # Get residue database
    residue_db = ResidueDB()

    # Standard amino acid one-letter codes
    standard_aas = "ACDEFGHIKLMNPQRSTVWY"

    # Build mass dictionary from PyOpenMS
    for aa in standard_aas:
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
    """Get water mass (H2O)."""
    # PyOpenMS stores this in Constants or can be computed
    # H2O = 2*H + O = 2*1.007825 + 15.994915 = 18.010565
    return 18.010564684


def get_modification_mass(mod_name: str) -> float:
    """
    Get modification delta mass from PyOpenMS ModificationsDB.

    Args:
        mod_name: Modification name (e.g., "Phospho", "Oxidation")

    Returns:
        Delta mass of the modification
    """
    mod_db = ModificationsDB()

    # Try different naming conventions
    search_names = [
        mod_name,
        f"{mod_name} (S)",
        f"{mod_name} (T)",
        f"{mod_name} (Y)",
        f"{mod_name} (M)",
    ]

    for name in search_names:
        try:
            mod = mod_db.getModification(name)
            return mod.getDiffMonoMass()
        except Exception:
            continue

    # Fallback to known values if PyOpenMS lookup fails
    fallbacks = {
        "Phospho": 79.966331,
        "Oxidation": 15.994915,
    }
    return fallbacks.get(mod_name, 0.0)


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
