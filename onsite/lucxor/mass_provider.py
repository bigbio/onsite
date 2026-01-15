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


def get_modification_mass(mod_name: str, residue: str = None) -> float:
    """
    Get modification delta mass from PyOpenMS ModificationsDB.

    When the residue is provided, uses precise three-argument lookup which
    avoids PyOpenMS warnings about multiple matches. Results are cached.

    Args:
        mod_name: Modification name (e.g., "Phospho", "Oxidation")
        residue: Single-letter amino acid code for precise lookup (recommended)

    Returns:
        Delta mass of the modification

    Raises:
        ValueError: If modification is not found in PyOpenMS ModificationsDB
    """
    # Check cache first
    cache_key = f"{mod_name}:{residue}" if residue else mod_name
    if cache_key in _MOD_MASS_CACHE:
        return _MOD_MASS_CACHE[cache_key]

    mod_db = ModificationsDB()

    # Use precise lookup when residue is known (silent, no warnings)
    if residue:
        try:
            mod = mod_db.getModification(
                mod_name, residue, ResidueModification.TermSpecificity.ANYWHERE
            )
            mass = mod.getDiffMonoMass()
            _MOD_MASS_CACHE[cache_key] = mass
            return mass
        except Exception:
            pass  # Fall through to generic lookup

    # Fall back to generic name lookup (may produce warning for ambiguous names)
    try:
        mod = mod_db.getModification(mod_name)
        mass = mod.getDiffMonoMass()
        _MOD_MASS_CACHE[cache_key] = mass
        return mass
    except Exception:
        pass

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
        PHOSPHO_MASS = get_modification_mass("Phospho", "S")  # Mass is same for S/T/Y
    return PHOSPHO_MASS


def get_oxidation_mass() -> float:
    """Get oxidation modification mass."""
    global OXIDATION_MASS
    if OXIDATION_MASS is None:
        OXIDATION_MASS = get_modification_mass("Oxidation", "M")  # Most common site
    return OXIDATION_MASS


# Initialize on module load for fastest access
_initialize()
