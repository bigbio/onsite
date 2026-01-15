"""
Constants and default configurations for pyLuciPHOr2
"""

from . import mass_provider

# Algorithm types
ALGORITHM_CID = 0
ALGORITHM_HCD = 1

# Units for MS2 tolerance
DALTONS = 0
PPM_UNITS = 1

# Input file types
PEPXML = 0
TSV = 1

# Terminal modification positions
NTERM_MOD = -100
CTERM_MOD = 100

# Run modes
DEFAULT_RUN_MODE = 0
REPORT_DECOYS = 1

# Debug modes
NO_DEBUG = 0
WRITE_MODEL_PKS = 1
WRITE_PERM_SCORES = 2
WRITE_SCORED_PKS = 3
WRITE_HCD_NONPARAM = 4
WRITE_ALL_MATCHED_PK_SCORES = 5

# Scoring methods
PEPPROPHET = 0
MASCOTIONSCORE = 1
NEGLOGEXPECT = 2
XTDHYPERSCORE = 3
XCORR = 4

# Physical constants - derived from PyOpenMS
WATER_MASS = mass_provider.get_water_mass()
PROTON_MASS = mass_provider.get_proton_mass()
PPM = 1.0 / 1000000.0
TINY_NUM = 1e-10
MIN_DELTA_SCORE = 0.1
FUNCTION_TIME_LIMIT = 120  # seconds

# Modification masses - derived from PyOpenMS
PHOSPHO_MOD_MASS = mass_provider.get_phospho_mass()
OXIDATION_MASS = mass_provider.get_oxidation_mass()

# Amino acid masses (monoisotopic) - derived from PyOpenMS ResidueDB
AA_MASSES = mass_provider.get_all_aa_masses()

# Add lowercase letter mass definitions for modification sites (including modification mass)
AA_MASSES.update(
    {
        "s": AA_MASSES["S"] + PHOSPHO_MOD_MASS,  # Ser + phosphorylation
        "t": AA_MASSES["T"] + PHOSPHO_MOD_MASS,  # Thr + phosphorylation
        "y": AA_MASSES["Y"] + PHOSPHO_MOD_MASS,  # Tyr + phosphorylation
        "a": AA_MASSES["A"] + PHOSPHO_MOD_MASS,  # Ala + PhosphoDecoy
        "m": AA_MASSES["M"] + OXIDATION_MASS,  # Met + oxidation
    }
)

# Decoy amino acid mapping
DECOY_AA_MAP = {
    "2": "A",
    "3": "R",
    "4": "N",
    "5": "D",
    "6": "C",
    "7": "E",
    "8": "Q",
    "9": "G",
    "0": "H",
    "@": "I",
    "#": "L",
    "$": "K",
    "%": "M",
    "&": "F",
    ";": "P",
    "?": "W",
    "~": "V",
    "^": "S",
    "*": "T",
    "=": "Y",
}

# Reverse mapping for decoy amino acids
AA_DECOY_MAP = {v: k for k, v in DECOY_AA_MAP.items()}

# Add mass definitions for all decoy symbols
# decoy amino acid mass = original amino acid mass + decoyMass (Phospho mass)
DECOY_MASS = PHOSPHO_MOD_MASS
for decoy_aa, orig_aa in DECOY_AA_MAP.items():
    if decoy_aa not in AA_MASSES and orig_aa in AA_MASSES:
        AA_MASSES[decoy_aa] = AA_MASSES[orig_aa] + DECOY_MASS

# Default configuration
DEFAULT_CONFIG = {
    # Algorithm settings
    "fragment_method": "CID",
    "fragment_mass_tolerance": 0.5,
    "fragment_error_units": "Da",
    "min_mz": 150.0,
    # Modification settings
    "target_modifications": ["Phospho (S)", "Phospho (T)", "Phospho (Y)"],
    "neutral_losses": [
        "sty -H3PO4 -97.97690"  # Amino acid list, neutral loss name, mass
    ],
    "decoy_neutral_losses": ["X -H3PO4 -97.97690"],  # Neutral loss for decoy sequences
    "decoy_mass": 79.966331,
    # Peptide settings
    "max_charge_state": 5,
    "max_peptide_length": 40,
    "max_num_perm": 16384,
    # Scoring settings
    "modeling_score_threshold": 0.95,
    "scoring_threshold": 0.0,
    "min_num_psms_model": 50,
    # Performance settings
    "num_threads": 6,
    "rt_tolerance": 0.01,
}

# Ion types
ION_TYPES = {
    "b": 1.007825,  # H
    "y": 19.01839,  # H2O + H
    "a": -26.98772,  # CO
    "c": 17.02655,  # NH3
    "x": 25.97913,  # CO2
    "z": 1.99184,  # NH2
}

# Neutral losses
NEUTRAL_LOSSES = {
    "H3PO4": 97.976896,  # Phosphoric acid
    "H2O": 18.010565,  # Water
    "NH3": 17.026549,  # Ammonia
    "CO": 27.994915,  # Carbon monoxide
    "CO2": 43.989829,  # Carbon dioxide
    "sty": -97.97690,  # H3PO4
    "S": 98.00039,  # Ser phosphorylation neutral loss
    "T": 98.00039,  # Thr phosphorylation neutral loss
    "Y": 98.00039,  # Tyr phosphorylation neutral loss
}

# Score types
SCORE_TYPES = {
    "Posterior Error Probability": 0,
    "Mascot Ion Score": 1,
    "-log(E-value)": 2,
    "X!Tandem Hyperscore": 3,
    "Sequest Xcorr": 4,
}

# Modification masses dict - derived from PyOpenMS
MOD_MASSES = {
    "Phospho": PHOSPHO_MOD_MASS,
    "Oxidation": OXIDATION_MASS,
}

# Decoy amino acid mapping
DECOY_AMINO_ACIDS = {
    "S": "A",  # Ser -> Ala
    "T": "V",  # Thr -> Val
    "Y": "F",  # Tyr -> Phe
}

# Character types
SINGLE_CHAR = 0

# PSM types
DECOY = 0
REAL = 1

# Minimum values
MIN_NUM_NEG_PKS = 50000

# Physical constants (aliases for backward compatibility)
WATER = WATER_MASS
PROTON = PROTON_MASS
