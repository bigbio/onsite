"""
Constants and default configurations for pyLuciPHOr2
"""

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

# Physical constants
WATER_MASS = 18.010564684
PROTON_MASS = 1.00727646688
PPM = 1.0 / 1000000.0
TINY_NUM = 1e-10
MIN_DELTA_SCORE = 0.1
FUNCTION_TIME_LIMIT = 120  # seconds

# Amino acid masses (monoisotopic)
AA_MASSES = {
    'A': 71.03711, 'C': 103.00919, 'D': 115.02694,
    'E': 129.04259, 'F': 147.06841, 'G': 57.02146,
    'H': 137.05891, 'I': 113.08406, 'K': 128.09496,
    'L': 113.08406, 'M': 131.04049, 'N': 114.04293,
    'P': 97.05276, 'Q': 128.05858, 'R': 156.10111,
    'S': 87.03203, 'T': 101.04768, 'V': 99.06841,
    'W': 186.07931, 'Y': 163.06333
}

# Add lowercase letter mass definitions for modification sites (including modification mass)
AA_MASSES.update({
    's': 87.03203 + 79.966331,  # Ser + phosphorylation
    't': 101.04768 + 79.966331,  # Thr + phosphorylation
    'y': 163.06333 + 79.966331,  # Tyr + phosphorylation
    'a': 71.03711 + 79.966331,  # Ala + PhosphoDecoy
    'm': 131.04049 + 15.994915,  # Met + oxidation
})

# Decoy amino acid mapping
DECOY_AA_MAP = {
    '2': 'A',
    '3': 'R',
    '4': 'N',
    '5': 'D',
    '6': 'C',
    '7': 'E',
    '8': 'Q',
    '9': 'G',
    '0': 'H',
    '@': 'I',
    '#': 'L',
    '$': 'K',
    '%': 'M',
    '&': 'F',
    ';': 'P',
    '?': 'W',
    '~': 'V',
    '^': 'S',
    '*': 'T',
    '=': 'Y'
}

# Reverse mapping for decoy amino acids
AA_DECOY_MAP = {v: k for k, v in DECOY_AA_MAP.items()}

# Add mass definitions for all decoy symbols
# decoy amino acid mass = original amino acid mass + decoyMass (79.966331)
DECOY_MASS = 79.966331
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
    "target_modifications": [
        "Phospho (S)",
        "Phospho (T)",
        "Phospho (Y)"
    ],
    "neutral_losses": [
        "sty -H3PO4 -97.97690"  # Amino acid list, neutral loss name, mass
    ],
    "decoy_neutral_losses": [
        "X -H3PO4 -97.97690"  # Neutral loss for decoy sequences
    ],
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
    "rt_tolerance": 0.01
}

# Ion types
ION_TYPES = {
    'b': 1.007825,  # H
    'y': 19.01839,   # H2O + H
    'a': -26.98772,  # CO
    'c': 17.02655,   # NH3
    'x': 25.97913,   # CO2
    'z': 1.99184     # NH2
}

# Neutral losses
NEUTRAL_LOSSES = {
    'H3PO4': 97.976896,  # Phosphoric acid
    'H2O': 18.010565,    # Water
    'NH3': 17.026549,    # Ammonia
    'CO': 27.994915,     # Carbon monoxide
    'CO2': 43.989829,    # Carbon dioxide
    'sty': -97.97690,    # H3PO4
    'S': 98.00039,  # Ser phosphorylation neutral loss
    'T': 98.00039,  # Thr phosphorylation neutral loss
    'Y': 98.00039   # Tyr phosphorylation neutral loss
}

# Score types
SCORE_TYPES = {
    'Posterior Error Probability': 0,
    'Mascot Ion Score': 1,
    '-log(E-value)': 2,
    'X!Tandem Hyperscore': 3,
    'Sequest Xcorr': 4
}

# Modification masses
MOD_MASSES = {
    'Phospho': 79.966331,
    'Oxidation': 15.994915,
}

# Decoy amino acid mapping
DECOY_AMINO_ACIDS = {
    'S': 'A',  # Ser -> Ala
    'T': 'V',  # Thr -> Val
    'Y': 'F'   # Tyr -> Phe
}

# New constants
PHOSPHO_MOD_MASS = 79.966331
OXIDATION_MASS = 15.994915

# Character types
SINGLE_CHAR = 0

# PSM types
DECOY = 0
REAL = 1

# Minimum values
MIN_NUM_NEG_PKS = 50000

# Physical constants
WATER = 18.01056
PROTON = 1.00728