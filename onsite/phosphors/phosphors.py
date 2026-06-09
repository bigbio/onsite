import itertools
import math
import re
from decimal import Decimal, ROUND_FLOOR
import numpy as np
from pyopenms import (
    AASequence,
    ResidueModification,
    ModificationsDB,
    TheoreticalSpectrumGenerator,
    MSSpectrum,
    MSExperiment,
    PeptideIdentification,
    PeptideHit,
    Precursor,
    Constants,
    Peak1D,
)
import sys  # For checking float limits

# --- Configuration ---
MODIFICATION_NAME = "Phospho"  # Base modification name
MODIFICATION_TYPES = {
    "S": "Phospho (S)",
    "T": "Phospho (T)",
    "Y": "Phospho (Y)",
    "A": "PhosphoDecoy (A)",  # Use PhosphoDecoy for A when add_decoys=True
}
POTENTIAL_SITES = set(["S", "T", "Y"])
ADD_DECOYS = False  # Configuration parameter to include A as phosphorylation site
FRAGMENT_TOLERANCE = 0.05  # Typical tolerance for PhosphoRS scoring (Da)
FRAGMENT_METHOD_PPM = False  # PhosphoRS typically uses Da tolerance
ADD_PRECURSOR_PEAK = False
ADD_ION_TYPES = ("b", "y", "a", "c", "x", "z")  # Add more ion types
MAX_ION_CHARGE = 2  # Adjust based on typical fragmentation
ADD_NEUTRAL_LOSSES = True  # Include neutral losses
WINDOW_SIZE = 100.0
MAX_DEPTH = 8

# --- Constants ---
LOG10_ZERO_REPLACEMENT = -100.0  # Value to use for log10(0) or very small numbers
MIN_PROBABILITY = 1e-10  # Minimum probability to avoid log10(0)
DEFAULT_OCCURRENCE_PROBABILITY = (
    0.1  # Default occurrence probability if calculation fails
)

# --- Distribution Cache ---
DISTRIBUTION_CACHE_SIZE = 1000
_distribution_cache = {}  # p -> {n -> BinomialDistribution}


# --- Utility helpers ---
def _floor_double(value: float, n_decimals: int) -> float:
    """Floor ``value`` to ``n_decimals`` decimal places, matching compomics
    ``Util.floorDouble = new BigDecimal(String.valueOf(x)).setScale(n, FLOOR)``:
    a DECIMAL-string floor, NOT a binary floor of ``value*10**n``.

    The previous binary form (``math.floor(value*10**n)/10**n``) dropped a digit
    whenever ``value*10**n`` landed a hair below an integer in IEEE arithmetic
    (e.g. ``0.29 -> 0.28``, ``0.0006 -> 0.0005``), which perturbed the random-
    match probability ``p`` fed to the binomial (D13c). ``repr(value)`` is the
    shortest round-trip decimal, mirroring Java's ``String.valueOf(double)``."""
    if n_decimals <= 0:
        return float(math.floor(value))
    if not math.isfinite(value):
        return value
    quantum = Decimal(1).scaleb(-n_decimals)  # 10**-n_decimals
    return float(Decimal(repr(value)).quantize(quantum, rounding=ROUND_FLOOR))


# --- Helper functions ---
def _copy_spectrum_subset(exp_spectrum: MSSpectrum, keep_indexes: list) -> MSSpectrum:
    """Create a new spectrum containing only the peaks at keep_indexes."""
    if not keep_indexes:
        return exp_spectrum
    peaks = exp_spectrum.get_peaks()
    mz = [peaks[0][i] for i in keep_indexes]
    it = [peaks[1][i] for i in keep_indexes]
    new_spec = MSSpectrum()
    new_spec.setMSLevel(exp_spectrum.getMSLevel())
    new_spec.setPrecursors(exp_spectrum.getPrecursors())
    new_spec.set_peaks((mz, it))
    return new_spec


def getp_style(n_peaks: int, w_mz: float, tol_da: float) -> float:
    """
    PhosphoRS getp equivalent:
    - If w == 0 or n <= 1: return 1.0
    - p = d * n / w, clamp to 1.0
    - floor to nDecimals, where nDecimals = int(-log10(d/w)) + 1
    """
    if w_mz == 0.0:
        return 1.0
    if n_peaks <= 1:
        return 1.0
    d_over_w = tol_da / w_mz if w_mz > 0 else 1.0
    if d_over_w <= 0:
        return 1.0
    d_over_w_log = -math.log10(d_over_w)
    n_decimals = int(d_over_w_log) + 1
    p = tol_da * float(n_peaks) / w_mz
    if p > 1.0:
        p = 1.0
    p = _floor_double(p, n_decimals)
    # Returns floored p within [0,1] without forcing a positive minimum
    p = min(1.0, max(0.0, p))
    return p


def _add_distribution_to_cache(p: float, n: int, prob: float):
    """Add a distribution result to cache and manage cache size."""

    if len(_distribution_cache) >= DISTRIBUTION_CACHE_SIZE:
        # Remove oldest entries (simple FIFO)
        keys_to_remove = list(_distribution_cache.keys())[
            : len(_distribution_cache) - DISTRIBUTION_CACHE_SIZE + 1
        ]
        for key in keys_to_remove:
            del _distribution_cache[key]

    if p not in _distribution_cache:
        _distribution_cache[p] = {}
    _distribution_cache[p][n] = prob


def binomial_tail_probability(k: int, n: int, p: float) -> float:
    """
    Descending cumulative probability P(X >= k) for X~Bin(n,p).
    Mirrors getPhosphoRsScoreP behavior exactly with caching.
    If k == 0: return 1.0.
    """
    if k <= 0:
        return 1.0
    if p <= 0.0:
        return 0.0 if k > 0 else 1.0
    if p >= 1.0:
        return 1.0

    if k > n:
        return 0.0

    # Check cache first
    if p in _distribution_cache and n in _distribution_cache[p]:
        cached_prob = _distribution_cache[p][n]
        # For cached results, we need to recalculate based on k
        # This is a simplified approach - in practice, we"d cache the full distribution
        pass  # Continue with calculation

    # For very small probabilities, use log-space calculation to maintain precision
    # This is crucial for distinguishing between very unlikely events

    # Calculate expected value and standard deviation
    expected = n * p
    std_dev = math.sqrt(n * p * (1 - p))

    # If k is much larger than expected, the probability is very small
    # Use log-space calculation to maintain numerical precision
    if k > expected + 2 * std_dev:
        # Use log-sum-exp trick for numerical stability
        log_terms = []

        for t in range(k, n + 1):
            if t > n:
                break

            # Calculate log(C(n,t)) using lgamma
            log_comb = math.lgamma(n + 1) - math.lgamma(t + 1) - math.lgamma(n - t + 1)

            # Calculate log(p^t * (1-p)^(n-t))
            log_p_term = t * math.log(p) if p > 0 else float("-inf")
            log_1_minus_p_term = (
                (n - t) * math.log(1.0 - p) if (1.0 - p) > 0 else float("-inf")
            )

            # Total log probability for this term
            log_term = log_comb + log_p_term + log_1_minus_p_term

            if log_term > -700:  # Avoid underflow
                log_terms.append(log_term)

            # Stop if terms become too small
            if log_term < -50:
                break

        if not log_terms:
            return 0.0

        # Use log-sum-exp trick
        max_log_term = max(log_terms)
        log_prob = max_log_term + math.log(
            sum(math.exp(log_term - max_log_term) for log_term in log_terms)
        )

        # Convert back to linear space
        prob = math.exp(log_prob)
        # Don"t truncate to 1e-15 - keep the actual small probability
        return prob

    # For normal cases, use scipy if available
    try:
        import scipy.stats

        prob = 1.0 - scipy.stats.binom.cdf(k - 1, n, p)
        # Clamp to [0,1] without enforcing a positive floor
        result = min(1.0, max(0.0, prob))
        # Cache the result for future use
        _add_distribution_to_cache(p, n, result)
        return result
    except ImportError:
        pass

    # Fallback: direct calculation
    prob = 0.0
    for t in range(k, n + 1):
        if t > n:
            break

        # Calculate C(n,t) * p^t * (1-p)^(n-t)
        try:
            from math import comb

            coeff = comb(n, t)
        except (ImportError, OverflowError):
            # Use log-space for large combinations
            log_comb = math.lgamma(n + 1) - math.lgamma(t + 1) - math.lgamma(n - t + 1)
            coeff = math.exp(log_comb)

        term = coeff * (p**t) * ((1 - p) ** (n - t))
        prob += term

        # No artificial early break; accumulate full tail within numeric limits
        # (Python float underflow will naturally limit extremely small terms)

    result = min(1.0, max(0.0, prob))
    # Cache the result for future use
    _add_distribution_to_cache(p, n, result)
    return result


def _count_matched_ions(theo_mz, exp_mz_sorted, fragment_tolerance, fragment_method_ppm):
    """Count theoretical fragment ions matched to experimental peaks.

    Two corrections vs. a naive scan, both of which otherwise inflate the
    binomial counts (and asymmetrically favor decoy isomers, which carry extra
    phospho neutral-loss ions):

    1. Theoretical ions falling within one tolerance window of each other are
       merged - an experimental peak cannot distinguish them, so they must count
       as ONE trial, not several (avoids counting the same theoretical ion
       multiple times).
    2. Each experimental peak is consumed by at most one theoretical ion, so a
       single peak cannot satisfy several theoretical ions.

    Args:
        theo_mz: iterable of theoretical m/z (any order).
        exp_mz_sorted: experimental peak m/z, ascending.
        fragment_tolerance: match tolerance (Da, or ppm if fragment_method_ppm).
        fragment_method_ppm: interpret tolerance as ppm.

    Returns:
        (n_expected, k_matches): unique theoretical ions, and how many matched.
    """
    def tol(mz):
        return (mz * fragment_tolerance / 1_000_000.0) if fragment_method_ppm else fragment_tolerance

    theo_unique = []
    for mz in sorted(theo_mz):
        if not theo_unique or (mz - theo_unique[-1]) > tol(mz):
            theo_unique.append(mz)

    k = 0
    ri = 0
    m = len(exp_mz_sorted)
    for mz in theo_unique:
        t = tol(mz)
        while ri < m and exp_mz_sorted[ri] < mz - t:
            ri += 1
        if ri < m and exp_mz_sorted[ri] <= mz + t:
            k += 1
            ri += 1  # consume this experimental peak

    return len(theo_unique), k


def _isomer_phospho_positions(seq_str):
    """0-based residue indices carrying Phospho or PhosphoDecoy in a modified
    peptide string (e.g. 'AS(Phospho)A(PhosphoDecoy)K' -> {1, 2})."""
    positions = set()
    i = 0
    pos = -1
    while i < len(seq_str):
        c = seq_str[i]
        if c == "(":
            j = seq_str.find(")", i)
            if j == -1:
                break
            if seq_str[i + 1 : j] in ("Phospho", "PhosphoDecoy") and pos >= 0:
                positions.add(pos)
            i = j + 1
        elif c.isalpha():
            pos += 1
            i += 1
        else:
            i += 1
    return positions


def site_deltas_from_isomers(isomer_list):
    """Per-site phosphoRS peptide-score delta - phosphoRS's own ranking signal.

    The phosphoRS peptide score is ``-10*log10(P_random)``; for each candidate
    site the delta is (best score among isoforms that phosphorylate the site)
    minus (best score among those that do not) - the rank1-vs-alternative score
    gap that phosphoRS itself maximizes when choosing peak depth (Taus et al.
    2011). Higher = more confident the site is phosphorylated. Unlike the
    normalized site probability it does NOT saturate toward 0/100, so it
    preserves the ranking resolution needed to threshold a global FLR.

    Args:
        isomer_list: list of (modified_sequence_str, P_random) over all isoforms.
    Returns:
        dict {residue_index: delta} (0-based); empty if no isoforms.
    """
    if not isomer_list:
        return {}

    _FLOOR = 1e-300  # keep -10*log10(P) finite if P_random underflowed to 0
    parsed = []
    sites = set()
    for seq_str, p_random in isomer_list:
        pep = -10.0 * math.log10(max(float(p_random), _FLOOR))
        occ = _isomer_phospho_positions(seq_str)
        parsed.append((occ, pep))
        sites |= occ

    deltas = {}
    for s in sites:
        best_with = max((pep for occ, pep in parsed if s in occ), default=None)
        best_without = max((pep for occ, pep in parsed if s not in occ), default=None)
        if best_with is None:
            continue
        # No competing isoform (single candidate) -> full score, like the per-PSM
        # convention; otherwise the rank1-vs-best-alternative gap.
        deltas[s] = best_with if best_without is None else (best_with - best_without)
    return deltas


# --- Pre-filtering (filterSpectrum) ---
def filter_spectrum_for_phosphors(
    exp_spectrum: MSSpectrum, fragment_tolerance: float, fragment_method_ppm: bool
) -> MSSpectrum:
    """
    Filter spectrum:
    - window = 10 * ms2Tolerance (Da) if ms2Tolerance <= 10 Da else 100 Da
    - maxPeaks = 10 if ms2Tolerance <= 10, else int(window / ms2Tolerance)
    Keep only the most intense peaks per window.
    """
    peaks = exp_spectrum.get_peaks()
    if not peaks or len(peaks[0]) == 0:
        return exp_spectrum
    mz = list(peaks[0])
    it = list(peaks[1])

    # Determine tolerance in Da at spectrum max m/z
    max_mz = mz[-1]
    if fragment_method_ppm:
        ms2_tol_da = max_mz * fragment_tolerance / 1_000_000.0
    else:
        ms2_tol_da = fragment_tolerance

    if ms2_tol_da <= 0:
        return exp_spectrum

    # Window and maxPeaks calculation
    if ms2_tol_da <= 10.0:
        window = 10.0 * ms2_tol_da
        max_peaks = 10
    else:
        window = WINDOW_SIZE
        max_peaks = int(window / ms2_tol_da) if ms2_tol_da > 0 else len(mz)

    if max_peaks < 1:
        raise ValueError("All peaks removed by filtering.")

    # Windowing with intensity-based selection
    to_remove = set()
    ref_index = 0
    ref_mz = mz[0]

    for i in range(len(mz)):
        cur_mz = mz[i]
        if cur_mz > ref_mz + window:
            # Process window [ref_index, i)
            if i - ref_index > max_peaks:
                # Create intensity map for this window
                intensity_map = {}
                for j in range(ref_index, i):
                    intensity_map[it[j]] = j

                # Sort by intensity descending and select top max_peaks
                sorted_intensities = sorted(intensity_map.keys(), reverse=True)
                count = 0
                for intensity in sorted_intensities:
                    count += 1
                    if count > max_peaks:
                        to_remove.add(intensity_map[intensity])

            ref_index = i
            ref_mz += window

    # Process tail window
    if len(mz) - ref_index > max_peaks:
        intensity_map = {}
        for j in range(ref_index, len(mz)):
            intensity_map[it[j]] = j

        sorted_intensities = sorted(intensity_map.keys(), reverse=True)
        count = 0
        for intensity in sorted_intensities:
            count += 1
            if count > max_peaks:
                to_remove.add(intensity_map[intensity])

    if not to_remove:
        return exp_spectrum

    # Create filtered spectrum
    filtered_mz = []
    filtered_intensity = []
    for i in range(len(mz)):
        if i not in to_remove:
            filtered_mz.append(mz[i])
            filtered_intensity.append(it[i])

    new_spec = MSSpectrum()
    new_spec.setMSLevel(exp_spectrum.getMSLevel())
    new_spec.setPrecursors(exp_spectrum.getPrecursors())
    new_spec.set_peaks((filtered_mz, filtered_intensity))
    return new_spec


# --- Site-determining ions and strict p/n/k pipeline ---
def _generate_isomer_profiles(
    original_sequence: AASequence,
    potential_site_indices: list,
    num_mods_present: int,
    modification_name: str,
    add_decoys: bool = False,
) -> list:
    """Return list of tuples: (AASequence, frozenset(site_indices)).

    Preserves all existing modifications except phosphorylation, then adds new phosphorylation sites.
    """
    profiles = []

    # Get the original sequence string and parse it to preserve non-phosphorylation modifications
    original_seq_str = original_sequence.toString()
    unmodified_seq_str = original_sequence.toUnmodifiedString()

    # Find all non-phosphorylation modifications in the original sequence
    non_phospho_mods = {}

    # Use AASequence to get modification positions directly (more reliable than regex parsing)
    for i in range(original_sequence.size()):
        residue = original_sequence.getResidue(i)
        if residue.getModification() is not None:
            mod_name = residue.getModificationName()
            # Only preserve non-phosphorylation modifications
            if mod_name != "Phospho" and not mod_name.startswith("Phospho"):
                non_phospho_mods[i] = mod_name

    # Create base sequence with preserved modifications
    base_seq = AASequence.fromString(unmodified_seq_str)
    for pos, mod_name in non_phospho_mods.items():
        base_seq.setModification(pos, mod_name)

    for site_indices_tuple in itertools.combinations(
        potential_site_indices, num_mods_present
    ):
        isomer_seq = AASequence(base_seq)  # Copy the base sequence
        for site_index in site_indices_tuple:
            # Use appropriate modification name based on amino acid and add_decoys parameter
            unmodified_seq_str = original_sequence.toUnmodifiedString()
            aa = unmodified_seq_str[site_index]
            if aa == "A" and add_decoys:
                mod_name = "PhosphoDecoy (A)"
            else:
                mod_name = modification_name
            isomer_seq.setModification(site_index, mod_name)
        profiles.append((isomer_seq, frozenset(site_indices_tuple)))
    return profiles


def _expected_fragment_mzs(
    seq: AASequence, precursor_charge: int, add_neutral_losses: bool
) -> list:
    """Return b/y fragment ion m/z values.

    - Generate only b/y ions
    - Charge range: 1 .. min(precursor_charge - 1, MAX_ION_CHARGE)
    - Optional neutral losses
    - Filter out neutral losses approximating phospho (HPO3/PO3H) by name
    """
    spec_gen = TheoreticalSpectrumGenerator()
    params = spec_gen.getParameters()
    params.setValue("add_metainfo", "true")
    params.setValue("add_precursor_peaks", "false")
    params.setValue("add_losses", "true" if add_neutral_losses else "false")
    for ion_type in ["a", "b", "c", "x", "y", "z"]:
        params.setValue(
            f"add_{ion_type}_ions", "true" if ion_type in ("b", "y") else "false"
        )
    spec_gen.setParameters(params)
    theo = MSSpectrum()
    try:
        # Limit fragment charge to (precursor_charge - 1) and not exceeding MAX_ION_CHARGE
        max_frag_z = max(
            1,
            min(
                MAX_ION_CHARGE,
                (
                    (precursor_charge - 1)
                    if precursor_charge and precursor_charge > 1
                    else 1
                ),
            ),
        )
        spec_gen.getSpectrum(theo, seq, 1, max_frag_z)
    except Exception:
        return []
    peaks = theo.get_peaks()
    if not peaks:
        return []
    # Optionally filter out losses approximately equal to phospho mass by name
    try:
        sdas = theo.getStringDataArrays()
        keep_idx = []
        for i, nm in enumerate(list(sdas[0])):
            s = str(nm)
            if ("-HPO3" in s) or ("-PO3H" in s):
                continue
            keep_idx.append(i)
        if keep_idx:
            mzs = [peaks[0][i] for i in keep_idx]
        else:
            mzs = list(peaks[0])
    except Exception:
        mzs = list(peaks[0])
    return mzs


def _get_window_indexes(mz_arr: list, start_mz: float, end_mz: float) -> tuple:
    start = 0
    while start < len(mz_arr) and mz_arr[start] < start_mz:
        start += 1
    end = start
    while end < len(mz_arr) and mz_arr[end] < end_mz:
        end += 1
    return start, end


# --- Helper: Calculate Occurrence Probability "p" ---
def get_occurrence_probability(exp_spectrum: MSSpectrum, tolerance_da: float) -> float:
    """
    Calculates the probability "p" of matching a single theoretical peak
    to any experimental peak by chance.

    Args:
        exp_spectrum: The experimental spectrum (should be peak-picked).
        tolerance_da: The fragment tolerance in Daltons.

    Returns:
        The occurrence probability "p". Returns MIN_PROBABILITY if spectrum is empty or range is invalid.
    """
    try:
        peaks = exp_spectrum.get_peaks()  # Get (mz, intensity) tuples
        num_peaks = len(peaks)
        if num_peaks == 0:
            print("Warning: Empty spectrum in occurrence probability calculation")
            return MIN_PROBABILITY

        # Calculate spectrum range
        mz_values = [p[0] for p in peaks]
        min_mz = min(mz_values)
        max_mz = max(mz_values)

        mz_range = max_mz - min_mz
        if mz_range <= 0:
            print("Warning: Invalid m/z range in occurrence probability calculation")
            return MIN_PROBABILITY

        # Calculate average peak spacing
        if num_peaks > 1:
            mz_diffs = [mz_values[i + 1] - mz_values[i] for i in range(num_peaks - 1)]
            avg_mz_diff = sum(mz_diffs) / len(mz_diffs)
        else:
            avg_mz_diff = tolerance_da * 2  # Use double tolerance if only one peak

        # Calculate probability
        occurrence_prob = (num_peaks * 2.0 * tolerance_da) / mz_range

        # Ensure probability is within reasonable bounds
        min_p = max(MIN_PROBABILITY, 0.01)  # Set minimum probability to 0.01
        max_p = 0.5  # Set maximum probability to 0.5
        occurrence_prob = max(min_p, min(occurrence_prob, max_p))

        print(f"Occurrence probability calculation:")
        print(f"  Number of peaks: {num_peaks}")
        print(f"  m/z range: {mz_range:.4f}")
        print(f"  Average m/z difference: {avg_mz_diff:.4f}")
        print(f"  Tolerance: {tolerance_da:.4f}")
        print(f"  Calculated probability: {occurrence_prob:.4f}")

        return occurrence_prob

    except Exception as e:
        print(f"Error in get_occurrence_probability: {e}")
        return MIN_PROBABILITY


# --- Helper: PhosphoRS-like Scoring Function ---
def calculate_phosphors_score(
    theo_spectrum: MSSpectrum,
    exp_spectrum: MSSpectrum,
    occurrence_probability: float,
    tolerance: float,
    is_ppm: bool,
) -> float:
    """
    Calculates a PhosphoRS-like score based on binomial probability.

    Args:
        theo_spectrum: Theoretical spectrum.
        exp_spectrum: Experimental spectrum (should be peak-picked).
        occurrence_probability (p): Probability of random peak match.
        tolerance: Fragment tolerance.
        is_ppm: Whether tolerance is in ppm.

    Returns:
        PhosphoRS-like score (lower is better).
    """
    try:
        theo_peaks = theo_spectrum.get_peaks()
        exp_peaks = exp_spectrum.get_peaks()

        if not theo_peaks or not exp_peaks:
            print("Warning: Empty spectrum detected")
            return -LOG10_ZERO_REPLACEMENT * (len(theo_peaks[0]) if theo_peaks else 1)

        theo_mz = theo_peaks[0]
        theo_intensity = theo_peaks[1]
        exp_mz = exp_peaks[0]
        exp_intensity = exp_peaks[1]

        p = max(MIN_PROBABILITY, min(occurrence_probability, 1.0 - MIN_PROBABILITY))
        log_p = math.log10(p)
        log_1_minus_p = math.log10(1.0 - p)

        score = 0.0
        exp_idx = 0
        matched_exp_indices = set()

        # Calculate experimental spectrum total intensity
        total_intensity = sum(exp_intensity)
        if total_intensity <= 0:
            total_intensity = 1.0  # Avoid division by zero

        # Add ion intensity weight
        def get_intensity_weight(intensity):
            return (
                math.log10(1 + intensity / total_intensity)
                if total_intensity > 0
                else 1.0
            )

        for i in range(len(theo_mz)):
            theo_mz_val = theo_mz[i]
            theo_intensity_val = theo_intensity[i]

            if is_ppm:
                tol_dalton = theo_mz_val * tolerance / 1_000_000.0
            else:
                tol_dalton = tolerance
            min_mz = theo_mz_val - tol_dalton
            max_mz = theo_mz_val + tol_dalton

            best_match_exp_idx = -1
            min_mass_diff = float("inf")
            best_intensity = 0.0

            # Find best matching experimental peak
            while exp_idx < len(exp_mz) and exp_mz[exp_idx] < min_mz:
                exp_idx += 1

            current_exp_idx = exp_idx
            while current_exp_idx < len(exp_mz) and exp_mz[current_exp_idx] < max_mz:
                if current_exp_idx not in matched_exp_indices:
                    mass_diff = abs(exp_mz[current_exp_idx] - theo_mz_val)
                    if mass_diff < min_mass_diff:
                        min_mass_diff = mass_diff
                        best_match_exp_idx = current_exp_idx
                        best_intensity = exp_intensity[current_exp_idx]
                current_exp_idx += 1

            if best_match_exp_idx != -1:
                # Add ion intensity weight
                intensity_weight = get_intensity_weight(best_intensity)
                # Add mass error weight
                if tol_dalton > 0:
                    mass_error_weight = 1.0 - (min_mass_diff / tol_dalton)
                else:
                    mass_error_weight = 1.0
                # Calculate weighted score
                score += -10.0 * log_p * intensity_weight * mass_error_weight
                matched_exp_indices.add(best_match_exp_idx)
            else:
                score += -10.0 * log_1_minus_p

        # Add penalty for unmatched theoretical peaks
        unmatched_theo_peaks = len(theo_mz) - len(matched_exp_indices)
        if unmatched_theo_peaks > 0:
            score += -10.0 * log_1_minus_p * unmatched_theo_peaks

        return score

    except Exception as e:
        print(f"Error in calculate_phosphors_score: {e}")
        return -LOG10_ZERO_REPLACEMENT


# --- phosphoRS dynamic per-window peak-depth optimization (Taus et al. 2011, ---
# --- pseudocode sections 9-12): per 100 m/z window, choose the peak depth   ---
# --- that maximizes the separation between the best and second-best isoform.---
# Faithful, tested reimplementation replacing the buggy _reduce_by_delta_selection
# (whose depth-selection ratio was inverted and used the experimental peak count
# as the binomial n). See bigbio/onsite#40.

ENABLE_PEAK_DEPTH_OPTIMIZATION = True  # if False, score against the filtered spectrum directly


# compomics chargeValidated gate for peptide-fragment ions, applied to the
# pyOpenMS theoretical spectrum so the binomial trial count n matches the
# reference (D9). Without it, getSpectrum(seq, 1, precursor_charge) emits
# fragments AT the precursor charge and at charges above the ion number (e.g.
# y1 at 2+) - physically impossible ions that inflated n by ~35%.
_ION_PREFIX_RE = re.compile(r"^([abcxyz])(\d+)")


def _theo_mz_charge_valid(spec_gen, seq, precursor_charge) -> list:
    """Charge-validated b/y theoretical fragment m/z for one (modified) peptide.

    Replicates compomics PhosphoRS chargeValidated for PEPTIDE_FRAGMENT_ION:
        fragment charge in 1 .. max(1, precursor_charge - 1)   (charge < precursor)
        and charge <= ion number                               (a y1 cannot be 2+)
    and drops phospho neutral-loss ions by annotation name (-HPO3 / -PO3H). The
    charge upper bound is enforced at generation; the charge<=ion-number gate and
    the loss filter are read from the ion annotations, so ``spec_gen`` MUST have
    ``add_metainfo='true'``. Returns m/z in generator order (caller sorts if
    needed)."""
    max_z = max(1, int(precursor_charge) - 1)
    spec = MSSpectrum()
    try:
        spec_gen.getSpectrum(spec, seq, 1, max_z)
    except Exception:
        return []
    peaks = spec.get_peaks()
    if not peaks or len(peaks[0]) == 0:
        return []
    mzs = peaks[0]
    try:
        annotations = spec.getStringDataArrays()[0]
    except Exception:
        # No annotations: cannot apply the per-ion gate; keep all generated m/z.
        return [float(m) for m in mzs]
    out = []
    for i, raw in enumerate(annotations):
        nm = raw.decode() if isinstance(raw, bytes) else str(raw)
        if ("-HPO3" in nm) or ("-PO3H" in nm):
            continue
        m = _ION_PREFIX_RE.match(nm)
        if m:
            ion_number = int(m.group(2))
            charge = len(nm) - len(nm.rstrip("+"))  # trailing '+' run = charge
            if charge > ion_number:  # chargeValidated: charge must be <= ion number
                continue
        out.append(float(mzs[i]))
    return out


def _isoform_theo_mz(spec_gen, seq_profile, precursor_charge):
    """Sorted, charge-validated theoretical b/y m/z for one isoform (same ion
    model and chargeValidated gate as the final scoring step). ``spec_gen`` must
    have ``add_metainfo='true'`` so the per-ion charge/loss gates apply."""
    return sorted(_theo_mz_charge_valid(spec_gen, seq_profile, precursor_charge))


def _window_has_site_determining_ions(isoform_theo_in_window, tol_da):
    """Pseudocode section 10: True if the isoforms differ in their (tolerance-binned)
    theoretical ion m/z within this window, i.e. the window can distinguish them."""
    if len(isoform_theo_in_window) < 2 or tol_da <= 0:
        return False

    def binned(theo):
        return frozenset(int(round(mz / tol_da)) for mz in theo)

    sets = [binned(t) for t in isoform_theo_in_window]
    return any(s != sets[0] for s in sets[1:])


def _isoform_peptide_scores(selected_mz_sorted, isoform_theo_in_window, tol_da, window_width):
    """phosphoRS peptide score (-10*log10 P_random) for each isoform against the
    selected peaks of one window. n = theoretical ions in window, k = matched,
    p = getp_style(N, window, tol) with N = number of selected peaks (section 9)."""
    n_sel = len(selected_mz_sorted)
    p = getp_style(n_sel, window_width, tol_da)
    scores = []
    for theo in isoform_theo_in_window:
        if not theo:
            scores.append(0.0)
            continue
        # consume-once matcher (Da tolerance within the window)
        n, k = _count_matched_ions(theo, selected_mz_sorted, tol_da, False)
        big_p = binomial_tail_probability(k, n if n > 0 else 1, p)
        scores.append(-10.0 * math.log10(max(big_p, 1e-300)))
    return scores


def _choose_window_depth(window_peaks, isoform_theo_in_window, has_sdi, tol_da,
                         window_width=WINDOW_SIZE, max_depth=MAX_DEPTH):
    """Pseudocode sections 9 & 11: pick the peak depth (1..max_depth) for one window.

    For each depth, score every isoform against the top-`depth` most intense peaks
    and rank them. When the window holds site-determining ions, choose the depth
    that maximizes rank1-rank2 (then rank1-rank3, rank1-rank4, then the best score)
    -- i.e. the depth that best SEPARATES isoforms. Otherwise maximize the best
    absolute score. Ties prefer the smaller depth (fewer noisy peaks).

    Args:
        window_peaks: list of (mz, intensity) in the window (any order).
        isoform_theo_in_window: per-isoform list of theoretical m/z within the window.
    Returns:
        chosen depth (int); 0 if the window has no peaks.
    """
    if not window_peaks:
        return 0
    by_intensity = sorted(window_peaks, key=lambda pk: pk[1], reverse=True)
    max_d = min(max_depth, len(by_intensity))

    best_depth = 0
    best_crit = None
    for depth in range(1, max_d + 1):
        selected = sorted(mz for mz, _it in by_intensity[:depth])
        scores = sorted(
            _isoform_peptide_scores(selected, isoform_theo_in_window, tol_da, window_width),
            reverse=True,
        )
        r1 = scores[0] if len(scores) > 0 else 0.0
        r2 = scores[1] if len(scores) > 1 else 0.0
        r3 = scores[2] if len(scores) > 2 else 0.0
        r4 = scores[3] if len(scores) > 3 else 0.0
        if has_sdi:
            crit = (r1 - r2, r1 - r3, r1 - r4, r1)   # maximize isoform separation
        else:
            crit = (r1, r1 - r2, r1 - r3, r1 - r4)   # maximize best absolute score
        # strict ">" so equal criteria keep the earlier (smaller) depth
        if best_crit is None or crit > best_crit:
            best_crit = crit
            best_depth = depth
    return best_depth


def _reduce_by_peak_depth_optimization(filtered_spec, profiles, fragment_tolerance,
                                       fragment_method_ppm, add_neutral_losses):
    """Pseudocode sections 8-12: split into 100 m/z windows and keep, per window,
    the top-`depth` peaks where `depth` is chosen to best separate the isoforms.
    Returns the reduced MSSpectrum (or the input unchanged if it has no peaks)."""
    peaks = filtered_spec.get_peaks()
    if not peaks or len(peaks[0]) == 0:
        return filtered_spec

    # work on m/z-sorted arrays (the matcher and windowing require ascending m/z)
    order = sorted(range(len(peaks[0])), key=lambda i: peaks[0][i])
    mz_arr = [float(peaks[0][i]) for i in order]
    it_arr = [float(peaks[1][i]) for i in order]
    min_mz, max_mz = mz_arr[0], mz_arr[-1]

    precursor_charge = 2
    if filtered_spec.getPrecursors():
        precursor_charge = filtered_spec.getPrecursors()[0].getCharge() or 2

    # one generator, matching the scoring step's ion model (b/y + optional losses)
    spec_gen = TheoreticalSpectrumGenerator()
    pr = spec_gen.getParameters()
    # metainfo ON: _theo_mz_charge_valid needs ion annotations for the
    # charge<=ion-number gate; it also makes depth selection use the same
    # charge-validated, loss-filtered ion set as final scoring (was D10).
    pr.setValue("add_metainfo", "true")
    pr.setValue("add_precursor_peaks", "false")
    pr.setValue("add_losses", "true" if add_neutral_losses else "false")
    for ion_type in ["a", "b", "c", "x", "y", "z"]:
        pr.setValue(f"add_{ion_type}_ions", "true" if ion_type in ("b", "y") else "false")
    spec_gen.setParameters(pr)

    isoform_theo = [
        _isoform_theo_mz(spec_gen, seq_profile, precursor_charge)
        for seq_profile, _sites in profiles
    ]

    keep = []
    cur = min_mz
    while cur < max_mz:
        hi = cur + WINDOW_SIZE
        # Final window: extend the (exclusive) upper bound past max_mz so a peak
        # sitting exactly on the boundary at max_mz is still included. `cur` still
        # advances by WINDOW_SIZE via `hi`, so the loop terminates.
        sel_hi = hi if hi < max_mz else max_mz + 1.0
        w_start, w_end = _get_window_indexes(mz_arr, cur, sel_hi)
        if w_end - w_start > 0:
            if fragment_method_ppm:
                tol_da = (cur + WINDOW_SIZE / 2.0) * fragment_tolerance / 1_000_000.0
            else:
                tol_da = fragment_tolerance
            win_peaks = [(mz_arr[i], it_arr[i]) for i in range(w_start, w_end)]
            theo_in_win = [[mz for mz in theo if cur <= mz < sel_hi] for theo in isoform_theo]
            has_sdi = _window_has_site_determining_ions(theo_in_win, tol_da)
            depth = _choose_window_depth(win_peaks, theo_in_win, has_sdi, tol_da)
            if depth > 0:
                top = sorted(
                    range(w_start, w_end), key=lambda i: it_arr[i], reverse=True
                )[:depth]
                keep.extend(top)
        cur = hi

    if not keep:
        return filtered_spec
    keep = sorted(set(keep))
    # map back to original (unsorted) indices for _copy_spectrum_subset
    original_idx = sorted(order[i] for i in keep)
    return _copy_spectrum_subset(filtered_spec, original_idx)


# --- Main PhosphoRS-like Localization Function ---
def calculate_phospho_localization_compomics_style(
    peptide_hit: PeptideHit,
    spectrum: MSSpectrum,
    modification_name: str = MODIFICATION_NAME,
    potential_sites: set = POTENTIAL_SITES,
    fragment_tolerance: float = FRAGMENT_TOLERANCE,
    fragment_method_ppm: bool = FRAGMENT_METHOD_PPM,
    add_precursor_peak: bool = ADD_PRECURSOR_PEAK,
    add_ion_types: tuple = ADD_ION_TYPES,
    max_ion_charge: int = MAX_ION_CHARGE,
    add_neutral_losses: bool = ADD_NEUTRAL_LOSSES,
    add_decoys: bool = ADD_DECOYS,
):
    """
    Compute phosphorylation site probabilities using a Compomics-inspired scoring method.
    """
    try:
        if fragment_method_ppm and fragment_tolerance <= 0:
            raise ValueError("PPM tolerance must be positive.")
        if not fragment_method_ppm and fragment_tolerance <= 0:
            raise ValueError("Dalton tolerance must be positive.")

        mod_db = ModificationsDB()

        # Determine potential phosphorylation sites based on add_decoys parameter
        if add_decoys:
            # Include A as potential phosphorylation site when add_decoys=True
            dynamic_potential_sites = potential_sites | {"A"}
        else:
            # Use original potential sites when add_decoys=False
            dynamic_potential_sites = potential_sites

        # Check all possible modification types
        valid_mods = {}
        for aa in dynamic_potential_sites:
            if aa in MODIFICATION_TYPES:
                mod = mod_db.getModification(MODIFICATION_TYPES[aa])
                if mod is not None and mod.getName() != "unknown modification":
                    valid_mods[aa] = mod

        if not valid_mods:
            raise ValueError(
                f"No valid phosphorylation modifications found for sites: {dynamic_potential_sites}"
            )

        # Use the first valid modification as reference
        first_mod = next(iter(valid_mods.values()))
        mod_mass = first_mod.getDiffMonoMass()
        target_residues = set(valid_mods.keys())

        if not target_residues:
            print(f"Warning: No valid target residues found for phosphorylation.")
            return None, None

        original_sequence = peptide_hit.getSequence()
        if not isinstance(original_sequence, AASequence):
            original_sequence = AASequence.fromString(str(original_sequence))

        # --- Determine number of modifications to place ---
        num_mods_present = 0
        try:
            # Count modifications based on add_decoys parameter
            if add_decoys:
                # Count both regular phospho and phospho decoy modifications
                num_mods_present = str(original_sequence).count(
                    f"({MODIFICATION_NAME})"
                ) + str(original_sequence).count("(PhosphoDecoy)")
            else:
                # Count only regular phospho modifications
                mod_string_for_check = f"({MODIFICATION_NAME})"
                num_mods_present = str(original_sequence).count(mod_string_for_check)

            # If no modifications found by string, try mass-based calculation
            if num_mods_present == 0:
                expected_mass_no_mods = AASequence.fromString(
                    original_sequence.toUnmodifiedString()
                ).getMonoWeight()
                expected_mass_with_mods = original_sequence.getMonoWeight()
                mass_diff = expected_mass_with_mods - expected_mass_no_mods

                if abs(mod_mass) > 1e-6:
                    num_mods_present = round(mass_diff / mod_mass)
                    if not math.isclose(
                        num_mods_present * mod_mass, mass_diff, abs_tol=0.1
                    ):
                        # Mass-based calculation failed, use string count (no warning needed)
                        if add_decoys:
                            num_mods_present = str(original_sequence).count(
                                f"({MODIFICATION_NAME})"
                            ) + str(original_sequence).count("(PhosphoDecoy)")
                        else:
                            num_mods_present = str(original_sequence).count(
                                f"({MODIFICATION_NAME})"
                            )
                else:
                    print(
                        f"Warning: Modification mass is zero. Using string count: {num_mods_present}"
                    )

        except Exception as e:
            print(
                f"Error determining number of modifications: {e}. Attempting string count."
            )
            try:
                if add_decoys:
                    num_mods_present = str(original_sequence).count(
                        f"({MODIFICATION_NAME})"
                    ) + str(original_sequence).count("(PhosphoDecoy)")
                else:
                    mod_string_for_check = f"({MODIFICATION_NAME})"
                    num_mods_present = str(original_sequence).count(
                        mod_string_for_check
                    )
            except Exception as e2:
                raise ValueError(
                    f"Could not determine number of modifications in sequence '{original_sequence}' by mass or string: {e2}"
                ) from e

        if num_mods_present <= 0:
            # print(f"Info: No phosphorylation found or inferred in sequence '{original_sequence}'. Cannot localize.")
            return None, None

        unmodified_sequence_str = original_sequence.toUnmodifiedString()
        potential_site_indices = [
            i for i, aa in enumerate(unmodified_sequence_str) if aa in target_residues
        ]

        if len(potential_site_indices) < num_mods_present:
            # print(f"Warning: Potential sites ({len(potential_site_indices)}) < mods to place ({num_mods_present}). Cannot localize.")
            return None, None
        if len(potential_site_indices) == num_mods_present:
            # print(f"Info: Potential sites ({len(potential_site_indices)}) == mods to place ({num_mods_present}). Localization is trivial.")
            trivial_probs = {
                idx: 100.0 for idx in potential_site_indices
            }  # Already percentage
            isomer_seq = AASequence.fromString(unmodified_sequence_str)

            # Copy all existing modifications except phosphorylation
            # Use AASequence to get modification positions directly (more reliable than regex parsing)
            for i in range(original_sequence.size()):
                residue = original_sequence.getResidue(i)
                if residue.getModification() is not None:
                    mod_name = residue.getModificationName()
                    # Only preserve non-phosphorylation modifications
                    if mod_name != "Phospho" and not mod_name.startswith("Phospho"):
                        isomer_seq.setModification(i, mod_name)

            for idx in potential_site_indices:
                aa = unmodified_sequence_str[idx]
                if aa in MODIFICATION_TYPES:
                    isomer_seq.setModification(idx, MODIFICATION_TYPES[aa])
            isomer_list = [(isomer_seq.toString(), 0.0)]
            return trivial_probs, isomer_list

        # --- Pre-calculate for Scoring ---
        if spectrum.getMSLevel() != 2:
            print("Warning: Spectrum MSLevel is not 2. Results may be incorrect.")
        if spectrum.size() == 0:
            raise ValueError("Experimental spectrum contains no peaks.")

        # Determine tolerance in Da. If ppm, convert using precursor or mean mz.
        if fragment_method_ppm:
            if spectrum.getPrecursors():
                prec_mz = spectrum.getPrecursors()[0].getMZ()
            else:
                exp_peaks_for_prec = spectrum.get_peaks()
                if exp_peaks_for_prec and len(exp_peaks_for_prec[0]) > 0:
                    prec_mz = float(np.mean(exp_peaks_for_prec[0]))
                else:
                    prec_mz = 1000.0
            tolerance_da_for_p = prec_mz * fragment_tolerance / 1_000_000.0
        else:
            tolerance_da_for_p = fragment_tolerance

        # --- Spectrum filtering and reduction ---
        # 1) filterSpectrum (keep top peaks per small window based on tolerance)
        filtered_spec = filter_spectrum_for_phosphors(
            spectrum, fragment_tolerance, fragment_method_ppm
        )

        # 2) Prepare all profile sequences (isomers) for SDI-based reduction
        profiles = []  # list of tuples (AASequence, frozenset(site_indices))
        for site_indices_tuple in itertools.combinations(
            potential_site_indices, num_mods_present
        ):
            seq_profile = AASequence.fromString(unmodified_sequence_str)
            # preserve non-phospho mods
            for i in range(original_sequence.size()):
                residue = original_sequence.getResidue(i)
                if residue.getModification() is not None:
                    mod_name = residue.getModificationName()
                    if mod_name != "Phospho" and not mod_name.startswith("Phospho"):
                        seq_profile.setModification(i, mod_name)
            for site_index in site_indices_tuple:
                aa = unmodified_sequence_str[site_index]
                if aa == "A" and add_decoys:
                    mod_name = "PhosphoDecoy (A)"
                else:
                    mod_name = modification_name
                seq_profile.setModification(site_index, mod_name)
            profiles.append((seq_profile, frozenset(site_indices_tuple)))

        if not profiles:
            print("Warning: No isomer profiles generated.")
            return None, None

        # 3) Dynamic per-window peak-depth optimization (phosphoRS, sections 9-12).
        if ENABLE_PEAK_DEPTH_OPTIMIZATION:
            phospho_rs_spec = _reduce_by_peak_depth_optimization(
                filtered_spec, profiles, fragment_tolerance,
                fragment_method_ppm, add_neutral_losses,
            )
        else:
            phospho_rs_spec = filtered_spec

        # --- Generate Theoretical Spectra and Score Against Reduced Spectrum ---
        spec_gen = TheoreticalSpectrumGenerator()
        params = spec_gen.getParameters()
        params.setValue("add_metainfo", "true")
        params.setValue(
            "add_precursor_peaks", "true" if add_precursor_peak else "false"
        )
        params.setValue("add_losses", "true" if add_neutral_losses else "false")
        for ion_type in ["a", "b", "c", "x", "y", "z"]:
            params.setValue(
                f"add_{ion_type}_ions", "true" if ion_type in ("b", "y") else "false"
            )
        spec_gen.setParameters(params)

        isomer_scores = []

        # precursor charge
        if not spectrum.getPrecursors():
            precursor_charge = 2
        else:
            precursor_charge = spectrum.getPrecursors()[0].getCharge()

        # Reduced experimental arrays
        red_peaks = phospho_rs_spec.get_peaks()
        if not red_peaks or len(red_peaks[0]) == 0:
            print("Warning: Reduced spectrum contains no peaks.")
            return None, None
        red_mz_arr = sorted(red_peaks[0])  # ascending: required by _count_matched_ions

        # Random single-ion match probability p = N*d/w (phosphoRS, section 13):
        #   N = number of extracted peaks, d = fragment tolerance,
        #   w = FULL m/z range of the MS/MS spectrum (NOT the extracted-peak span).
        full_mz = spectrum.get_peaks()[0]
        if full_mz is not None and len(full_mz) > 1:
            w = float(np.max(full_mz) - np.min(full_mz))
        else:
            w = float(red_mz_arr[-1] - red_mz_arr[0]) if len(red_mz_arr) > 1 else 0.0
        n_exp_peaks = int(len(red_mz_arr))
        p_calc = getp_style(n_exp_peaks, w, tolerance_da_for_p)

        for seq_profile, site_indices_set in profiles:
            # Charge-validated b/y theoretical m/z (compomics chargeValidated:
            # fragment charge 1..precursor-1, charge <= ion number) with the
            # phospho neutral-loss name filter. Replaces the previous
            # 1..precursor_charge ladder, which over-generated physically
            # impossible ions and inflated the binomial trial count n (D9).
            theo_mz = _theo_mz_charge_valid(spec_gen, seq_profile, precursor_charge)
            if not theo_mz:
                continue

            # Count unique theoretical ions and matches, merging indistinguishable
            # theoretical ions and consuming each experimental peak at most once
            # (red_mz_arr is ascending; see w/n_exp_peaks above).
            n_expected, k_matches = _count_matched_ions(
                theo_mz, red_mz_arr, fragment_tolerance, fragment_method_ppm
            )

            big_p = binomial_tail_probability(
                k_matches, n_expected if n_expected > 0 else 1, p_calc
            )
            p_inv = 1.0 / big_p if big_p > 0 else 0.0

            isomer_scores.append(
                {
                    "isomer_seq": seq_profile,
                    "p_inv": p_inv,
                    "big_p": big_p,
                    "sites": set(site_indices_set),
                }
            )

        # --- Calculate Probabilities using PhosphoRS normalization ---
        if not isomer_scores:
            print("Warning: No isomers were generated or scored.")
            return None, None

        total_p_inv = sum(item["p_inv"] for item in isomer_scores)
        if total_p_inv <= 0.0:
            print("Warning: Total inverse probability is zero. Aborting.")
            return None, None
        for item in isomer_scores:
            item["probability"] = item["p_inv"] / total_p_inv

        # Calculate site probabilities
        site_probabilities = {}
        for item in isomer_scores:
            prob = item["probability"]
            for site_idx in item["sites"]:
                site_probabilities[site_idx] = (
                    site_probabilities.get(site_idx, 0.0) + prob
                )

        # Scale to percentage (keep 0-based indexing)
        site_probabilities = {k: v * 100.0 for k, v in site_probabilities.items()}

        # Format output: return bigP for each profile for transparency
        isomer_list_out = [
            (item["isomer_seq"].toString(), item["big_p"]) for item in isomer_scores
        ]

        return site_probabilities, isomer_list_out

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None


# --- Example Usage ---
if __name__ == "__main__":
    # --- 1. Create Dummy Data ---
    peptide_sequence_str_unmod = "TESTPEPTIDESEK"
    target_phospho_sites_indices = [1, 3, 9]  # Indices for S(1), T(3), S(9)
    num_phospho_to_place = 1

    # Create AASequence with one phospho placed initially (e.g., on S(1))
    initial_seq = AASequence.fromString(peptide_sequence_str_unmod)
    initial_seq.setModification(target_phospho_sites_indices[0], MODIFICATION_NAME)

    ph = PeptideHit()
    ph.setSequence(initial_seq)
    ph.setCharge(2)
    ph.setScore(50.0)  # Dummy search engine score

    # Example Experimental Spectrum
    exp_spec = MSSpectrum()
    exp_spec.setMSLevel(2)

    # Create precursor information
    precursor = Precursor()
    prec_mz = (
        initial_seq.getMonoWeight() + (ph.getCharge() * Constants.PROTON_MASS_U)
    ) / ph.getCharge()
    precursor.setMZ(prec_mz)
    precursor.setCharge(ph.getCharge())
    exp_spec.setPrecursors([precursor])

    # Create peak list
    mz_array = [
        147.11,
        276.15,
        363.18,
        499.13,
        591.30,
        772.31,
        990.33,
        300.1,
        550.2,
        800.3,
    ]
    intensity_array = [
        1000.0,
        800.0,
        600.0,
        1800.0,
        1400.0,
        2000.0,
        900.0,
        200.0,
        300.0,
        240.0,
    ]

    # Sort peaks by m/z
    sorted_indices = sorted(range(len(mz_array)), key=lambda i: mz_array[i])
    mz_array = [mz_array[i] for i in sorted_indices]
    intensity_array = [intensity_array[i] for i in sorted_indices]

    # Add peaks to spectrum
    exp_spec.set_peaks((mz_array, intensity_array))

    # Print spectrum information for debugging
    print("\nSpectrum Information:")
    print(f"Number of peaks: {exp_spec.size()}")
    print(f"Precursor m/z: {prec_mz:.4f}")
    print(f"Precursor charge: {ph.getCharge()}")

    # Verify peaks were added correctly
    peaks = exp_spec.get_peaks()
    if peaks and len(peaks[0]) > 0:
        print(f"First peak m/z: {peaks[0][0]:.4f}, intensity: {peaks[1][0]:.4f}")
        print(f"Last peak m/z: {peaks[0][-1]:.4f}, intensity: {peaks[1][-1]:.4f}")
        print(f"Total intensity: {sum(peaks[1]):.4f}")
    else:
        print("Warning: No peaks found in spectrum after adding them")
        print("Localization could not be performed.")
        sys.exit(1)

    # --- 2. Run Localization ---
    print(f"\nRunning Compomics-style localization for: {ph.getSequence()}")
    print(f"Number of mods to place: {num_phospho_to_place}")
    print(f"Potential site indices (0-based): {target_phospho_sites_indices}")
    print(
        f"Fragment Tolerance: {FRAGMENT_TOLERANCE} {'ppm' if FRAGMENT_METHOD_PPM else 'Da'}"
    )
    print("-" * 20)

    try:
        site_probs, isomer_details = calculate_phospho_localization_compomics_style(
            ph,
            exp_spec,
            fragment_tolerance=FRAGMENT_TOLERANCE,
            fragment_method_ppm=FRAGMENT_METHOD_PPM,
            add_neutral_losses=ADD_NEUTRAL_LOSSES,
        )

        # --- 3. Print Results ---
        if site_probs is not None:
            print("Isomer Scores (Lower is better):")
            # Sort by score (ascending)
            for seq_str, score in sorted(isomer_details, key=lambda x: x[1]):
                print(f"  - {seq_str}: {score:.4f}")

            print("\nSite Probabilities:")
            # Sort by index for clarity
            sorted_sites = sorted(site_probs.items())
            total_prob = 0.0
            for site_index, probability in sorted_sites:
                aa = peptide_sequence_str_unmod[site_index]
                print(f"  - Site {site_index} ({aa}): {probability:.4f}")
                total_prob += probability
            print(f"Total Probability Sum: {total_prob:.4f}")  # Should be close to 1.0

            # Construct sequence string with probabilities
            result_seq = ""
            max_prob_site = -1
            max_prob = -1.0
            sites_with_prob = set(site_probs.keys())
            for i, aa in enumerate(peptide_sequence_str_unmod):
                result_seq += aa
                if i in sites_with_prob and site_probs[i] > 0.0001:
                    prob = site_probs[i]
                    result_seq += f"({MODIFICATION_NAME};{prob:.2f})"
                    if prob > max_prob:
                        max_prob = prob
                        max_prob_site = i
            print(f"\nSequence with probabilities: {result_seq}")
            print(
                f"(Highest probability site: {max_prob_site} ({peptide_sequence_str_unmod[max_prob_site]}) with P={max_prob:.4f})"
            )

        else:
            print("Localization could not be performed.")

    except ValueError as e:
        print(f"ValueError during localization: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
