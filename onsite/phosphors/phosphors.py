import itertools
import math
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
MIN_DEPTH = 2

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
    """Mimic Util.floorDouble(x, nDecimals)."""
    if n_decimals <= 0:
        return math.floor(value)
    factor = 10.0**n_decimals
    return math.floor(value * factor) / factor


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
    global _distribution_cache

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
    global _distribution_cache
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


# --- Window reduction helpers ---
def _get_intensity_thresholds(mz: list, intensity: list, start: int, end: int) -> list:
    """Return up to MAX_DEPTH descending unique intensity thresholds in window [start, end)."""
    window_intensities = sorted(set(intensity[start:end]), reverse=True)
    return window_intensities[:MAX_DEPTH]


def _count_peaks_above_threshold(
    intensity: list, start: int, end: int, thr: float
) -> int:
    c = 0
    for i in range(start, end):
        if intensity[i] >= thr:
            c += 1
    return c


def _reduce_spectrum_by_windows(
    exp_spectrum: MSSpectrum,
    tol_da_for_p: float,
    is_ppm: bool,
    fragment_tolerance_value: float,
) -> tuple:
    """
    Reduce spectrum: split into WINDOW_SIZE mz windows, choose best depth between MIN_DEPTH..MAX_DEPTH
    that minimizes bigP with p computed using window nPeaks and tolerance.
    Returns reduced mz and intensity arrays.
    """
    peaks = exp_spectrum.get_peaks()
    mz_arr = list(peaks[0])
    it_arr = list(peaks[1])
    if len(mz_arr) == 0:
        return mz_arr, it_arr
    min_mz = float(mz_arr[0])
    max_mz = float(mz_arr[-1])
    reduced_indexes = []
    cur_min = min_mz
    half_window = WINDOW_SIZE / 2.0
    while cur_min < max_mz:
        temp_max = cur_min + WINDOW_SIZE
        # determine tolerance d in Da at window center if ppm
        if is_ppm:
            ref_mz = cur_min + half_window
            d = ref_mz * (fragment_tolerance_value) / 1_000_000.0
        else:
            d = tol_da_for_p
        # window indexes [start, end)
        # binary search could be used; linear scan ok for modest N
        start = 0
        while start < len(mz_arr) and mz_arr[start] < cur_min:
            start += 1
        end = start
        while end < len(mz_arr) and mz_arr[end] < temp_max:
            end += 1
        if end - start > 0:
            thresholds = _get_intensity_thresholds(mz_arr, it_arr, start, end)
            if thresholds:
                best_i = 0
                best_bigp = 1.0
                for depth_idx in range(1, len(thresholds) + 1):
                    thr = thresholds[depth_idx - 1]
                    n_peaks = _count_peaks_above_threshold(it_arr, start, end, thr)
                    p_win = getp_style(n_peaks, WINDOW_SIZE, d)
                    # approximate expected ions: use n_peaks as n for window selection
                    bigp = binomial_tail_probability(
                        max(1, min(n_peaks, 1)), n_peaks if n_peaks > 0 else 1, p_win
                    )
                    if bigp < best_bigp:
                        best_bigp = bigp
                        best_i = depth_idx - 1
                # enforce depth bounds
                if best_i < MIN_DEPTH - 1 and (MIN_DEPTH - 1) < len(thresholds):
                    best_i = MIN_DEPTH - 1
                if best_i > MAX_DEPTH - 1:
                    best_i = MAX_DEPTH - 1
                best_thr = thresholds[best_i]
                window_best = [i for i in range(start, end) if it_arr[i] >= best_thr]
                if window_best:
                    reduced_indexes.extend(window_best)
        cur_min = temp_max
    if not reduced_indexes:
        return mz_arr, it_arr
    reduced_indexes.sort()
    red_mz = [mz_arr[i] for i in reduced_indexes]
    red_it = [it_arr[i] for i in reduced_indexes]
    return red_mz, red_it


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
    original_seq_str = str(original_sequence)
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


def _site_determining_ions(
    profiles: list, precursor_charge: int, add_neutral_losses: bool
) -> tuple:
    """Return (sdi_map, profile_to_mzs, union_mzs).

    - sdi_map: mz -> set(profile_index) for ions not common to all profiles
    - profile_to_mzs: list[set] of m/z per profile
    - union_mzs: set of all m/z across profiles
    """
    site_determining_ions = {}
    common_ions = {}
    profile_to_mzs = []

    for profile_idx, (seq, _sites) in enumerate(profiles):
        mzs = _expected_fragment_mzs(seq, precursor_charge, add_neutral_losses)
        # Use raw theoretical m/z (no rounding) to align with expected ions
        profile_mzs = set(float(m) for m in mzs)
        profile_to_mzs.append(profile_mzs)

        for mz in profile_mzs:
            if not common_ions:  # First profile
                common_ions[mz] = {profile_idx}
            elif mz not in common_ions:
                # This m/z is not common to all previous profiles
                if mz in site_determining_ions:
                    site_determining_ions[mz].add(profile_idx)
                else:
                    site_determining_ions[mz] = {profile_idx}
            else:
                # This m/z is still common
                common_ions[mz].add(profile_idx)

        # Check if any previously common ions are no longer common
        ions_to_remove = []
        for mz in common_ions:
            if mz not in profile_mzs:
                # This ion is no longer common, move to site-determining
                site_determining_ions[mz] = common_ions[mz]
                ions_to_remove.append(mz)
            else:
                # Still common, add current profile
                common_ions[mz].add(profile_idx)

        # Remove ions that are no longer common
        for mz in ions_to_remove:
            del common_ions[mz]

    # All remaining common ions are truly common to all profiles
    all_mzs = set()
    for mz_set in profile_to_mzs:
        all_mzs.update(mz_set)

    return site_determining_ions, profile_to_mzs, all_mzs


def _window_intensity_thresholds(
    filtered_spec: MSSpectrum, start: int, end: int
) -> list:
    peaks = filtered_spec.get_peaks()
    intens = sorted(set(float(peaks[1][i]) for i in range(start, end)), reverse=True)
    return intens[:MAX_DEPTH]


def _get_window_indexes(mz_arr: list, start_mz: float, end_mz: float) -> tuple:
    start = 0
    while start < len(mz_arr) and mz_arr[start] < start_mz:
        start += 1
    end = start
    while end < len(mz_arr) and mz_arr[end] < end_mz:
        end += 1
    return start, end


def _reduce_by_delta_selection(
    filtered_spec: MSSpectrum,
    profiles: list,
    fragment_tolerance: float,
    fragment_method_ppm: bool,
) -> MSSpectrum:
    """Reduce spectrum using delta-based depth selection with site-determining ions across profiles."""
    peaks = filtered_spec.get_peaks()
    if not peaks or len(peaks[0]) == 0:
        return filtered_spec
    mz_arr = list(peaks[0])
    it_arr = list(peaks[1])
    min_mz = float(mz_arr[0])
    max_mz = float(mz_arr[-1])

    # precursor charge estimation
    if filtered_spec.getPrecursors():
        precursor_charge = filtered_spec.getPrecursors()[0].getCharge() or 2
    else:
        precursor_charge = 2

    sdi_map, profile_to_mzs, union_mzs = _site_determining_ions(
        profiles, precursor_charge, ADD_NEUTRAL_LOSSES
    )
    reduced_idx = []
    cur_min = min_mz
    half_win = WINDOW_SIZE / 2.0

    while cur_min < max_mz:
        temp_max = cur_min + WINDOW_SIZE
        # tolerance in Da at window center if ppm
        if fragment_method_ppm:
            ref_mz = cur_min + half_win
            d = ref_mz * fragment_tolerance / 1_000_000.0
        else:
            d = fragment_tolerance

        w_start, w_end = _get_window_indexes(mz_arr, cur_min, temp_max)
        if w_end - w_start > 0:
            thresholds = _window_intensity_thresholds(filtered_spec, w_start, w_end)
            if thresholds:
                # Delta selection logic
                deltas = []
                n_deltas = 0

                for depth_idx in range(1, len(thresholds) + 1):
                    thr = thresholds[depth_idx - 1]
                    n_peaks = _count_peaks_above_threshold(it_arr, w_start, w_end, thr)
                    p_win = getp_style(n_peaks, WINDOW_SIZE, d)

                    # Check if there are site-determining ions in this window
                    profile_to_sdi_mz = {}
                    for mz_rounded, present_profiles in sdi_map.items():
                        mz_val = float(mz_rounded)
                        # Boundaries: (ionMz > minMz) && (ionMz <= tempMax)
                        if (mz_val > cur_min) and (mz_val <= temp_max):
                            for prof_idx in present_profiles:
                                if prof_idx not in profile_to_sdi_mz:
                                    profile_to_sdi_mz[prof_idx] = set()
                                profile_to_sdi_mz[prof_idx].add(mz_val)

                    if profile_to_sdi_mz:
                        # Branch: site-determining ions present
                        big_ps = []
                        scored_sets = []
                        profile_with_no_sdi_scored = False

                        for prof_idx, (_seq, _sites) in enumerate(profiles):
                            profile_sdi = profile_to_sdi_mz.get(prof_idx, set())
                            profile_theo_mzs = (
                                profile_to_mzs[prof_idx]
                                if prof_idx < len(profile_to_mzs)
                                else set()
                            )

                            if not profile_sdi:
                                if not profile_with_no_sdi_scored:
                                    profile_with_no_sdi_scored = True
                                    # Count matches for profiles without SDIs using all expected ions for the profile
                                    k = 0
                                    for mz_theo in profile_theo_mzs:
                                        if (mz_theo > cur_min) and (
                                            mz_theo <= temp_max
                                        ):
                                            min_b = mz_theo - d
                                            max_b = mz_theo + d
                                            for i_idx in range(w_start, w_end):
                                                if it_arr[i_idx] >= thr:
                                                    mz_exp = mz_arr[i_idx]
                                                    if (mz_exp >= min_b) and (
                                                        mz_exp < max_b
                                                    ):
                                                        k += 1
                                                        break
                                    big_p = binomial_tail_probability(k, n_peaks, p_win)
                                    big_ps.append(big_p)
                            else:
                                # Check if this SDI set was already scored
                                already_scored = False
                                for scored_set in scored_sets:
                                    if profile_sdi == scored_set:
                                        already_scored = True
                                        break

                                if not already_scored:
                                    scored_sets.append(profile_sdi)
                                    # Count matches for this profile using ALL matched fragments (not only SDIs)
                                    k = 0
                                    for mz_theo in profile_theo_mzs:
                                        if (mz_theo > cur_min) and (
                                            mz_theo <= temp_max
                                        ):
                                            min_b = mz_theo - d
                                            max_b = mz_theo + d
                                            for i_idx in range(w_start, w_end):
                                                if it_arr[i_idx] >= thr:
                                                    mz_exp = mz_arr[i_idx]
                                                    if (mz_exp >= min_b) and (
                                                        mz_exp < max_b
                                                    ):
                                                        k += 1
                                                        break
                                    big_p = binomial_tail_probability(k, n_peaks, p_win)
                                    big_ps.append(big_p)

                        if len(big_ps) > 1:
                            big_ps.sort()
                            current_deltas = []
                            for j in range(len(big_ps) - 1):
                                pj = big_ps[j]
                                pj1 = big_ps[j + 1]
                                if pj1 > 0:
                                    delta = pj / pj1
                                    current_deltas.append(delta)

                            if len(current_deltas) > n_deltas:
                                n_deltas = len(current_deltas)
                            deltas.append(current_deltas)
                        else:
                            deltas.append([])
                    else:
                        # Branch: no site-determining ions present in this window
                        # Select depth that minimizes bigP across depths
                        pass

                # If no SDIs for the entire window, perform best depth selection by minimal bigP
                if all(len(d) == 0 for d in deltas):
                    best_i = 0
                    best_bigp = 1.0
                    for depth_idx in range(1, len(thresholds) + 1):
                        thr = thresholds[depth_idx - 1]
                        n_peaks = _count_peaks_above_threshold(
                            it_arr, w_start, w_end, thr
                        )
                        p_win = getp_style(n_peaks, WINDOW_SIZE, d)
                        # Approximate k by matching union of theoretical ions across profiles within tolerance
                        k = 0
                        for mz_theo in union_mzs:
                            if (mz_theo > cur_min) and (mz_theo <= temp_max):
                                min_b = mz_theo - d
                                max_b = mz_theo + d
                                for i_idx in range(w_start, w_end):
                                    if it_arr[i_idx] >= thr:
                                        mz_exp = mz_arr[i_idx]
                                        if (mz_exp >= min_b) and (mz_exp < max_b):
                                            k += 1
                                            break
                        big_p = binomial_tail_probability(
                            k, n_peaks if n_peaks > 0 else 1, p_win
                        )
                        if big_p < best_bigp:
                            best_bigp = big_p
                            best_i = depth_idx - 1

                    # Enforce depth bounds
                    if best_i < MIN_DEPTH - 1 and (MIN_DEPTH - 1) < len(thresholds):
                        best_i = MIN_DEPTH - 1
                    if best_i > MAX_DEPTH - 1:
                        best_i = MAX_DEPTH - 1

                    best_thr = thresholds[best_i]
                    window_best = [
                        i for i in range(w_start, w_end) if it_arr[i] >= best_thr
                    ]
                    if window_best:
                        reduced_idx.extend(window_best)
                else:
                    # Find best depth using delta selection
                    best_i = 0
                    largest_delta = 0.0
                    for j in range(n_deltas):
                        if largest_delta == 0.0:
                            for i in range(len(deltas)):
                                temp_deltas = deltas[i]
                                if (
                                    j < len(temp_deltas)
                                    and temp_deltas[j] > largest_delta
                                ):
                                    largest_delta = temp_deltas[j]
                                    best_i = i
                    # Enforce depth bounds
                    if best_i < MIN_DEPTH - 1 and (MIN_DEPTH - 1) < len(thresholds):
                        best_i = MIN_DEPTH - 1
                    if best_i > MAX_DEPTH - 1:
                        best_i = MAX_DEPTH - 1
                    best_thr = thresholds[best_i]
                    window_best = [
                        i for i in range(w_start, w_end) if it_arr[i] >= best_thr
                    ]
                    if window_best:
                        reduced_idx.extend(window_best)

        cur_min = temp_max

    if not reduced_idx:
        return filtered_spec
    reduced_idx.sort()
    return _copy_spectrum_subset(filtered_spec, reduced_idx)


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
            isomer_list = [(str(isomer_seq), 0.0)]
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

        # 3) Reduce spectrum by SDI delta-selection to obtain phosphoRsSpectrum
        phospho_rs_spec = _reduce_by_delta_selection(
            filtered_spec, profiles, fragment_tolerance, fragment_method_ppm
        )

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
        red_mz_arr = list(red_peaks[0])

        # p uses number of peaks and window width of reduced spectrum, tolerance in Da
        w = float(red_mz_arr[-1] - red_mz_arr[0]) if len(red_mz_arr) > 1 else float(0.0)
        n_exp_peaks = int(len(red_mz_arr))
        p_calc = getp_style(n_exp_peaks, w, tolerance_da_for_p)

        for seq_profile, site_indices_set in profiles:
            theo_spectrum = MSSpectrum()
            try:
                spec_gen.getSpectrum(theo_spectrum, seq_profile, 1, precursor_charge)
            except Exception as e:
                print(f"Warning: Failed to generate theoretical spectrum: {e}")
                continue

            theo_peaks = theo_spectrum.get_peaks()
            if not theo_peaks or len(theo_peaks[0]) == 0:
                continue

            # Filter out neutral losses ~ phospho mass by name
            theo_mz_all = list(theo_peaks[0])
            theo_keep_idx = list(range(len(theo_mz_all)))
            try:
                sdas = theo_spectrum.getStringDataArrays()
                if sdas and len(sdas) > 0:
                    annotations = list(sdas[0])
                    filtered_idx = []
                    for i, name in enumerate(annotations):
                        nm = str(name)
                        if ("-HPO3" in nm) or ("-PO3H" in nm):
                            continue
                        filtered_idx.append(i)
                    theo_keep_idx = filtered_idx if filtered_idx else theo_keep_idx
            except Exception:
                pass
            theo_mz = [theo_mz_all[i] for i in theo_keep_idx]
            n_expected = len(theo_mz)

            # Count matches k using reduced spectrum and tolerance
            k_matches = 0
            red_len = len(red_mz_arr)
            red_idx = 0
            for mz_theo in theo_mz:
                tol_da = (
                    (mz_theo * fragment_tolerance / 1_000_000.0)
                    if fragment_method_ppm
                    else fragment_tolerance
                )
                min_b = mz_theo - tol_da
                max_b = mz_theo + tol_da
                while red_idx < red_len and red_mz_arr[red_idx] < min_b:
                    red_idx += 1
                cur_idx = red_idx
                matched = False
                while cur_idx < red_len and red_mz_arr[cur_idx] < max_b:
                    matched = True
                    break
                if matched:
                    k_matches += 1

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
            (str(item["isomer_seq"]), item["big_p"]) for item in isomer_scores
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
