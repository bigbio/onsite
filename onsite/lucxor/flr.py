"""
FLR (False Localization Rate) calculation module.

This module contains the FLRCalculator class for calculating false localization rates.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from .constants import REAL, DECOY, TINY_NUM

logger = logging.getLogger(__name__)


class FLRCalculator:
    """False Localization Rate calculator."""

    def __init__(self, min_delta_score: float = 0.1, min_psms_per_charge: int = 50):
        """
        Initialize the FLR calculator.

        Args:
            min_delta_score: Minimum delta score threshold
            min_psms_per_charge: Minimum PSM count per charge
        """
        self.real_psms = []  # Target sequence PSMs
        self.decoy_psms = []  # Decoy sequence PSMs
        self.max_delta_score = 0.0
        self.n_real = 0
        self.n_decoy = 0

        # Kernel density estimation parameters
        self.delta_score_var_pos = 0.0  # Target sequence delta score variance
        self.delta_score_var_neg = 0.0  # Decoy sequence delta score variance
        self.delta_score_mu_pos = 0.0  # Target sequence delta score mean
        self.delta_score_mu_neg = 0.0  # Decoy sequence delta score mean
        self.bw_real = 0.0  # Target sequence bandwidth
        self.bw_decoy = 0.0  # Decoy sequence bandwidth

        # Constants
        self.NMARKS = 10001  # Number of tick marks
        self.min_delta_score = min_delta_score  # Minimum delta score threshold
        self.min_psms_per_charge = min_psms_per_charge  # Minimum PSM count per charge

        # Store density estimation results
        self.tick_marks = None
        self.f0 = None  # Decoy sequence density
        self.f1 = None  # Target sequence density

        # FDR results
        self.global_fdr = None
        self.local_fdr = None
        self.minor_map_g = {}  # Global FDR mapping
        self.minor_map_l = {}  # Local FDR mapping

        # Store delta score to FLR mapping (for second round calculation)
        self.delta_score_to_flr_map = {}  # delta_score -> (global_flr, local_flr)

        logger.debug("FLRCalculator initialized")

    def add_psm(self, delta_score: float, is_decoy: bool) -> None:
        """
        Add PSM to calculator

        Args:
            delta_score: Delta score
            is_decoy: Whether it's a decoy sequence
        """
        logger.debug(f"Adding PSM - delta_score: {delta_score}, is_decoy: {is_decoy}")

        if delta_score > self.max_delta_score:
            self.max_delta_score = delta_score

        if is_decoy:
            self.decoy_psms.append(delta_score)
            self.n_decoy += 1
        else:
            self.real_psms.append(delta_score)
            self.n_real += 1

        logger.debug(
            f"Current real_psms count: {self.n_real}, decoy_psms count: {self.n_decoy}"
        )

    def prep_arrays(self) -> None:
        """Prepare arrays"""
        self.pos = np.array(self.real_psms)
        self.neg = np.array(self.decoy_psms)
        self.n_real = len(self.pos)
        self.n_decoy = len(self.neg)

    def initialize_tick_marks(self) -> None:
        """Initialize tick marks (vectorized)"""
        # First prepare arrays
        self.prep_arrays()

        self.max_delta_score *= 1.001  # Need to be slightly larger for binning

        # Vectorized tick mark creation using linspace
        self.tick_marks = np.linspace(0, self.max_delta_score, self.NMARKS)

        # Initialize other arrays
        self.local_fdr = np.zeros(self.n_real)
        self.global_fdr = np.zeros(self.n_real)

        self.calc_delta_score_mean()
        self.calc_delta_score_var()

        self.get_bandwidth(DECOY)  # Decoy
        self.get_bandwidth(REAL)  # Target

        # Output bandwidth information
        logger.info(f"FLR bandwidth (pos): {self.bw_real:.6f}")
        logger.info(f"FLR bandwidth (neg): {self.bw_decoy:.6f}")

    def get_bandwidth(self, data_type: int) -> None:
        """
        Calculate bandwidth

        Args:
            data_type: Data type (REAL or DECOY)
        """
        if data_type == REAL:
            sigma = np.sqrt(self.delta_score_var_pos)
            N = float(self.n_real)
            x = np.power(N, 0.2)
            result = 1.06 * (sigma / x)
            self.bw_real = result

        elif data_type == DECOY:
            sigma = np.sqrt(self.delta_score_var_neg)
            N = float(self.n_decoy)
            x = np.power(N, 0.2)
            result = 1.06 * (sigma / x)
            self.bw_decoy = result

    def calc_delta_score_mean(self) -> None:
        """Calculate delta score mean (vectorized)"""
        # Target sequence - use NumPy's optimized mean
        self.delta_score_mu_pos = np.mean(self.pos) if self.n_real > 0 else 0.0

        # Decoy sequence - use NumPy's optimized mean
        self.delta_score_mu_neg = np.mean(self.neg) if self.n_decoy > 0 else 0.0

    def calc_delta_score_var(self) -> None:
        """Calculate delta score variance (vectorized)"""
        # Target sequence - use NumPy's optimized variance (ddof=1 for sample variance)
        self.delta_score_var_pos = np.var(self.pos, ddof=1) if self.n_real > 1 else 0.0

        # Decoy sequence - use NumPy's optimized variance (ddof=1 for sample variance)
        self.delta_score_var_neg = np.var(self.neg, ddof=1) if self.n_decoy > 1 else 0.0

    def normal_density(self, cur_tick_mark: float, cur_score: float, h: float) -> float:
        """
        Calculate normal density

        Args:
            cur_tick_mark: Current tick mark
            cur_score: Current score
            h: Bandwidth

        Returns:
            Density value
        """
        x = (cur_tick_mark - cur_score) / h
        return np.exp(-0.5 * x * x) / (h * np.sqrt(2.0 * np.pi))

    def eval_tick_marks(self, data_type: int) -> None:
        """
        Evaluate tick marks, kernel density estimation implementation (VECTORIZED)
        Args:
            data_type: Data type (REAL or DECOY)
        """
        if data_type == DECOY:
            data_ary = self.neg
            bw = self.bw_decoy
            N = float(self.n_decoy)
        else:  # REAL
            data_ary = self.pos
            bw = self.bw_real
            N = float(self.n_real)

        if N == 0 or bw == 0:
            # Avoid division by zero
            if data_type == DECOY:
                self.f0 = np.full(self.NMARKS, TINY_NUM)
            else:
                self.f1 = np.full(self.NMARKS, TINY_NUM)
            return

        # VECTORIZED kernel density estimation - O(n+m) instead of O(n*m)
        # Use broadcasting: tick_marks (NMARKS,) and data_ary (N,)
        # Compute (tick_marks[:, None] - data_ary[None, :]) / bw for all pairs
        # For large datasets, process in chunks to avoid memory issues
        CHUNK_SIZE = 5000  # Process data in chunks to avoid huge memory allocation

        NORMAL_CONSTANT = 1.0 / np.sqrt(2.0 * np.pi)
        kernel_result = np.zeros(self.NMARKS)

        if len(data_ary) <= CHUNK_SIZE:
            # Small dataset: fully vectorized computation
            # Shape: (NMARKS, N_data)
            diff = (self.tick_marks[:, np.newaxis] - data_ary[np.newaxis, :]) / bw
            kernel = NORMAL_CONSTANT * np.exp(-0.5 * diff * diff)
            kernel_result = kernel.sum(axis=1)  # Accumulate only, normalize later
        else:
            # Large dataset: process in chunks to avoid memory issues
            for chunk_start in range(0, len(data_ary), CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, len(data_ary))
                chunk_data = data_ary[chunk_start:chunk_end]

                diff = (self.tick_marks[:, np.newaxis] - chunk_data[np.newaxis, :]) / bw
                kernel = NORMAL_CONSTANT * np.exp(-0.5 * diff * diff)
                kernel_result += kernel.sum(axis=1)

        # Single normalization after accumulation for both branches
        # This is critical for numerical stability - normalizing once at the end
        # preserves precision better than normalizing during accumulation
        kernel_result /= N * bw

        # Apply minimum threshold
        kernel_result = np.maximum(kernel_result, TINY_NUM)

        if data_type == DECOY:
            self.f0 = kernel_result
        else:
            self.f1 = kernel_result

        # Add debug information
        if data_type == DECOY:
            logger.info(
                f"DECOY KDE calculation completed - data points: {len(data_ary)}, bandwidth: {bw:.6f}"
            )
            logger.info(
                f"DECOY density range: min={np.min(self.f0):.6e}, max={np.max(self.f0):.6e}"
            )
        else:
            logger.info(
                f"REAL KDE calculation completed - data points: {len(data_ary)}, bandwidth: {bw:.6f}"
            )
            logger.info(
                f"REAL density range: min={np.min(self.f1):.6e}, max={np.max(self.f1):.6e}"
            )

    def _interpolate_density_vectorized(self, x_values: np.ndarray, f: np.ndarray) -> np.ndarray:
        """
        Vectorized linear interpolation of density values at given points.

        Uses binary search (O(log n)) instead of linear search (O(n)).

        Args:
            x_values: Array of score values to interpolate at
            f: Density array (f0 or f1)

        Returns:
            Array of interpolated density values
        """
        # Use searchsorted for O(log n) binary search to find intervals
        # Returns index where x would be inserted to maintain sorted order
        indices = np.searchsorted(self.tick_marks, x_values, side='right') - 1

        # Clamp indices to valid range [0, NMARKS-2] for interpolation
        indices = np.clip(indices, 0, self.NMARKS - 2)

        # Get interval boundaries
        a = self.tick_marks[indices]      # Left boundary
        b = self.tick_marks[indices + 1]  # Right boundary

        # Linear interpolation weights
        t = (x_values - a) / (b - a)  # Interpolation parameter [0, 1]

        # Interpolated values: f[i] * (1-t) + f[i+1] * t
        result = f[indices] * (1 - t) + f[indices + 1] * t

        # Handle boundary cases
        result = np.where(x_values <= self.tick_marks[0], f[0], result)
        result = np.where(x_values >= self.tick_marks[-1], f[-1], result)

        return result

    def _compute_cumulative_auc_from_end(self, f: np.ndarray) -> np.ndarray:
        """
        Pre-compute cumulative AUC from end of tick_marks array.

        This allows O(1) lookup of the area from any tick mark to the end.

        Args:
            f: Density array (f0 or f1)

        Returns:
            Array where cumulative_auc[i] = area from tick_marks[i] to end
        """
        # Compute trapezoid areas between consecutive tick marks
        # Area of trapezoid = (b - a) * (f[i] + f[i+1]) / 2
        dx = np.diff(self.tick_marks)  # Width of each interval
        trapezoid_areas = dx * 0.5 * (f[:-1] + f[1:])

        # Cumulative sum from end (reverse cumsum)
        # cumulative_auc[i] = sum of all trapezoid areas from i to end
        cumulative_auc = np.zeros(self.NMARKS)
        cumulative_auc[:-1] = np.cumsum(trapezoid_areas[::-1])[::-1]

        return cumulative_auc

    def _global_auc_vectorized(self, x_values: np.ndarray, f: np.ndarray,
                                cumulative_auc: np.ndarray) -> np.ndarray:
        """
        Vectorized global AUC calculation (area from x to end of distribution).

        Uses pre-computed cumulative AUC and binary search for O(log n) per value.

        Args:
            x_values: Array of score values
            f: Density array (f0 or f1)
            cumulative_auc: Pre-computed cumulative AUC from end

        Returns:
            Array of AUC values (area from each x to end)
        """
        # Find which interval each x falls into
        indices = np.searchsorted(self.tick_marks, x_values, side='right') - 1
        indices = np.clip(indices, 0, self.NMARKS - 2)

        # Get interval boundaries
        a = self.tick_marks[indices]      # Left boundary of interval containing x
        b = self.tick_marks[indices + 1]  # Right boundary

        # Interpolate density at x
        t = (x_values - a) / (b - a)
        fx = f[indices] * (1 - t) + f[indices + 1] * t

        # Area from x to right boundary of current interval
        # Trapezoid with vertices at (x, 0), (x, fx), (b, f[i+1]), (b, 0)
        partial_area = (b - x_values) * 0.5 * (fx + f[indices + 1])

        # Total area = partial area in current interval + cumulative area from next interval
        result = partial_area + cumulative_auc[indices + 1]

        # Handle boundary cases
        result = np.where(x_values <= self.tick_marks[0], cumulative_auc[0], result)
        result = np.where(x_values >= self.tick_marks[-1], 0.0, result)

        return result

    def get_local_auc(self, x: float, which_f: int) -> float:
        """
        Calculate local density value (density at point x).

        Note: This is kept for backwards compatibility. For batch processing,
        use _interpolate_density_vectorized() directly.

        Args:
            x: Score
            which_f: Which distribution to use (f0 or f1)

        Returns:
            Density value
        """
        f = self.f0 if which_f == DECOY else self.f1
        return float(self._interpolate_density_vectorized(np.array([x]), f)[0])

    def get_global_auc(self, x: float, which_f: int) -> float:
        """
        Calculate global AUC (area from x to end of distribution).

        Note: This is kept for backwards compatibility. For batch processing,
        use calc_both_fdrs() which uses vectorized computation.

        Args:
            x: Score
            which_f: Which distribution to use (f0 or f1)

        Returns:
            AUC value
        """
        if which_f == DECOY:
            f = self.f0
            if not hasattr(self, '_cumulative_auc_f0'):
                self._cumulative_auc_f0 = self._compute_cumulative_auc_from_end(self.f0)
            cumulative_auc = self._cumulative_auc_f0
        else:
            f = self.f1
            if not hasattr(self, '_cumulative_auc_f1'):
                self._cumulative_auc_f1 = self._compute_cumulative_auc_from_end(self.f1)
            cumulative_auc = self._cumulative_auc_f1

        return float(self._global_auc_vectorized(np.array([x]), f, cumulative_auc)[0])

    def calc_both_fdrs(self) -> None:
        """
        Calculate global and local FDR for all PSMs.

        Fully vectorized implementation using NumPy for O(n log m) complexity
        where n = number of PSMs and m = number of tick marks.
        """
        Nreal2 = float(self.n_real)
        Ndecoy2 = float(self.n_decoy)

        logger.info(
            f"FDR calculation - Real PSM count: {Nreal2}, Decoy PSM count: {Ndecoy2}"
        )

        # Pre-compute cumulative AUC arrays for global FDR (O(m) once)
        cumulative_auc_f0 = self._compute_cumulative_auc_from_end(self.f0)
        cumulative_auc_f1 = self._compute_cumulative_auc_from_end(self.f1)

        # Cache for potential reuse
        self._cumulative_auc_f0 = cumulative_auc_f0
        self._cumulative_auc_f1 = cumulative_auc_f1

        # Apply minimum threshold to all scores at once
        x_values = np.maximum(self.pos, 0.1)

        # Vectorized global AUC calculation (O(n log m))
        g_auc_f0 = self._global_auc_vectorized(x_values, self.f0, cumulative_auc_f0)
        g_auc_f1 = self._global_auc_vectorized(x_values, self.f1, cumulative_auc_f1)

        # Vectorized local density calculation (O(n log m))
        l_auc_f0 = self._interpolate_density_vectorized(x_values, self.f0)
        l_auc_f1 = self._interpolate_density_vectorized(x_values, self.f1)

        # Vectorized FDR calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            self.global_fdr = (Ndecoy2 / Nreal2) * (g_auc_f0 / g_auc_f1)
            self.local_fdr = (Ndecoy2 / Nreal2) * (l_auc_f0 / l_auc_f1)

        # Handle division by zero (set to large value)
        self.global_fdr = np.where(np.isfinite(self.global_fdr), self.global_fdr, 1e10)
        self.local_fdr = np.where(np.isfinite(self.local_fdr), self.local_fdr, 1e10)

        # Debug output for first few PSMs
        for i in range(min(5, self.n_real)):
            logger.info(
                f"PSM {i}: delta_score={x_values[i]:.6f}, g_auc_f0={g_auc_f0[i]:.6f}, "
                f"g_auc_f1={g_auc_f1[i]:.6f}, global_fdr={self.global_fdr[i]:.6f}"
            )
            logger.info(
                f"PSM {i}: l_auc_f0={l_auc_f0[i]:.6f}, l_auc_f1={l_auc_f1[i]:.6f}, "
                f"local_fdr={self.local_fdr[i]:.6f}"
            )

        # Statistics of FDR value distribution
        global_fdr_vals = self.global_fdr[: min(10, len(self.global_fdr))]
        local_fdr_vals = self.local_fdr[: min(10, len(self.local_fdr))]
        logger.info(f"First 10 global FDR values: {global_fdr_vals}")
        logger.info(f"First 10 local FDR values: {local_fdr_vals}")

    def set_minor_maps(self) -> None:
        """Set minor maps"""
        # Global FDR mapping
        for i in range(self.n_real):
            self.minor_map_g[i] = [self.pos[i], self.global_fdr[i]]

        # Local FDR mapping
        for i in range(self.n_real):
            self.minor_map_l[i] = [self.pos[i], self.local_fdr[i]]

    def perform_minorization(self) -> None:
        """Perform minorization with validation"""
        for iter_type in ["global", "local"]:
            fdr_array = self.global_fdr if iter_type == "global" else self.local_fdr
            minor_map = self.minor_map_g if iter_type == "global" else self.minor_map_l

            n = len(minor_map)
            if n == 0:
                continue

            # Fill data (vectorized)
            pairs = [minor_map[i] for i in range(n)]
            x = np.array([p[0] for p in pairs])  # delta scores
            f = np.array([p[1] for p in pairs])  # FDR values
            is_minor_point = np.zeros(n, dtype=bool)
            
            # Find minimum value and its index (vectorized)
            min_idx = np.argmin(f)
            min_val = f[min_idx]

            # Calculate slope and apply
            slope = (0.0 - min_val) / (self.max_delta_score * 1.1 - x[min_idx])
            i = min_idx
            while i < n:
                f[i] = min_val + slope * (x[i] - x[min_idx])
                i += 1

            # Find maximum value and its index
            max_idx = 0
            max_val = f[0]
            i = 1
            while i < n and x[i] < (x[-1] / 2.0):
                if f[i] >= max_val:
                    max_val = f[i]
                    max_idx = i
                i += 1

            # Calculate slope for points before maximum
            slope = max_val / (x[max_idx] - x[-1])
            i = max_idx - 1
            while i >= 0:
                f[i] = max_val - slope * (x[max_idx] - x[i])
                i -= 1

            # Mark minor points (vectorized)
            is_minor_point[:max_idx] = True

            cur_start = max_idx
            cur_end = max_idx + 1

            while cur_start < n - 1 and cur_end < n:
                i = cur_start + 1
                slope = (f[cur_end] - f[cur_start]) / (x[cur_end] - x[cur_start])

                while i < cur_end:
                    f_expect = f[cur_start] + slope * (x[i] - x[cur_start])
                    if f[i] > f_expect:
                        f[i] = f_expect
                    i += 1

                cur_start = cur_end
                cur_end = cur_start + 1
                if cur_end >= n:
                    cur_end = n - 1
                while cur_end < n and not is_minor_point[cur_end]:
                    cur_end += 1

            # Map results back to FDR array using original indices
            # Since pairs were built from minor_map which uses sequential indices,
            # we can create a reverse mapping from sorted x back to original indices
            original_indices = np.argsort(np.argsort(self.pos[:n]))  # Maps sorted position to original
            for j in range(n):
                orig_idx = original_indices[j]
                fdr_array[orig_idx] = f[j]

            if iter_type == "global":
                self.global_fdr = fdr_array
            else:
                self.local_fdr = fdr_array

    def assign_fdrs(self, psms: List) -> None:
        """Assign FDR values to PSMs by finding closest delta_score in sorted mapping"""
        total_real_psms = sum(1 for psm in psms if not psm.is_decoy)
        logger.info(
            f"assign_fdrs: Total PSM count={len(psms)}, Real PSM count={total_real_psms}"
        )

        # Check if sorted mapping exists
        if not hasattr(self, 'sorted_delta_scores') or len(self.sorted_delta_scores) == 0:
            logger.warning("No sorted FLR mapping available, using default values")
            for psm in psms:
                if not psm.is_decoy:
                    psm.global_flr = 1.0
                    psm.local_flr = 1.0
                else:
                    psm.global_flr = float("nan")
                    psm.local_flr = float("nan")
            return

        # Assign FLR values to each PSM
        assigned_count = 0
        default_count = 0
        decoy_count = 0
        
        for psm in psms:
            if not psm.is_decoy:
                if (
                    hasattr(psm, "delta_score")
                    and not np.isnan(psm.delta_score)
                    and psm.delta_score > self.min_delta_score
                ):
                    psm_delta = float(psm.delta_score)
                    
                    # Use binary search to find the closest delta_score
                    # searchsorted finds the index where psm_delta would be inserted
                    idx = np.searchsorted(self.sorted_delta_scores, psm_delta)
                    
                    # Determine the closest index
                    if idx == 0:
                        # PSM delta_score is smaller than all values, use first
                        closest_idx = 0
                    elif idx >= len(self.sorted_delta_scores):
                        # PSM delta_score is larger than all values, use last
                        closest_idx = len(self.sorted_delta_scores) - 1
                    else:
                        # Check which is closer: idx-1 or idx
                        dist_left = abs(psm_delta - self.sorted_delta_scores[idx - 1])
                        dist_right = abs(psm_delta - self.sorted_delta_scores[idx])
                        closest_idx = (idx - 1) if dist_left <= dist_right else idx
                    
                    # Assign FLR values from the closest match
                    psm.global_flr = float(self.sorted_global_flr[closest_idx])
                    psm.local_flr = float(self.sorted_local_flr[closest_idx])
                    assigned_count += 1
                    
                    # Log if the match is not very close
                    delta_diff = abs(psm_delta - self.sorted_delta_scores[closest_idx])
                    if delta_diff > 1.0:
                        logger.debug(
                            f"PSM delta_score {psm_delta:.4f} matched to {self.sorted_delta_scores[closest_idx]:.4f} "
                            f"(diff={delta_diff:.4f})"
                        )
                else:
                    # For real PSMs with delta_score <= min_delta_score, set default values
                    psm.global_flr = 1.0
                    psm.local_flr = 1.0
                    default_count += 1
            else:
                # Set decoy PSMs to NaN
                psm.global_flr = float("nan")
                psm.local_flr = float("nan")
                decoy_count += 1
        
        logger.info(
            f"FLR assignment completed - Assigned: {assigned_count}, Default: {default_count}, Decoy: {decoy_count}"
        )
        
        # Validate monotonicity
        self._validate_flr_monotonicity(psms)

    def _validate_flr_monotonicity(self, psms: List) -> None:
        """Validate that FLR values are monotonically decreasing with increasing delta_score"""
        # Collect real PSMs with valid delta_score and FLR
        real_psms_data = []
        for psm in psms:
            if (not psm.is_decoy and 
                hasattr(psm, 'delta_score') and 
                not np.isnan(psm.delta_score) and
                hasattr(psm, 'global_flr') and
                not np.isnan(psm.global_flr) and
                psm.global_flr < 1.0):  # Exclude default values
                real_psms_data.append({
                    'delta_score': psm.delta_score,
                    'global_flr': psm.global_flr,
                    'local_flr': psm.local_flr
                })
        
        if len(real_psms_data) < 2:
            logger.debug("Insufficient PSMs for monotonicity validation")
            return
        
        # Sort by delta_score
        sorted_data = sorted(real_psms_data, key=lambda x: x['delta_score'])
        
        # Check monotonicity
        global_violations = 0
        local_violations = 0
        
        for i in range(len(sorted_data) - 1):
            curr = sorted_data[i]
            next_psm = sorted_data[i + 1]
            
            # Global FLR should decrease (or stay same) as delta_score increases
            if curr['global_flr'] < next_psm['global_flr']:
                global_violations += 1
                if global_violations <= 3:  # Log first 3 violations
                    logger.debug(
                        f"Global FLR violation: delta {curr['delta_score']:.4f} (FLR={curr['global_flr']:.4f}) "
                        f"-> {next_psm['delta_score']:.4f} (FLR={next_psm['global_flr']:.4f})"
                    )
            
            # Local FLR should decrease (or stay same) as delta_score increases
            if curr['local_flr'] < next_psm['local_flr']:
                local_violations += 1
                if local_violations <= 3:  # Log first 3 violations
                    logger.debug(
                        f"Local FLR violation: delta {curr['delta_score']:.4f} (FLR={curr['local_flr']:.4f}) "
                        f"-> {next_psm['delta_score']:.4f} (FLR={next_psm['local_flr']:.4f})"
                    )
        
        # Report results
        total_pairs = len(sorted_data) - 1
        if global_violations == 0 and local_violations == 0:
            logger.info(f"✓ FLR monotonicity validated: {total_pairs} pairs checked, no violations")
        else:
            logger.warning(
                f"⚠ FLR monotonicity violations detected: "
                f"Global={global_violations}/{total_pairs}, Local={local_violations}/{total_pairs}"
            )

    def calculate_flr(self, psms):
        """Calculate FLR

        Args:
            psms: List of PSM objects
        """
        # Use all collected PSM data
        real_count = len(self.real_psms)
        decoy_count = len(self.decoy_psms)
        logger.info(
            f"Starting FLR calculation, total PSM count - Real PSMs: {real_count}, Decoy PSMs: {decoy_count}"
        )

        if real_count < 2 or decoy_count < 2:
            logger.warning("Insufficient PSM count for FLR calculation")
            return

        # Prepare arrays
        self.prep_arrays()

        # Initialize tick marks
        self.initialize_tick_marks()

        # Evaluate tick marks
        self.eval_tick_marks(DECOY)
        self.eval_tick_marks(REAL)

        # Calculate FDR
        self.calc_both_fdrs()

        # Set minor maps
        self.set_minor_maps()

        # Perform minorization to ensure monotonicity
        self.perform_minorization()

        # Create sorted delta_score to FLR mapping for efficient lookup
        self._create_sorted_flr_mapping()

        # Assign FDR values to each PSM
        self.assign_fdrs(psms)

        logger.info("FLR calculation completed")

    def _create_sorted_flr_mapping(self) -> None:
        """
        Create a sorted mapping of delta_score to FLR values.
        This allows efficient lookup when assigning FLR to PSMs.
        """
        # Create list of (delta_score, global_flr, local_flr) tuples
        flr_data = []
        for i in range(len(self.pos)):
            if i < len(self.global_fdr) and i < len(self.local_fdr):
                flr_data.append({
                    'delta_score': float(self.pos[i]),
                    'global_flr': min(1.0, float(self.global_fdr[i])),
                    'local_flr': min(1.0, float(self.local_fdr[i]))
                })
        
        # Sort by delta_score
        flr_data.sort(key=lambda x: x['delta_score'])
        
        # Store sorted arrays for binary search
        self.sorted_delta_scores = np.array([d['delta_score'] for d in flr_data])
        self.sorted_global_flr = np.array([d['global_flr'] for d in flr_data])
        self.sorted_local_flr = np.array([d['local_flr'] for d in flr_data])
        
        if len(self.sorted_delta_scores) > 0:
            logger.info(
                f"Created sorted FLR mapping with {len(self.sorted_delta_scores)} entries, "
                f"delta_score range: [{self.sorted_delta_scores[0]:.4f}, {self.sorted_delta_scores[-1]:.4f}]"
            )
        else:
            logger.warning("Created empty sorted FLR mapping - no valid FLR data available")

    def calculate_flr_estimates(self, psms: List) -> None:
        """
        Calculate FLR estimates

        Args:
            psms: List of PSM objects
        """
        logger.info("Starting FLR estimate calculation")

        # Check if data has already been collected
        if len(self.real_psms) == 0 and len(self.decoy_psms) == 0:
            # If no data, collect delta score and decoy information from all PSMs
            self.max_delta_score = 0.0

            for psm in psms:
                if hasattr(psm, "delta_score") and not np.isnan(psm.delta_score):
                    if psm.delta_score > self.max_delta_score:
                        self.max_delta_score = psm.delta_score

                    if psm.delta_score > self.min_delta_score:
                        if psm.is_decoy:
                            self.decoy_psms.append(psm.delta_score)
                            self.n_decoy += 1
                        else:
                            self.real_psms.append(psm.delta_score)
                            self.n_real += 1

            logger.info(
                f"Collected {self.n_real} real PSMs and {self.n_decoy} decoy PSMs"
            )
        else:
            logger.info(
                f"Using already collected data - Real PSMs: {self.n_real}, Decoy PSMs: {self.n_decoy}"
            )

        # Calculate FLR
        if self.n_real > 0 and self.n_decoy > 0:
            self.calculate_flr(psms)
            logger.info("FLR estimate calculation completed")
        else:
            logger.warning(
                f"Insufficient PSM count, cannot calculate FLR estimates - Real PSMs: {self.n_real}, Decoy PSMs: {self.n_decoy}"
            )
            # Set default FLR values for all PSMs
            for psm in psms:
                if not psm.is_decoy:
                    psm.global_flr = 1.0
                    psm.local_flr = 1.0
                else:
                    psm.global_flr = float("nan")
                    psm.local_flr = float("nan")

    def record_flr_estimates(self, psms: List) -> None:
        """
        Record FLR estimates

        Args:
            psms: List of PSM objects
        """
        logger.info("Recording FLR estimates")

        # Create FLR estimate mapping
        self.flr_estimate_map = {}

        for psm in psms:
            if psm.is_decoy:
                continue  # Skip FLR data for decoy PSMs

            if hasattr(psm, "delta_score") and not np.isnan(psm.delta_score):
                # Store global and local FLR values
                flr_values = [psm.global_flr, psm.local_flr]
                self.flr_estimate_map[psm.delta_score] = flr_values

        logger.info(f"Recorded {len(self.flr_estimate_map)} FLR estimates")

    def assign_flr_to_psms(self, psms: List) -> None:
        """
        Assign FLR values to PSMs

        Args:
            psms: List of PSM objects
        """
        if not hasattr(self, "flr_estimate_map") or not self.flr_estimate_map:
            logger.warning("FLR estimate mapping is empty, cannot assign FLR values")
            return

        logger.info("Assigning FLR values to PSMs")

        # Get all observed delta scores and sort them
        observed_delta_scores = sorted(self.flr_estimate_map.keys())
        n = len(observed_delta_scores)

        if n == 0:
            logger.warning("No available delta scores for FLR assignment")
            return

        for psm in psms:
            obs_ds = psm.delta_score
            assigned = False

            # Iterate through delta scores, find the closest value
            for i in range(1, n):
                cur_ds = observed_delta_scores[i]
                if cur_ds > obs_ds:  # Found upper bound, use previous delta score
                    flr_values = self.flr_estimate_map[observed_delta_scores[i - 1]]
                    psm.global_flr = flr_values[0]
                    psm.local_flr = flr_values[1]
                    assigned = True
                    break

            if not assigned:  # High-scoring PSM, use FLR value of highest delta score
                flr_values = self.flr_estimate_map[observed_delta_scores[n - 1]]
                psm.global_flr = flr_values[0]
                psm.local_flr = flr_values[1]

        logger.info(f"Assigned FLR values to {len(psms)} PSMs")

    def save_delta_score_flr_mapping(self) -> None:
        """
        Save delta score to FLR mapping for second round calculation
        """
        try:
            if self.global_fdr is None or self.local_fdr is None or self.pos is None:
                logger.warning("FLR calculation not completed, cannot save mapping")
                return

            # Clear previous mapping
            self.delta_score_to_flr_map.clear()

            # Create delta score to FLR mapping
            for i in range(len(self.pos)):
                delta_score = self.pos[i]
                global_flr = (
                    min(1.0, self.global_fdr[i]) if i < len(self.global_fdr) else 1.0
                )
                local_flr = (
                    min(1.0, self.local_fdr[i]) if i < len(self.local_fdr) else 1.0
                )

                self.delta_score_to_flr_map[delta_score] = (global_flr, local_flr)

            logger.info(
                f"Saved {len(self.delta_score_to_flr_map)} delta score to FLR mappings"
            )

        except Exception as e:
            logger.error(f"Error saving delta score to FLR mapping: {str(e)}")

    def find_closest_flr(self, delta_score: float) -> Tuple[float, float]:
        """
        Find closest FLR value based on delta score

        Args:
            delta_score: Delta score to search for

        Returns:
            Tuple[float, float]: (global_flr, local_flr) Closest FLR values
        """
        try:
            if not self.delta_score_to_flr_map:
                logger.warning(
                    "Delta score to FLR mapping is empty, returning default values"
                )
                return (1.0, 1.0)

            # Find closest delta score
            closest_delta = min(
                self.delta_score_to_flr_map.keys(), key=lambda x: abs(x - delta_score)
            )

            global_flr, local_flr = self.delta_score_to_flr_map[closest_delta]

            logger.debug(
                f"Delta score {delta_score:.6f} closest to {closest_delta:.6f}, "
                f"corresponding FLR: global={global_flr:.6f}, local={local_flr:.6f}"
            )

            return (global_flr, local_flr)

        except Exception as e:
            logger.error(f"Error finding closest FLR value: {str(e)}")
            return (1.0, 1.0)

    def assign_flr_from_mapping(self, psms: List) -> None:
        """
        Assign FLR values to PSMs using saved mapping (for second round calculation)

        Args:
            psms: List of PSM objects
        """
        try:
            if not self.delta_score_to_flr_map:
                logger.warning(
                    "Delta score to FLR mapping is empty, cannot assign FLR values"
                )
                return

            assigned_count = 0
            for psm in psms:
                if (
                    not psm.is_decoy
                    and hasattr(psm, "delta_score")
                    and not np.isnan(psm.delta_score)
                ):
                    if psm.delta_score > self.min_delta_score:
                        global_flr, local_flr = self.find_closest_flr(psm.delta_score)
                        psm.global_flr = global_flr
                        psm.local_flr = local_flr
                        assigned_count += 1
                    else:
                        # For PSMs with delta_score <= min_delta_score, set default values
                        psm.global_flr = 1.0
                        psm.local_flr = 1.0
                else:
                    # Set decoy PSMs to NaN
                    psm.global_flr = float("nan")
                    psm.local_flr = float("nan")

            logger.info(
                f"Assigned FLR values to {assigned_count} real PSMs using mapping"
            )

        except Exception as e:
            logger.error(f"Error assigning FLR values using mapping: {str(e)}")
