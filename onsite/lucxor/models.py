"""
Models module.

This module contains the CIDModel and HCDModel classes, which implement
the statistical models used for scoring peptide-spectrum matches.
"""

import logging
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from .constants import ALGORITHM_CID, ALGORITHM_HCD, TINY_NUM
from .peak import Peak

logger = logging.getLogger(__name__)


class ModelData_CID:
    """
    CID model data class
    """
    
    def __init__(self, charge_state: int, peaks: List[Dict]):
        """
        Initialize CID model data
        
        Args:
            charge_state: Charge state
            peaks: Peak list
        """
        self.charge_state = charge_state
        self.num_psm = 0  # Number of PSMs for this charge state
        
        # Intensity parameters
        self.mu_int_b = 0.0
        self.mu_int_y = 0.0
        self.mu_int_u = 0.0
        self.var_int_b = 0.0
        self.var_int_y = 0.0
        self.var_int_u = 0.0
        
        # Distance parameters
        self.mu_dist_b = 0.0
        self.mu_dist_y = 0.0
        self.mu_dist_u = 0.0
        self.var_dist_b = 0.0
        self.var_dist_y = 0.0
        self.var_dist_u = 0.0
        
        # Data arrays
        self.b_intensity = None
        self.b_distance = None
        self.y_intensity = None
        self.y_distance = None
        self.u_intensity = None
        self.u_distance = None
        
        # CID adjustment factor
        self.CID_ADJUST = 16.0 / 25.0
        
        # Initialize data arrays
        self._initialize_arrays(peaks)
    
    def _initialize_arrays(self, peaks: List[Peak]) -> None:
        """
        Initialize data arrays using vectorized operations for better performance
        
        Args:
            peaks: Peak list
        """
        # Convert peaks to structured arrays for efficient processing
        matched_peaks = []
        unmatched_peaks = []
        
        for peak in peaks:
            if peak.get('matched', False):
                matched_peaks.append({
                    'ion_str': peak.get('matched_ion_str', ''),
                    'norm_intensity': peak.get('norm_intensity', 0.0),
                    'mass_diff': peak.get('mass_diff', 0.0)
                })
            else:
                unmatched_peaks.append({
                    'norm_intensity': peak.get('norm_intensity', 0.0),
                    'mass_diff': peak.get('mass_diff', 0.0)
                })
        
        # Separate b and y ions using list comprehensions for better performance
        b_ions = [p for p in matched_peaks if p['ion_str'].startswith('b')]
        y_ions = [p for p in matched_peaks if p['ion_str'].startswith('y')]
        
        # Initialize arrays with pre-allocated sizes
        nb, ny = len(b_ions), len(y_ions)
        
        self.b_intensity = np.array([ion['norm_intensity'] for ion in b_ions]) if nb > 0 else np.array([])
        self.b_distance = np.array([ion['mass_diff'] for ion in b_ions]) if nb > 0 else np.array([])
        self.y_intensity = np.array([ion['norm_intensity'] for ion in y_ions]) if ny > 0 else np.array([])
        self.y_distance = np.array([ion['mass_diff'] for ion in y_ions]) if ny > 0 else np.array([])
        
        # Handle unmatched peaks more efficiently
        nu = len(unmatched_peaks)
        limit_n = min(nb + ny + 50000, nu) if nu > 0 else 0
        
        if limit_n > 0:
            # Randomly sample unmatched peaks
            if limit_n < nu:
                selected_indices = np.random.choice(nu, limit_n, replace=False)
                selected_unmatched = [unmatched_peaks[i] for i in selected_indices]
            else:
                selected_unmatched = unmatched_peaks
            
            self.u_intensity = np.array([p['norm_intensity'] for p in selected_unmatched])
            self.u_distance = np.array([p['mass_diff'] for p in selected_unmatched])
        else:
            self.u_intensity = np.array([])
            self.u_distance = np.array([])
    
    def calc_mean(self) -> None:
        """
        Calculate means
        """
        # Intensity means
        if len(self.b_intensity) > 0:
            self.mu_int_b = np.mean(self.b_intensity)
        else:
            self.mu_int_b = 0.0
            
        if len(self.y_intensity) > 0:
            self.mu_int_y = np.mean(self.y_intensity)
        else:
            self.mu_int_y = 0.0
            
        if len(self.u_intensity) > 0:
            self.mu_int_u = np.mean(self.u_intensity)
        else:
            self.mu_int_u = 0.0
        
        # Distance means
        if len(self.b_distance) > 0:
            self.mu_dist_b = np.mean(self.b_distance)
        else:
            self.mu_dist_b = 0.0
            
        if len(self.y_distance) > 0:
            self.mu_dist_y = np.mean(self.y_distance)
        else:
            self.mu_dist_y = 0.0
            
        if len(self.u_distance) > 0:
            self.mu_dist_u = 0.0
        else:
            self.mu_dist_u = 0.0
    
    def calc_var(self) -> None:
        """
        Calculate variances
        """
        # Intensity variances
        if len(self.b_intensity) > 1:
            self.var_int_b = np.var(self.b_intensity, ddof=1)
        else:
            self.var_int_b = 1.0
            
        if len(self.y_intensity) > 1:
            self.var_int_y = np.var(self.y_intensity, ddof=1)
        else:
            self.var_int_y = 1.0
            
        if len(self.u_intensity) > 1:
            self.var_int_u = np.var(self.u_intensity, ddof=1)
        else:
            self.var_int_u = 1.0
        
        # Distance variances
        if len(self.b_distance) > 1:
            self.var_dist_b = np.var(self.b_distance, ddof=1)
        else:
            self.var_dist_b = 1.0
            
        if len(self.y_distance) > 1:
            self.var_dist_y = np.var(self.y_distance, ddof=1)
        else:
            self.var_dist_y = 1.0
            
        if len(self.u_distance) > 1:
            self.var_dist_u = np.var(self.u_distance, ddof=1)
        else:
            self.var_dist_u = 1.0
        
        # Apply CID adjustment factor
        self.var_dist_b *= self.CID_ADJUST
        self.var_dist_y *= self.CID_ADJUST
    
    def print_summary_stats(self) -> None:
        """
        Print summary statistics
        """
        logger.info(f"+{self.charge_state}: {self.num_psm} PSMs for modeling.")
        logger.info("-----------------------------------------------------")
        
        logger.info(f"+{self.charge_state}\tb-ions Intensity (mu, sigma): ({self.mu_int_b:.4f}, {np.sqrt(self.var_int_b):.4f}) N = {len(self.b_intensity)}")
        logger.info(f"+{self.charge_state}\ty-ions Intensity (mu, sigma): ({self.mu_int_y:.4f}, {np.sqrt(self.var_int_y):.4f}) N = {len(self.y_intensity)}")
        logger.info(f"+{self.charge_state}\tNoise Intensity (mu, sigma): ({self.mu_int_u:.4f}, {np.sqrt(self.var_int_u):.4f}) N = {len(self.u_intensity)}")
        logger.info("")
        
        logger.info(f"+{self.charge_state}\tb-ions m/z Accuracy (mu, sigma): ({self.mu_dist_b:.4f}, {np.sqrt(self.var_dist_b):.4f}) N = {len(self.b_distance)}")
        logger.info(f"+{self.charge_state}\ty-ions m/z Accuracy (mu, sigma): ({self.mu_dist_y:.4f}, {np.sqrt(self.var_dist_y):.4f}) N = {len(self.y_distance)}")
        logger.info(f"+{self.charge_state}\tNoise Distance (mu, sigma): ({self.mu_dist_u:.4f}, {np.sqrt(self.var_dist_u):.4f}) N = {len(self.u_distance)}")
        logger.info("")
    
    def clear_arrays(self) -> None:
        """
        Clear arrays to free memory
        """
        self.b_distance = None
        self.b_intensity = None
        self.y_distance = None
        self.y_intensity = None
        self.u_distance = None
        self.u_intensity = None


class ModelData_HCD:
    """
    HCD model data class
    """
    
    def __init__(self, charge_state: int, peaks: List[Dict]):
        """
        Initialize HCD model data
        
        Args:
            charge_state: Charge state
            peaks: Peak list
        """
        self.NORMAL_CONSTANT = 1.0 / np.sqrt(2.0 * np.pi)
        self.charge_state = charge_state
        self.num_psm = 0
        self.ntick = 2000  # Number of ticks
        
        # Bandwidth parameters
        self.b_int_bw = 0.0
        self.y_int_bw = 0.0
        self.neg_int_bw = 0.0
        self.pos_dist_bw = 0.0
        
        # Statistical parameters
        self.b_int_mean = 0.0
        self.b_int_var = 0.0
        self.y_int_mean = 0.0
        self.y_int_var = 0.0
        self.neg_int_mean = 0.0
        self.neg_int_var = 0.0
        self.pos_dist_mean = 0.0
        self.pos_dist_var = 0.0
        self.neg_dist_mean = 0.0
        self.neg_dist_var = 0.0
        
        # Density estimation arrays
        self.b_tick_marks_int = None
        self.y_tick_marks_int = None
        self.neg_tick_marks_int = None
        self.pos_tick_marks_dist = None
        self.neg_tick_marks_dist = None
        
        self.f_int_b = None
        self.f_int_y = None
        self.f_int_neg = None
        self.f_dist = None
        
        # Data arrays
        self.b_int = None
        self.y_int = None
        self.n_int = None
        self.pos_dist = None
        
        # Initialize data
        self._initialize_arrays(peaks)
    
    def _initialize_arrays(self, peaks: List[Peak]) -> None:
        """
        Initialize data arrays using vectorized operations for better performance
        
        Args:
            peaks: Peak list
        """
        # Separate peaks into matched and unmatched more efficiently
        matched_peaks = []
        unmatched_peaks = []
        
        for peak in peaks:
            if peak.get('matched', False):
                matched_peaks.append({
                    'ion_str': peak.get('matched_ion_str', ''),
                    'norm_intensity': peak.get('norm_intensity', 0.0),
                    'mass_diff': peak.get('mass_diff', 0.0)
                })
            else:
                unmatched_peaks.append(peak.get('norm_intensity', 0.0))
        
        # Separate b and y ions using vectorized operations
        b_ions = [p for p in matched_peaks if p['ion_str'].startswith('b')]
        y_ions = [p for p in matched_peaks if p['ion_str'].startswith('y')]
        
        # Initialize arrays directly from lists
        self.b_int = np.array([ion['norm_intensity'] for ion in b_ions]) if b_ions else np.array([])
        self.y_int = np.array([ion['norm_intensity'] for ion in y_ions]) if y_ions else np.array([])
        self.pos_dist = np.array([ion['mass_diff'] for ion in matched_peaks]) if matched_peaks else np.array([])
        
        # Handle negative peaks more efficiently
        n_total = len(unmatched_peaks)
        limit_n = min(len(b_ions) + len(y_ions) + 50000, n_total) if n_total > 0 else 0
        
        if limit_n > 0:
            # Randomly sample negative peaks using NumPy
            if limit_n < n_total:
                selected_indices = np.random.choice(n_total, limit_n, replace=False)
                self.n_int = np.array([unmatched_peaks[i] for i in selected_indices])
            else:
                self.n_int = np.array(unmatched_peaks)
        else:
            self.n_int = np.array([])
    
    def percentile_trim(self, ion_type: str, data_type: int, percent_trim: float) -> None:
        """
        Percentile trimming
        
        Args:
            ion_type: Ion type
            data_type: Data type
            percent_trim: Trimming percentage
        """
        d = None
        
        if ion_type == 'b':
            if data_type == 0:
                d = self.b_int
        elif ion_type == 'y':
            if data_type == 0:
                d = self.y_int
        elif ion_type == 'n':
            d = self.n_int
        
        if d is None or len(d) == 0:
            return
        
        N = len(d)
        n = int(N * percent_trim * 0.5)
        
        # Get limits
        a = n
        b = N - 1 - n
        
        d_sorted = np.sort(d)
        f = d_sorted[a:b]
        
        if ion_type == 'b':
            if data_type == 0:
                self.b_int = f
        elif ion_type == 'y':
            if data_type == 0:
                self.y_int = f
        elif ion_type == 'n':
            self.n_int = f
    
    def calc_mean(self) -> None:
        """
        Calculate mean
        """
        # B ions
        if len(self.b_int) > 0:
            self.b_int_mean = np.mean(self.b_int)
        else:
            self.b_int_mean = 0.0
        
        # Y ions
        if len(self.y_int) > 0:
            self.y_int_mean = np.mean(self.y_int)
        else:
            self.y_int_mean = 0.0
        
        # Positive peak distance
        if len(self.pos_dist) > 0:
            self.pos_dist_mean = self._get_mode(self.pos_dist)
        else:
            self.pos_dist_mean = 0.0
        
        # Noise peaks
        if len(self.n_int) > 0:
            self.neg_int_mean = np.mean(self.n_int)
        else:
            self.neg_int_mean = 0.0
        
        # Negative distribution assumed to be uniform on 1 Dalton window
        self.neg_dist_mean = 0.0
    
    def calc_var(self) -> None:
        """
        Calculate variance
        """
        # B ions
        if len(self.b_int) > 1:
            self.b_int_var = np.var(self.b_int, ddof=1)
        else:
            self.b_int_var = 1.0
        
        # Y ions
        if len(self.y_int) > 1:
            self.y_int_var = np.var(self.y_int, ddof=1)
        else:
            self.y_int_var = 1.0
        
        # Positive peak distance
        if len(self.pos_dist) > 1:
            self.pos_dist_var = np.var(self.pos_dist, ddof=1)
        else:
            self.pos_dist_var = 1.0
        
        # Noise peaks
        if len(self.n_int) > 1:
            self.neg_int_var = np.var(self.n_int, ddof=1)
        else:
            self.neg_int_var = 1.0
    
    def _get_mode(self, ary: np.ndarray) -> float:
        """
        Calculate mode
        
        Args:
            ary: Data array
            
        Returns:
            Mode
        """
        mode = 0.0
        nbins = self.ntick
        bin_width = 0.0001
        limit = 0.1
        
        v = np.zeros(nbins)
        
        # Iterate through values
        for L in ary:
            for j in range(nbins - 1):
                a = -limit + (j * bin_width)
                b = a + bin_width
                
                if a <= L < b:
                    v[j] += 1.0
                    break
        
        # Find maximum value
        max_value = 0.0
        max_val_idx = 0
        for i in range(nbins):
            if v[i] > max_value:
                max_value = v[i]
                max_val_idx = i
        
        mode = -limit + (max_val_idx * bin_width) + (bin_width / 2)
        
        return mode
    
    def estimate_np_intensity(self, ion_type: str) -> None:
        """
        Estimate non-parametric density - intensity, using NumPy vectorized operations to optimize calculation speed
        
        Args:
            ion_type: Ion type ('b', 'y', 'n')
        """
        start_time = time.time()
        
        N = 0
        norm_ints = None
        tick_marks_int = None
        f_int = None
        min_i = 0.0
        max_i = 0.0
        variance = 0.0
        bw = 0.0
        
        if ion_type == 'b':
            N = len(self.b_int)
            norm_ints = self.b_int
            variance = self.b_int_var
            logger.info(f"+{self.charge_state}  Estimating NP Model for b-ion intensities (N={N})")
        elif ion_type == 'y':
            N = len(self.y_int)
            norm_ints = self.y_int
            variance = self.y_int_var
            logger.info(f"+{self.charge_state}  Estimating NP Model for y-ion intensities (N={N})")
        else:  # 'n'
            N = len(self.n_int)
            norm_ints = self.n_int
            variance = self.neg_int_var
            logger.info(f"+{self.charge_state}  Estimating NP Model for noise peak intensities (N={N})")
        
        if N == 0:
            return
        
        # Get range of normalized intensities
        norm_ints_sorted = np.sort(norm_ints)
        min_i = norm_ints_sorted[0]
        max_i = norm_ints_sorted[-1]
        
        # Add padding
        padding = 0.1
        if min_i < 0:
            min_i = min_i + (min_i * padding)
        else:
            min_i = min_i - (min_i * padding)
        
        if max_i < 0:
            max_i = max_i - (max_i * padding)
        else:
            max_i = max_i + (max_i * padding)
        
        # Build tick marks - vectorized
        tick_marks_int = np.linspace(min_i, max_i, self.ntick)
        
        # Calculate positive intensity bandwidth estimation
        sigma = np.sqrt(variance)
        bw = (1.06 * (sigma / np.power(float(N), 0.2))) * 0.5
        
        # Use NumPy vectorized operations to calculate kernel density estimation
        # Convert norm_ints to column vector, tick_marks_int to row vector
        norm_ints_array = np.array(norm_ints).reshape(-1, 1)  # (N, 1)
        tick_marks_array = tick_marks_int.reshape(1, -1)      # (1, ntick)
        
        # Calculate distance matrix between all point pairs (N, ntick)
        diff_matrix = tick_marks_array - norm_ints_array
        
        # Calculate Gaussian kernel (N, ntick)
        kernel_matrix = self.NORMAL_CONSTANT * np.exp(-0.5 * (diff_matrix / bw) ** 2)
        
        # Sum all data points to get density for each tick mark (ntick,)
        f_int = np.sum(kernel_matrix, axis=0) / (N * bw)
        
        # Handle minimum values
        f_int = np.maximum(f_int, TINY_NUM)
        
        # Store results
        if ion_type == 'b':
            self.b_tick_marks_int = tick_marks_int
            self.f_int_b = f_int
            self.b_int_bw = bw
        elif ion_type == 'y':
            self.y_tick_marks_int = tick_marks_int
            self.f_int_y = f_int
            self.y_int_bw = bw
        else:  # 'n'
            self.neg_tick_marks_int = tick_marks_int
            self.f_int_neg = f_int
            self.neg_int_bw = bw
        
        # Record performance information
        elapsed_time = time.time() - start_time
        logger.info(f"+{self.charge_state}  {ion_type}-ion NP estimation completed in {elapsed_time:.2f}s")
    
    def estimate_np_pos_dist(self) -> None:
        """
        Estimate non-parametric density - distance, using NumPy vectorized operations to optimize calculation speed
        """
        start_time = time.time()
        
        N = len(self.pos_dist)
        min_dist = 0.0
        max_dist = 0.0
        variance = self.pos_dist_var
        bw = 0.0
        
        logger.info(f"+{self.charge_state}  Estimating NP Model for b and y ions m/z distances (N={N})")
        
        if N == 0:
            return
        
        # Get range of m/z distance values
        pos_dist_sorted = np.sort(self.pos_dist)
        min_dist = pos_dist_sorted[0]
        max_dist = pos_dist_sorted[-1]
        
        # Add padding
        padding = 0.1
        if min_dist < 0:
            min_dist = min_dist + (min_dist * padding)
        else:
            min_dist = min_dist - (min_dist * padding)
        
        if max_dist < 0:
            max_dist = max_dist - (max_dist * padding)
        else:
            max_dist = max_dist + (max_dist * padding)
        
        # Build tick marks - vectorized
        self.pos_tick_marks_dist = np.linspace(min_dist, max_dist, self.ntick)
        
        # Calculate positive intensity bandwidth estimation
        sigma = np.sqrt(variance)
        bw = (1.06 * (sigma / np.power(float(N), 0.2))) * 0.1
        
        # Use NumPy vectorized operations to calculate kernel density estimation
        # Convert pos_dist to column vector, pos_tick_marks_dist to row vector
        pos_dist_array = np.array(self.pos_dist).reshape(-1, 1)  # (N, 1)
        tick_marks_array = self.pos_tick_marks_dist.reshape(1, -1)  # (1, ntick)
        
        # Calculate distance matrix between all point pairs (N, ntick)
        diff_matrix = tick_marks_array - pos_dist_array
        
        # Calculate Gaussian kernel (N, ntick)
        kernel_matrix = self.NORMAL_CONSTANT * np.exp(-0.5 * (diff_matrix / bw) ** 2)
        
        # Sum all data points to get density for each tick mark (ntick,)
        f_ary = np.sum(kernel_matrix, axis=0) / (N * bw)
        
        # Handle minimum values
        f_ary = np.maximum(f_ary, TINY_NUM)
        
        self.f_dist = f_ary
        self.pos_dist_bw = bw
        
        # Record performance information
        elapsed_time = time.time() - start_time
        logger.info(f"+{self.charge_state}  Distance NP estimation completed in {elapsed_time:.2f}s")
    
    def get_log_np_density_int(self, ion_type: str, x: float) -> float:
        """
        Get log non-parametric density of intensity using optimized lookup
        
        Args:
            ion_type: Ion type ('b', 'y', 'n')
            x: Intensity value
            
        Returns:
            Log density value
        """
        tick_marks_int = None
        f_int = None
        
        if ion_type == 'b':
            tick_marks_int = self.b_tick_marks_int
            f_int = self.f_int_b
        elif ion_type == 'y':
            tick_marks_int = self.y_tick_marks_int
            f_int = self.f_int_y
        elif ion_type == 'n':
            tick_marks_int = self.neg_tick_marks_int
            f_int = self.f_int_neg
        
        if tick_marks_int is None or f_int is None or len(tick_marks_int) == 0:
            return float('-inf')
        
        # Use binary search for faster lookup
        if x <= tick_marks_int[0]:
            sum_val = f_int[0]
        elif x >= tick_marks_int[-1]:
            sum_val = f_int[-1]
        else:
            # Use NumPy searchsorted for faster interval finding
            idx = np.searchsorted(tick_marks_int, x, side='right') - 1
            idx = max(0, min(idx, len(tick_marks_int) - 2))
            
            # Linear interpolation
            a, b = tick_marks_int[idx], tick_marks_int[idx + 1]
            if b > a:  # Avoid division by zero
                weight = (x - a) / (b - a)
                sum_val = f_int[idx] * (1 - weight) + f_int[idx + 1] * weight
            else:
                sum_val = f_int[idx]
        
        return np.log(sum_val) if sum_val > 0 else float('-inf')
    
    def get_log_np_density_dist_pos(self, x: float) -> float:
        """
        Get log non-parametric density of distance using optimized lookup
        
        Args:
            x: Distance value
            
        Returns:
            Log density value
        """
        if self.pos_tick_marks_dist is None or self.f_dist is None or len(self.pos_tick_marks_dist) == 0:
            return float('-inf')
        
        # Use binary search for faster lookup
        if x <= self.pos_tick_marks_dist[0]:
            sum_val = self.f_dist[0]
        elif x >= self.pos_tick_marks_dist[-1]:
            sum_val = self.f_dist[-1]
        else:
            # Use NumPy searchsorted for faster interval finding
            idx = np.searchsorted(self.pos_tick_marks_dist, x, side='right') - 1
            idx = max(0, min(idx, len(self.pos_tick_marks_dist) - 2))
            
            # Linear interpolation
            a, b = self.pos_tick_marks_dist[idx], self.pos_tick_marks_dist[idx + 1]
            if b > a:  # Avoid division by zero
                weight = (x - a) / (b - a)
                sum_val = self.f_dist[idx] * (1 - weight) + self.f_dist[idx + 1] * weight
            else:
                sum_val = self.f_dist[idx]
        
        return np.log(sum_val) if sum_val > 0 else float('-inf')
    
    def print_stats(self) -> None:
        """
        Print statistics
        """
        line = (f"Z = {self.charge_state}:  "
                f"b-ions Intensity: (mean, std): ({self.b_int_mean:.4f}, {np.sqrt(self.b_int_var):.4f}) N = {len(self.b_int)}\n"
                f"Z = {self.charge_state}:  "
                f"y-ions Intensity: (mean, std): ({self.y_int_mean:.4f}, {np.sqrt(self.y_int_var):.4f}) N = {len(self.y_int)}\n"
                f"Z = {self.charge_state}:  "
                f"Matched Peak Distance: (mean, std): ({self.pos_dist_mean:.4f}, {np.sqrt(self.pos_dist_var):.4f}) N = {len(self.pos_dist)}\n"
                f"Z = {self.charge_state}:  "
                f"Noise peak Intensity: (mean, std): ({self.neg_int_mean:.4f}, {np.sqrt(self.neg_int_var):.4f}) N = {len(self.n_int)}")
        
        logger.info(line)


class CIDModel:
    """
    CID model class, using ModelData_CID
    """
    
    def __init__(self, config: Dict):
        """
        Initialize CID model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.charge_models = {}  # Charge state -> ModelData_CID
    
    def build(self, psms: List) -> None:
        """
        Build CID model
        
        Args:
            psms: PSM list
        """
        if not psms or len(psms) < self.config.get('min_num_psms_model', 50):
            raise RuntimeError(f"Insufficient high-scoring PSMs to train CID model (need at least {self.config.get('min_num_psms_model', 50)}, actual {len(psms)})")
        
        # Group PSMs by charge state
        charge_psms = defaultdict(list)
        for psm in psms:
            charge_psms[psm.charge].append(psm)
        
        # Check if each charge state has sufficient PSMs for modeling
        min_psms_per_charge = self.config.get('min_num_psms_model', 50)
        bad_charges = set()
        
        logger.info("PSMs for modeling:")
        logger.info("------------------")
        for charge in sorted(charge_psms.keys()):
            n_psms = len(charge_psms[charge])
            logger.info(f"+{charge}: {n_psms} PSMs")
            if n_psms < min_psms_per_charge:
                bad_charges.add(charge)
                logger.warning(f"Charge state +{charge} has insufficient PSMs for modeling (need {min_psms_per_charge}, got {n_psms})")
        
        # Remove charge states with insufficient PSMs
        for bad_charge in bad_charges:
            del charge_psms[bad_charge]
        
        if not charge_psms:
            raise RuntimeError(f"No charge states have sufficient PSMs for modeling (minimum {min_psms_per_charge} PSMs per charge state required)")
        
        logger.info(f"Will build models for charge states: {sorted(charge_psms.keys())}")
        
        # Get thread count configuration
        num_threads = self.config.get('num_threads', os.cpu_count() or 4)
        logger.info(f"Using {num_threads} threads for CID model training...")
        
        # Parallel processing for each charge state
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for charge, charge_psm_list in charge_psms.items():
                future = executor.submit(self._build_charge_model, charge, charge_psm_list)
                futures.append(future)
            
            # Wait for all tasks to complete
            for future in as_completed(futures):
                try:
                    charge, model_data = future.result()
                    self.charge_models[charge] = model_data
                except Exception as e:
                    logger.error(f"CID model training error: {str(e)}")
    
    def _build_charge_model(self, charge: int, charge_psm_list: List) -> Tuple[int, ModelData_CID]:
        """
        Build CID model for single charge state
        
        Args:
            charge: Charge state
            charge_psm_list: PSM list for this charge state
            
        Returns:
            (charge, model_data) tuple
        """
        logger.info(f"Processing charge state {charge} with {len(charge_psm_list)} PSMs")
        logger.debug(f"Building CID model for charge {charge}")
        
        # Collect all peaks
        pos_peaks = []
        neg_peaks = []
        for psm in charge_psm_list:
            # Get matched peaks using modified peptide sequence
            matched_peaks = psm._match_peaks(psm.peptide.mod_peptide, psm.config.get('fragment_mass_tolerance', 0.5))
            pos_peaks.extend(matched_peaks)
            
            # Get unmatched peaks (get all peaks from spectrum, then exclude matched peaks)
            mz_values, intensities = psm.spectrum.get_peaks()
            matched_mz = {peak['mz'] for peak in matched_peaks}
            
            for i in range(len(mz_values)):
                if mz_values[i] not in matched_mz:
                    neg_peak = {
                        'mz': mz_values[i],
                        'intensity': intensities[i],
                        'norm_intensity': psm.spectrum.norm_intensity[i] if hasattr(psm.spectrum, 'norm_intensity') else intensities[i],
                        'mass_diff': 0.0,
                        'matched': False,
                        'matched_ion_str': 'n'
                    }
                    neg_peaks.append(neg_peak)
        
        model_data = ModelData_CID(charge, pos_peaks + neg_peaks)
        model_data.num_psm = len(charge_psm_list)
        
        model_data.calc_mean()
        model_data.calc_var()
        
        model_data.print_summary_stats()
        
        return charge, model_data
    
    def get_charge_model(self, charge: int) -> Optional[ModelData_CID]:
        """
        Get model for specified charge state
        
        Args:
            charge: Charge state
            
        Returns:
            ModelData_CID instance, returns None if not found
        """
        # First try to get the exact charge model
        if charge in self.charge_models:
            return self.charge_models[charge]
        
        # If exact charge model not available, try to use a fallback model
        # Prefer nearby charge states
        available_charges = sorted(self.charge_models.keys())
        if not available_charges:
            return None
        
        # Find the closest charge state
        closest_charge = min(available_charges, key=lambda x: abs(x - charge))
        logger.debug(f"Using charge {closest_charge} model as fallback for charge {charge}")
        return self.charge_models[closest_charge]
    
    def get_log_np_density_int(self, ion_type: str, x: float, charge: int = None) -> float:
        """
        Get log non-parametric density of intensity
        
        Args:
            ion_type: Ion type ('b', 'y', 'n')
            x: Intensity value
            charge: Charge state, uses default if None
            
        Returns:
            Log density value
        """
        if charge is None:
            available_charges = list(self.charge_models.keys())
            if not available_charges:
                return float('-inf')
            charge = available_charges[0]  
        
        charge_model = self.get_charge_model(charge)
        if charge_model is None:
            return float('-inf')
        
        if ion_type == 'n':
            mu = charge_model.mu_int_u
            var = charge_model.var_int_u
        elif ion_type == 'b':
            mu = charge_model.mu_int_b
            var = charge_model.var_int_b
        elif ion_type == 'y':
            mu = charge_model.mu_int_y
            var = charge_model.var_int_y
        else:
            return float('-inf')
        
        if var <= 0:
            return float('-inf')
        
        log_prob = -0.5 * np.log(2 * np.pi * var) - 0.5 * ((x - mu) ** 2) / var
        return log_prob
    
    def get_log_np_density_dist_pos(self, x: float, charge: int = None) -> float:
        """
        Get log non-parametric density of distance
        
        Args:
            x: Distance value
            charge: Charge state, uses default if None
            
        Returns:
            Log density value
        """
        if charge is None:
            available_charges = list(self.charge_models.keys())
            if not available_charges:
                return float('-inf')
            charge = available_charges[0] 
        
        charge_model = self.get_charge_model(charge)
        if charge_model is None:
            return float('-inf')
        
        mu = 0.0 
        var = charge_model.var_dist_b  
        
        if var <= 0:
            return float('-inf')
        
        log_prob = -0.5 * np.log(2 * np.pi * var) - 0.5 * (x ** 2) / var
        return log_prob


class HCDModel:
    """
    HCD model class, using ModelData_HCD
    """
    
    def __init__(self, config: Dict):
        """
        Initialize HCD model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.charge_models = {}  
        
        self.use_vectorization = config.get('use_vectorization', True) 
        self.enable_profiling = config.get('enable_profiling', True)  
    
    def build(self, psms: List) -> None:
        """
        Build HCD model
        
        Args:
            psms: PSM list
        """
        if not psms or len(psms) < self.config.get('min_num_psms_model', 50):
            raise RuntimeError(f"Insufficient high-scoring PSMs to train HCD model (need at least {self.config.get('min_num_psms_model', 50)}, actual {len(psms)})")
        
        charge_psms = defaultdict(list)
        for psm in psms:
            charge_psms[psm.charge].append(psm)
        
        # Check if each charge state has sufficient PSMs for modeling
        min_psms_per_charge = self.config.get('min_num_psms_model', 50)
        bad_charges = set()
        
        logger.info("PSMs for modeling:")
        logger.info("------------------")
        for charge in sorted(charge_psms.keys()):
            n_psms = len(charge_psms[charge])
            logger.info(f"+{charge}: {n_psms} PSMs")
            if n_psms < min_psms_per_charge:
                bad_charges.add(charge)
                logger.warning(f"Charge state +{charge} has insufficient PSMs for modeling (need {min_psms_per_charge}, got {n_psms})")
        
        # Remove charge states with insufficient PSMs
        for bad_charge in bad_charges:
            del charge_psms[bad_charge]
        
        if not charge_psms:
            raise RuntimeError(f"No charge states have sufficient PSMs for modeling (minimum {min_psms_per_charge} PSMs per charge state required)")
        
        logger.info(f"Will build models for charge states: {sorted(charge_psms.keys())}")
        
        num_threads = self.config.get('num_threads', os.cpu_count() or 4)
        logger.info(f"Using {num_threads} threads for HCD model training...")
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for charge, charge_psm_list in charge_psms.items():
                future = executor.submit(self._build_charge_model, charge, charge_psm_list)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    charge, model_data = future.result()
                    self.charge_models[charge] = model_data
                except Exception as e:
                    logger.error(f"HCD model training error: {str(e)}")
    
    def _build_charge_model(self, charge: int, charge_psm_list: List) -> Tuple[int, ModelData_HCD]:
        """
        Build HCD model for single charge state
        
        Args:
            charge: Charge state
            charge_psm_list: PSM list for this charge state
            
        Returns:
            (charge, model_data) tuple
        """
        logger.info(f"Processing charge state {charge} with {len(charge_psm_list)} PSMs")
        logger.debug(f"Building HCD model for charge {charge}")
        
        pos_peaks = []
        neg_peaks = []
        for psm in charge_psm_list:
            matched_peaks = psm._match_peaks(psm.peptide.peptide, psm.config.get('fragment_mass_tolerance', 0.5))
            pos_peaks.extend(matched_peaks)
            
            mz_values, intensities = psm.spectrum.get_peaks()
            matched_mz = {peak['mz'] for peak in matched_peaks}
            
            for i in range(len(mz_values)):
                if mz_values[i] not in matched_mz:
                    neg_peak = {
                        'mz': mz_values[i],
                        'intensity': intensities[i],
                        'norm_intensity': psm.spectrum.norm_intensity[i] if hasattr(psm.spectrum, 'norm_intensity') else intensities[i],
                        'mass_diff': 0.0,
                        'matched': False,
                        'matched_ion_str': 'n'
                    }
                    neg_peaks.append(neg_peak)
        
        model_data = ModelData_HCD(charge, pos_peaks + neg_peaks)
        model_data.num_psm = len(charge_psm_list)
        
        model_data.calc_mean()
        model_data.calc_var()
        
        model_data.estimate_np_intensity('b')
        model_data.estimate_np_intensity('y')
        model_data.estimate_np_intensity('n')
        model_data.estimate_np_pos_dist()
        
        model_data.print_stats()
        
        return charge, model_data
    
    def get_charge_model(self, charge: int) -> Optional[ModelData_HCD]:
        """
        Get model for specified charge state
        
        Args:
            charge: Charge state
            
        Returns:
            ModelData_HCD instance, returns None if not found
        """
        # First try to get the exact charge model
        if charge in self.charge_models:
            return self.charge_models[charge]
        
        # If exact charge model not available, try to use a fallback model
        # Prefer nearby charge states
        available_charges = sorted(self.charge_models.keys())
        if not available_charges:
            return None
        
        # Find the closest charge state
        closest_charge = min(available_charges, key=lambda x: abs(x - charge))
        logger.debug(f"Using charge {closest_charge} model as fallback for charge {charge}")
        return self.charge_models[closest_charge]
    
    def get_log_np_density_int(self, ion_type: str, x: float, charge: int = None) -> float:
        """
        Get log non-parametric density of intensity
        
        Args:
            ion_type: Ion type ('b', 'y', 'n')
            x: Intensity value
            charge: Charge state, uses default if None
            
        Returns:
            Log density value
        """
        if charge is None:
            available_charges = list(self.charge_models.keys())
            if not available_charges:
                return float('-inf')
            charge = available_charges[0]  
        
        charge_model = self.get_charge_model(charge)
        if charge_model is None:
            return float('-inf')
        
        return charge_model.get_log_np_density_int(ion_type, x)
    
    def get_log_np_density_dist_pos(self, x: float, charge: int = None) -> float:
        """
        Get log non-parametric density of distance
        
        Args:
            x: Distance value
            charge: Charge state, uses default if None
            
        Returns:
            Log density value
        """
        if charge is None:
            available_charges = list(self.charge_models.keys())
            if not available_charges:
                return float('-inf')
            charge = available_charges[0]  
        
        charge_model = self.get_charge_model(charge)
        if charge_model is None:
            return float('-inf')
        
        return charge_model.get_log_np_density_dist_pos(x)