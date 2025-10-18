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
        self.delta_score_mu_pos = 0.0   # Target sequence delta score mean
        self.delta_score_mu_neg = 0.0   # Decoy sequence delta score mean
        self.bw_real = 0.0    # Target sequence bandwidth
        self.bw_decoy = 0.0   # Decoy sequence bandwidth
        
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
            
        logger.debug(f"Current real_psms count: {self.n_real}, decoy_psms count: {self.n_decoy}")
    
    def prep_arrays(self) -> None:
        """Prepare arrays"""
        self.pos = np.array(self.real_psms)
        self.neg = np.array(self.decoy_psms)
        self.n_real = len(self.pos)
        self.n_decoy = len(self.neg)
    
    def initialize_tick_marks(self) -> None:
        """Initialize tick marks"""
        # First prepare arrays
        self.prep_arrays()
        
        self.max_delta_score *= 1.001  # Need to be slightly larger for binning
        
        self.tick_marks = np.zeros(self.NMARKS)
        for i in range(self.NMARKS):
            x = (i * self.max_delta_score) / self.NMARKS
            self.tick_marks[i] = x
        
        # Initialize other arrays
        self.local_fdr = np.zeros(self.n_real)
        self.global_fdr = np.zeros(self.n_real)
        
        self.calc_delta_score_mean()
        self.calc_delta_score_var()
        
        self.get_bandwidth(DECOY)  # Decoy
        self.get_bandwidth(REAL)   # Target
        
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
        """Calculate delta score mean"""
        # Target sequence
        sum_pos = 0.0
        for d in self.pos:
            sum_pos += d
        N_pos = float(self.n_real)
        self.delta_score_mu_pos = sum_pos / N_pos if N_pos > 0 else 0.0
        
        # Decoy sequence
        sum_neg = 0.0
        for d in self.neg:
            sum_neg += d
        N_neg = float(self.n_decoy)
        self.delta_score_mu_neg = sum_neg / N_neg if N_neg > 0 else 0.0
    
    def calc_delta_score_var(self) -> None:
        """Calculate delta score variance"""
        # Target sequence
        v = 0.0
        for d in self.pos:
            x = d - self.delta_score_mu_pos
            v += x * x
        N = float(self.n_real - 1)
        self.delta_score_var_pos = v / N if N > 0 else 0.0
        
        # Decoy sequence
        v = 0.0
        for d in self.neg:
            x = d - self.delta_score_mu_neg
            v += x * x
        N = float(self.n_decoy - 1)
        self.delta_score_var_neg = v / N if N > 0 else 0.0
    
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
        Evaluate tick marks, kernel density estimation implementation
        Args:
            data_type: Data type (REAL or DECOY)
        """
        if data_type == DECOY:
            data_ary = self.neg
            bw = self.bw_decoy
            N = float(self.n_decoy)
            self.f0 = np.zeros(self.NMARKS)
        else:  # REAL
            data_ary = self.pos
            bw = self.bw_real
            N = float(self.n_real)
            self.f1 = np.zeros(self.NMARKS)

        if N == 0 or bw == 0:
            # Avoid division by zero
            if data_type == DECOY:
                self.f0 = np.full(self.NMARKS, TINY_NUM)
            else:
                self.f1 = np.full(self.NMARKS, TINY_NUM)
            return

        # Kernel density estimation implementation
        NORMAL_CONSTANT = 1.0 / np.sqrt(2.0 * np.pi)
        
        for i in range(self.NMARKS):
            tic = self.tick_marks[i]
            kernel_result = 0.0
            
            # Loop to calculate kernel density
            for score in data_ary:
                x = (tic - score) / bw
                kernel_result += NORMAL_CONSTANT * np.exp(-0.5 * x * x)
            
            # Normalize
            kernel_result /= (N * bw)
            
            if kernel_result > TINY_NUM:
                if data_type == DECOY:
                    self.f0[i] = kernel_result
                else:
                    self.f1[i] = kernel_result
            else:
                if data_type == DECOY:
                    self.f0[i] = TINY_NUM
                else:
                    self.f1[i] = TINY_NUM
        
        # Add debug information
        if data_type == DECOY:
            logger.info(f"DECOY KDE calculation completed - data points: {len(data_ary)}, bandwidth: {bw:.6f}")
            logger.info(f"DECOY density range: min={np.min(self.f0):.6e}, max={np.max(self.f0):.6e}")
        else:
            logger.info(f"REAL KDE calculation completed - data points: {len(data_ary)}, bandwidth: {bw:.6f}")
            logger.info(f"REAL density range: min={np.min(self.f1):.6e}, max={np.max(self.f1):.6e}")
    
    def get_local_auc(self, x: float, which_f: int) -> float:
        """
        Calculate local density value (density at point x)
        
        Args:
            x: Score
            which_f: Which distribution to use (f0 or f1)
            
        Returns:
            Density value
        """
        result = 0.0
        a = 0.0
        b = 0.0
        tmp1 = 0.0
        tmp2 = 0.0
        fx = 0.0
        
        start_tick = self.tick_marks[0]
        end_tick = self.tick_marks[self.NMARKS-1]
        start_val = 0.0
        end_val = 0.0
        
        # For f0 (decoy)
        if which_f == DECOY:
            start_val = self.f0[0]
            end_val = self.f0[self.NMARKS-1]
            
            # Find interval containing x from back to front
            for j in range(self.NMARKS-1, 0, -1):
                i = j - 1
                a = self.tick_marks[i]
                b = self.tick_marks[j]
                
                if x >= a:
                    # Found interval containing x, calculate density value at x (linear interpolation)
                    tmp1 = (b-x) / (b-a)
                    tmp2 = (x-a) / (b-a)
                    fx = (tmp1 * self.f0[i]) + (tmp2 * self.f0[j])
                    result = fx
                    break
            
            if x <= start_tick:
                result = start_val
            elif x >= end_tick:
                result = end_val
            
        # For f1 (real)
        elif which_f == REAL:
            start_val = self.f1[0]
            end_val = self.f1[self.NMARKS-1]
            
            # Find interval containing x from back to front
            for j in range(self.NMARKS-1, 0, -1):
                i = j - 1
                a = self.tick_marks[i]
                b = self.tick_marks[j]
                
                if x >= a:
                    # Found interval containing x, calculate density value at x (linear interpolation)
                    tmp1 = (b-x) / (b-a)
                    tmp2 = (x-a) / (b-a)
                    fx = (tmp1 * self.f1[i]) + (tmp2 * self.f1[j])
                    result = fx
                    break
            
            if x <= start_tick:
                result = start_val
            elif x >= end_tick:
                result = end_val
        
        return result
    
    def get_global_auc(self, x: float, which_f: int) -> float:
        """
        Calculate global AUC
        
        Args:
            x: Score
            which_f: Which distribution to use (f0 or f1)
            
        Returns:
            AUC value
        """
        a = 0.0
        b = 0.0
        sum_val = 0.0
        tmp1 = 0.0
        tmp2 = 0.0
        fx = 0.0
        
        # For f0 (decoy)
        if which_f == DECOY:
            sum_val = 0.0
            # Find interval containing x from back to front
            for j in range(self.NMARKS-1, 0, -1):
                i = j - 1
                a = self.tick_marks[i]
                b = self.tick_marks[j]
                
                if x < a:
                    sum_val += (b - a) * (0.5 * (self.f0[j] + self.f0[i]))
                else:
                    # Found interval containing x, calculate area under curve up to x
                    tmp1 = (b-x) / (b-a)
                    tmp2 = (x-a) / (b-a)
                    fx = (tmp1 * self.f0[i]) + (tmp2 * self.f0[j])
                    sum_val += (b-x) * (0.5 * (fx + self.f0[j]))
                    break
        
        # For f1 (real)
        elif which_f == REAL:
            sum_val = 0.0
            # Find interval containing x from back to front
            for j in range(self.NMARKS-1, 0, -1):
                i = j - 1
                a = self.tick_marks[i]
                b = self.tick_marks[j]
                
                if x < a:
                    sum_val += (b - a) * (0.5 * (self.f1[j] + self.f1[i]))
                else:
                    # Found interval containing x, calculate area under curve up to x
                    tmp1 = (b-x) / (b-a)
                    tmp2 = (x-a) / (b-a)
                    fx = (tmp1 * self.f1[i]) + (tmp2 * self.f1[j])
                    sum_val += (b-x) * (0.5 * (fx + self.f1[j]))
                    break
        
        return sum_val
    
    def calc_both_fdrs(self) -> None:
        """Calculate global and local FDR"""
        Nreal2 = float(self.n_real)
        Ndecoy2 = float(self.n_decoy)
        
        logger.info(f"FDR calculation - Real PSM count: {Nreal2}, Decoy PSM count: {Ndecoy2}")
        
        for i in range(self.n_real):
            x = self.pos[i]
            if x < 0.1:
                x = 0.1
            
            # Calculate global FDR
            g_auc_f0 = self.get_global_auc(x, DECOY)
            g_auc_f1 = self.get_global_auc(x, REAL)
            
            ratio = (Ndecoy2/Nreal2) * (g_auc_f0 / g_auc_f1)
            self.global_fdr[i] = ratio
            
            # Calculate local FDR
            l_auc_f0 = self.get_local_auc(x, DECOY)
            l_auc_f1 = self.get_local_auc(x, REAL)
            
            ratio = (Ndecoy2/Nreal2) * (l_auc_f0 / l_auc_f1)
            self.local_fdr[i] = ratio
            
            # Add debug information (only for first few PSMs)
            if i < 5:
                logger.info(f"PSM {i}: delta_score={x:.6f}, g_auc_f0={g_auc_f0:.6f}, g_auc_f1={g_auc_f1:.6f}, global_fdr={self.global_fdr[i]:.6f}")
                logger.info(f"PSM {i}: l_auc_f0={l_auc_f0:.6f}, l_auc_f1={l_auc_f1:.6f}, local_fdr={self.local_fdr[i]:.6f}")
        
        # Statistics of FDR value distribution
        global_fdr_vals = self.global_fdr[:min(10, len(self.global_fdr))]
        local_fdr_vals = self.local_fdr[:min(10, len(self.local_fdr))]
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
        """Perform minorization"""
        for iter_type in ['global', 'local']:
            fdr_array = self.global_fdr if iter_type == 'global' else self.local_fdr
            minor_map = self.minor_map_g if iter_type == 'global' else self.minor_map_l
            
            n = len(minor_map)
            if n == 0:
                continue
                
            # Initialize arrays
            x = np.zeros(n)
            f = np.zeros(n)
            fcopy = np.zeros(n)
            forig = np.zeros(n)
            is_minor_point = np.zeros(n, dtype=bool)
            
            # Fill data
            for i in range(n):
                pair = minor_map[i]
                x[i] = pair[0]  # delta score
                f[i] = pair[1]  # FDR
                forig[i] = f[i]
                fcopy[i] = 0
                is_minor_point[i] = False
                
            # Find minimum value and its index
            min_idx = 0
            min_val = f[0]
            for i in range(1, n):
                if f[i] < min_val:
                    min_val = f[i]
                    min_idx = i
            
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
            while i < n and x[i] < (x[-1]/2.0):
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
            
            # Mark minor points
            for i in range(max_idx):
                is_minor_point[i] = True
            
            cur_start = max_idx
            cur_end = max_idx + 1
            
            while cur_start < n-1 and cur_end < n:
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
                    cur_end = n-1
                while cur_end < n and not is_minor_point[cur_end]:
                    cur_end += 1
            
            # Map results back to FDR array
            for i in range(n):
                for j in range(n):
                    if self.pos[i] == x[j]:
                        fdr_array[i] = f[j]
                        break
            
            if iter_type == 'global':
                self.global_fdr = fdr_array
            else:
                self.local_fdr = fdr_array
    
    def assign_fdrs(self, psms: List) -> None:
        """Assign FDR values to PSMs"""
        real_psm_index = 0
        total_real_psms = sum(1 for psm in psms if not psm.is_decoy)
        logger.info(f"assign_fdrs: Total PSM count={len(psms)}, Real PSM count={total_real_psms}, FDR array size={len(self.global_fdr)}")
        
        for psm in psms:
            if not psm.is_decoy:  # Only assign FDR values to real PSMs
                # Only process PSMs with delta_score > min_delta_score
                if hasattr(psm, 'delta_score') and not np.isnan(psm.delta_score) and psm.delta_score > self.min_delta_score:
                    # Check if index is out of bounds
                    if real_psm_index >= len(self.global_fdr):
                        logger.error(f"FDR array index out of bounds: real_psm_index={real_psm_index}, global_fdr_size={len(self.global_fdr)}")
                        # Set default values for remaining PSMs
                        psm.global_flr = 1.0
                        psm.local_flr = 1.0
                        real_psm_index += 1  # Still need to increment index
                        continue
                    
                    # Apply min(1.0, ...) limit
                    original_g_fdr = self.global_fdr[real_psm_index]
                    original_l_fdr = self.local_fdr[real_psm_index]
                    g_fdr = min(1.0, original_g_fdr)
                    l_fdr = min(1.0, original_l_fdr)
                    
                    # Add debug information
                    if original_g_fdr > 1.0 or original_l_fdr > 1.0:
                        logger.warning(f"PSM {real_psm_index}: Original FDR value > 1.0 - global_fdr={original_g_fdr:.6f}, local_fdr={original_l_fdr:.6f}")
                    
                    psm.global_flr = g_fdr
                    psm.local_flr = l_fdr
                    real_psm_index += 1
                else:
                    # For real PSMs with delta_score <= min_delta_score, set default values
                    psm.global_flr = 1.0
                    psm.local_flr = 1.0
            else:
                # Set decoy PSMs to NaN
                psm.global_flr = float('nan')
                psm.local_flr = float('nan')
    
    def calculate_flr(self, psms):
        """Calculate FLR
        
        Args:
            psms: List of PSM objects
        """
        # Use all collected PSM data
        real_count = len(self.real_psms)
        decoy_count = len(self.decoy_psms)
        logger.info(f"Starting FLR calculation, total PSM count - Real PSMs: {real_count}, Decoy PSMs: {decoy_count}")
        
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
        
        # Perform minorization
        self.perform_minorization()
        
        # Assign FDR values to each PSM
        self.assign_fdrs(psms)
        
        logger.info("FLR calculation completed") 

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
                if hasattr(psm, 'delta_score') and not np.isnan(psm.delta_score):
                    if psm.delta_score > self.max_delta_score:
                        self.max_delta_score = psm.delta_score
                    
                    if psm.delta_score > self.min_delta_score:
                        if psm.is_decoy:
                            self.decoy_psms.append(psm.delta_score)
                            self.n_decoy += 1
                        else:
                            self.real_psms.append(psm.delta_score)
                            self.n_real += 1
            
            logger.info(f"Collected {self.n_real} real PSMs and {self.n_decoy} decoy PSMs")
        else:
            logger.info(f"Using already collected data - Real PSMs: {self.n_real}, Decoy PSMs: {self.n_decoy}")
        
        # Calculate FLR
        if self.n_real > 0 and self.n_decoy > 0:
            self.calculate_flr(psms)
            logger.info("FLR estimate calculation completed")
        else:
            logger.warning(f"Insufficient PSM count, cannot calculate FLR estimates - Real PSMs: {self.n_real}, Decoy PSMs: {self.n_decoy}")
            # Set default FLR values for all PSMs
            for psm in psms:
                if not psm.is_decoy:
                    psm.global_flr = 1.0
                    psm.local_flr = 1.0
                else:
                    psm.global_flr = float('nan')
                    psm.local_flr = float('nan')
    
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
            
            if hasattr(psm, 'delta_score') and not np.isnan(psm.delta_score):
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
        if not hasattr(self, 'flr_estimate_map') or not self.flr_estimate_map:
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
                    flr_values = self.flr_estimate_map[observed_delta_scores[i-1]]
                    psm.global_flr = flr_values[0]
                    psm.local_flr = flr_values[1]
                    assigned = True
                    break
            
            if not assigned:  # High-scoring PSM, use FLR value of highest delta score
                flr_values = self.flr_estimate_map[observed_delta_scores[n-1]]
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
                global_flr = min(1.0, self.global_fdr[i]) if i < len(self.global_fdr) else 1.0
                local_flr = min(1.0, self.local_fdr[i]) if i < len(self.local_fdr) else 1.0
                
                self.delta_score_to_flr_map[delta_score] = (global_flr, local_flr)
            
            logger.info(f"Saved {len(self.delta_score_to_flr_map)} delta score to FLR mappings")
            
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
                logger.warning("Delta score to FLR mapping is empty, returning default values")
                return (1.0, 1.0)
            
            # Find closest delta score
            closest_delta = min(self.delta_score_to_flr_map.keys(), 
                              key=lambda x: abs(x - delta_score))
            
            global_flr, local_flr = self.delta_score_to_flr_map[closest_delta]
            
            logger.debug(f"Delta score {delta_score:.6f} closest to {closest_delta:.6f}, "
                        f"corresponding FLR: global={global_flr:.6f}, local={local_flr:.6f}")
            
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
                logger.warning("Delta score to FLR mapping is empty, cannot assign FLR values")
                return
            
            assigned_count = 0
            for psm in psms:
                if not psm.is_decoy and hasattr(psm, 'delta_score') and not np.isnan(psm.delta_score):
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
                    psm.global_flr = float('nan')
                    psm.local_flr = float('nan')
            
            logger.info(f"Assigned FLR values to {assigned_count} real PSMs using mapping")
            
        except Exception as e:
            logger.error(f"Error assigning FLR values using mapping: {str(e)}")