"""
Spectrum module.

This module contains the Spectrum class for handling mass spectrometry data.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
import pyopenms

logger = logging.getLogger(__name__)

class Spectrum:
    """Class representing a mass spectrum."""
    
    def __init__(self, mz_array=None, intensity_array=None):
        """
        Initialize a new Spectrum instance.
        
        Args:
            mz_array: Array of m/z values
            intensity_array: Array of intensity values
        """
        self.N = 0  # number of peaks
        self.max_i_index = 0  # index of max intensity peak
        self.max_i = 0.0  # max intensity value
        
        # Peak arrays - use NumPy arrays for better performance
        self.mz = np.array([])  # m/z values
        self.raw_intensity = np.array([])  # raw intensity values
        self.rel_intensity = np.array([])  # relative intensity values (0-100)
        self.norm_intensity = np.array([])  # normalized intensity values (log scale)
        
        # Pre-computed indices for faster searching
        self._mz_sorted_indices = None
        self._mz_sorted = None
        
        if mz_array is not None and intensity_array is not None:
            self._init_peaks(mz_array, intensity_array)
    
    def _init_peaks(self, mz_array, intensity_array):
        """Initialize peak arrays from input data."""
        # Convert to NumPy arrays if not already
        mz_array = np.asarray(mz_array)
        intensity_array = np.asarray(intensity_array)
        
        # Ensure arrays have same length
        N = min(len(mz_array), len(intensity_array))
        
        # Filter out zero intensity peaks using vectorized operations
        valid_mask = intensity_array[:N] > 0
        
        if not np.any(valid_mask):
            # No valid peaks
            self.N = 0
            return
        
        # Extract valid peaks
        self.mz = mz_array[:N][valid_mask]
        self.raw_intensity = intensity_array[:N][valid_mask]
        self.N = len(self.mz)
        
        # Find max intensity
        max_idx = np.argmax(self.raw_intensity)
        self.max_i = self.raw_intensity[max_idx]
        self.max_i_index = max_idx
        
        # Initialize other arrays
        self.rel_intensity = np.zeros(self.N)
        self.norm_intensity = np.zeros(self.N)
        
        # Calculate relative intensities
        self.calc_relative_intensity()
        
        # Pre-compute sorted indices for faster searching
        self._update_sorted_indices()
    
    def calc_relative_intensity(self):
        """Calculate relative intensities (0-100) based on max intensity."""
        if self.max_i > 0:
            self.rel_intensity = (self.raw_intensity / self.max_i) * 100.0
    
    def median_normalize_spectra(self):
        """
        Normalize intensities using median.
        Computes log(rel_intensity/median_intensity).
        """
        if self.N == 0:
            return
            
        # Use NumPy median for better performance
        median_i = np.median(self.rel_intensity)
        
        # Vectorized calculation of normalized intensities
        with np.errstate(divide='ignore', invalid='ignore'):
            d = self.rel_intensity / median_i
            self.norm_intensity = np.where(d > 0, np.log(d), float('-inf'))
    
    def get_peaks(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get peak arrays.
        
        Returns:
            Tuple of (m/z array, intensity array)
        """
        return self.mz, self.raw_intensity
    
    def is_empty(self) -> bool:
        """
        Check if spectrum is empty.
        
        Returns:
            True if spectrum has no peaks
        """
        return self.N == 0
    
    def _update_sorted_indices(self):
        """Update sorted indices for faster searching."""
        if self.N > 0:
            self._mz_sorted_indices = np.argsort(self.mz)
            self._mz_sorted = self.mz[self._mz_sorted_indices]
    
    def find_index_by_mz(self, mz: float, tolerance: float = 1e-6) -> int:
        """
        Find peak index by m/z value using binary search for better performance.
        
        Args:
            mz: m/z value to find
            tolerance: m/z tolerance for matching
            
        Returns:
            Index of peak with matching m/z, or -1 if not found
        """
        if self.N == 0:
            return -1
        
        # Use binary search on sorted m/z values
        if self._mz_sorted is None:
            self._update_sorted_indices()
        
        # Find closest m/z using searchsorted
        idx = np.searchsorted(self._mz_sorted, mz)
        
        # Check both adjacent positions
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(self._mz_sorted):
            candidates.append(idx)
        
        # Find the closest match within tolerance
        best_idx = -1
        best_diff = float('inf')
        
        for candidate_idx in candidates:
            if candidate_idx < len(self._mz_sorted):
                diff = abs(self._mz_sorted[candidate_idx] - mz)
                if diff < tolerance and diff < best_diff:
                    best_diff = diff
                    best_idx = self._mz_sorted_indices[candidate_idx]
        
        return best_idx
    
    def find_peaks_in_range(self, mz_min: float, mz_max: float) -> np.ndarray:
        """
        Find all peak indices within a m/z range using binary search.
        
        Args:
            mz_min: Minimum m/z value
            mz_max: Maximum m/z value
            
        Returns:
            Array of indices of peaks within the range
        """
        if self.N == 0:
            return np.array([])
        
        if self._mz_sorted is None:
            self._update_sorted_indices()
        
        # Use searchsorted to find range boundaries
        start_idx = np.searchsorted(self._mz_sorted, mz_min, side='left')
        end_idx = np.searchsorted(self._mz_sorted, mz_max, side='right')
        
        # Return original indices
        return self._mz_sorted_indices[start_idx:end_idx] 