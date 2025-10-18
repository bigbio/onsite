"""
Peak module.

This module contains the Peak class, which represents a mass spectrometry peak.
"""

from typing import Optional, Dict, Any
import numpy as np


class Peak:
    """
    Class representing a mass spectrometry peak.
    
    This class contains information about a mass spectrometry peak, including
    its m/z value, intensity, and whether it matches a theoretical fragment ion.
    Optimized with __slots__ for better memory efficiency and performance.
    """
    
    __slots__ = [
        'mz', 'raw_intensity', 'rel_intensity', 'norm_intensity',
        'matched', 'dist', 'matched_ion_str', 'matched_ion_mz',
        'score', 'intensity_score', 'dist_score', '_hash_cache'
    ]
    
    def __init__(self, mz: float, intensity: float, norm_intensity: Optional[float] = None):
        """
        Initialize a new Peak instance.
        
        Args:
            mz: m/z value
            intensity: Raw intensity
            norm_intensity: Normalized intensity (optional)
        """
        self.mz = float(mz)  # Ensure float type for consistency
        self.raw_intensity = float(intensity)
        self.rel_intensity = 0.0  # Relative intensity (percentage of max)
        self.norm_intensity = float(norm_intensity) if norm_intensity is not None else 0.0  # Normalized intensity (log(rel/median))
        
        # Matching information
        self.matched = False  # Whether this peak matches a theoretical fragment ion
        self.dist = 0.0  # Distance from theoretical m/z
        self.matched_ion_str = ""  # String representation of the matched ion
        self.matched_ion_mz = 0.0  # Theoretical m/z of the matched ion
        
        # Scoring information
        self.score = 0.0  # Score assigned to this peak
        self.intensity_score = 0.0  # Intensity component of the score
        self.dist_score = 0.0  # Distance component of the score
        
        # Cache for hash computation
        self._hash_cache = None
    
    def __eq__(self, other):
        """
        Check if two peaks are equal.
        
        Two peaks are considered equal if they have the same m/z value.
        
        Args:
            other: Another Peak object
            
        Returns:
            True if the peaks are equal, False otherwise
        """
        if not isinstance(other, Peak):
            return False
        
        return abs(self.mz - other.mz) < 1e-6
    
    def __hash__(self):
        """
        Get the hash value of the peak with caching for better performance.
        
        Returns:
            Hash value
        """
        if self._hash_cache is None:
            # Round to avoid floating point precision issues
            self._hash_cache = hash(round(self.mz, 6))
        return self._hash_cache
    
    def __str__(self):
        """
        Get a string representation of the peak.
        
        Returns:
            String representation
        """
        return f"Peak(mz={self.mz:.4f}, intensity={self.raw_intensity:.1f}, matched={self.matched})"
    
    def __repr__(self):
        """
        Get a string representation of the peak.
        
        Returns:
            String representation
        """
        return self.__str__()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the peak to a dictionary.
        
        Returns:
            Dictionary representation of the peak
        """
        return {
            "mz": self.mz,
            "raw_intensity": self.raw_intensity,
            "rel_intensity": self.rel_intensity,
            "norm_intensity": self.norm_intensity,
            "matched": self.matched,
            "dist": self.dist,
            "matched_ion_str": self.matched_ion_str,
            "matched_ion_mz": self.matched_ion_mz,
            "score": self.score,
            "intensity_score": self.intensity_score,
            "dist_score": self.dist_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Peak':
        """
        Create a peak from a dictionary.
        
        Args:
            data: Dictionary representation of the peak
            
        Returns:
            Peak object
        """
        peak = cls(data["mz"], data["raw_intensity"], data["norm_intensity"])
        peak.rel_intensity = data["rel_intensity"]
        peak.matched = data["matched"]
        peak.dist = data["dist"]
        peak.matched_ion_str = data["matched_ion_str"]
        peak.matched_ion_mz = data["matched_ion_mz"]
        peak.score = data["score"]
        peak.intensity_score = data["intensity_score"]
        peak.dist_score = data["dist_score"]
        
        return peak