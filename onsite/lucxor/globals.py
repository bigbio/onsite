"""
Global variables module.
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from .constants import DECOY_AA_MAP

logger = logging.getLogger(__name__)

@dataclass
class globals:
    """Global variables class."""
    
    # PSM lists
    real_psms: List = None
    decoy_psms: List = None
    
    # FLR estimate mapping
    flr_estimate_map: Dict = None
    
    @classmethod
    def init_globals(cls, args):
        """Initialize global variables"""
        cls.real_psms = []
        cls.decoy_psms = []
        cls.flr_estimate_map = {}
        
        logger.debug("Global variables initialized")
        
    @classmethod
    def record_flr_estimates(cls):
        """Record FLR estimates for all delta scores"""
        cls.flr_estimate_map = {}
        
        for psm in cls.real_psms:
            if psm.is_decoy:
                continue  # Skip decoy FLR data
            cls.flr_estimate_map[psm.delta_score] = [psm.global_flr, psm.local_flr]
            
        logger.debug(f"Recorded {len(cls.flr_estimate_map)} FLR estimates")
        
    @classmethod
    def assign_flr(cls):
        """Assign FLR values based on delta score"""
        obs_delta_scores = sorted(cls.flr_estimate_map.keys())
        n = len(obs_delta_scores)
        
        for psm in cls.real_psms:
            obs_ds = psm.delta_score
            assigned = False
            
            # Iterate through delta scores until finding the closest value
            for i in range(1, n):
                cur_ds = obs_delta_scores[i]
                if cur_ds > obs_ds:  # Reached limit, get previous delta score
                    d = cls.flr_estimate_map[obs_delta_scores[i-1]]
                    psm.global_flr = d[0]
                    psm.local_flr = d[1]
                    assigned = True
                    break
                    
            if not assigned:  # Very high delta score
                d = cls.flr_estimate_map[obs_delta_scores[-1]]
                psm.global_flr = d[0]
                psm.local_flr = d[1]
                
        logger.debug("Assigned FLR values to all PSMs")
        
    @classmethod
    def clear_psms(cls):
        """Clear scores for all PSMs to prepare for second round"""
        for psm in cls.real_psms:
            psm.clear_scores()
        for psm in cls.decoy_psms:
            psm.clear_scores()
            
        logger.debug("Cleared scores for all PSMs")

def get_decoy_symbol(c: str) -> str:
    """
    Get decoy symbol for amino acid
    
    Args:
        c: Amino acid character
        
    Returns:
        str: Decoy symbol
    """
    ret = ""
    src_char = c.upper()
    
    for k, v in DECOY_AA_MAP.items():
        if v.upper() == src_char:
            ret = k
            break
            
    return ret 