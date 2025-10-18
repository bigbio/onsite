"""
Parallel processing module for lucxor.

This module contains classes and functions for parallel processing.
"""

import os
import logging
from typing import List, Any, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)

# Default thread count
DEFAULT_NUM_THREADS = 4

class ScoringWorker:
    """Worker thread class for parallel scoring"""
    
    def __init__(self, model):
        self.model = model
    
    def score_peptide(self, psm) -> None:
        """Score a single PSM"""
        try:
            psm.score_permutations(self.model)
        except Exception as e:
            logger.error(f"PSM scoring error: {str(e)}")
    
    def process_psms(self, psms: List) -> None:
        """Process multiple PSMs in parallel"""
        for psm in psms:
            self.score_peptide(psm)

class PSMProcessingWorker:
    """Worker thread class for PSM processing"""
    
    def __init__(self, model=None, flr_calculator=None, round_number=0):
        self.model = model
        self.flr_calculator = flr_calculator
        self.round_number = round_number
    
    def process_psm(self, psm) -> None:
        """Process a single PSM"""
        try:
            if self.round_number == 0:
                # Update PSM config with model
                psm.config['model'] = self.model
                psm.process(psm.config, round_number=0)
            elif self.round_number == 2:
                psm.process_round2(self.flr_calculator)
        except Exception as e:
            logger.error(f"PSM processing error: {str(e)}")
    
    def process_psm_batch(self, psms: List) -> None:
        """Process a batch of PSMs"""
        for psm in psms:
            self.process_psm(psm)

class NormalDensityWorker:
    """Worker thread class for normal density calculation"""
    
    def __init__(self, model):
        self.model = model
    
    def calculate_density(self, data) -> Any:
        """Calculate density for a single dataset"""
        return self.model.calculate_normal_density(data)
    
    def process_all(self, data_sets: List) -> List:
        """Process multiple datasets in parallel"""
        return [self.calculate_density(data) for data in data_sets]

class ModelParameterWorker:
    """Worker thread class for model parameter calculation"""
    
    def __init__(self, model):
        self.model = model
    
    def calculate_parameters(self, data: dict) -> dict:
        """Calculate parameters for a single dataset"""
        try:
            mean = self.model.calculate_mean(data)
            var = self.model.calculate_variance(data, mean)
            return {'mean': mean, 'var': var}
        except Exception as e:
            logging.error(f"Error calculating parameters: {str(e)}")
            return {'mean': 0.0, 'var': 1.0}
    
    def process_all(self, data_sets: List[dict]) -> List[dict]:
        """Process multiple datasets in parallel"""
        return [self.calculate_parameters(data) for data in data_sets]

class SpectrumMatchingWorker:
    """Worker thread class for spectrum matching"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def match_spectrum_peptide(self, psm) -> Dict:
        """Match a single spectrum and peptide"""
        try:
            if psm.spectrum is None or psm.peptide is None:
                return None
            
            # Get spectrum basic information
            spectrum_native_id = getattr(psm.spectrum, 'native_id', None)
            rt = getattr(psm.spectrum, 'rt', None)
            
            # Get matched peaks
            try:
                matched_peaks = psm.peptide.match_peaks(psm.spectrum)
            except Exception as e:
                matched_peaks = []
            
            match_info = {
                "peptide": getattr(psm.peptide, "peptide", None),
                "modified_peptide": getattr(psm.peptide, "mod_peptide", getattr(psm.peptide, "peptide", None)),
                "charge": getattr(psm, "charge", None),
                "scan_num": getattr(psm, "scan_num", None),
                "spectrum_native_id": getattr(psm, "spectrum_native_id", getattr(psm.spectrum, "native_id", None)),
                "rt": getattr(psm, "spectrum_rt", getattr(psm.spectrum, "rt", None)),
                "matched_peaks": matched_peaks
            }
            return match_info
        except Exception as e:
            logger.error(f"Spectrum matching error: {str(e)}")
            return None
    
    def process_psm_batch(self, psms: List) -> List[Dict]:
        """Process spectrum matching for a batch of PSMs"""
        results = []
        for psm in psms:
            result = self.match_spectrum_peptide(psm)
            if result is not None:
                results.append(result)
        return results

def parallel_process(data: List[Any], worker_class: Any, num_threads: Optional[int] = None) -> List[Any]:
    """
    Generic parallel processing function
    
    Args:
        data: List of data to process
        worker_class: Worker thread class
        num_threads: Number of threads, uses default if None
        
    Returns:
        List of processing results
    """
    if num_threads is None:
        num_threads = os.cpu_count() or DEFAULT_NUM_THREADS
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for item in data:
            future = executor.submit(worker_class.process_all, [item])
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.extend(result)
            except Exception as e:
                logging.error(f"Parallel processing error: {str(e)}")
                results.append(None)
    
    return results

def parallel_psm_processing(psms: List, model=None, flr_calculator=None, round_number=0, num_threads: Optional[int] = None) -> None:
    """
    Process PSM list in parallel
    
    Args:
        psms: PSM list
        model: Model object
        flr_calculator: FLR calculator
        round_number: Round number
        num_threads: Number of threads
    """
    if num_threads is None:
        num_threads = os.cpu_count() or DEFAULT_NUM_THREADS
    
    # Split PSM list into chunks
    chunk_size = max(1, len(psms) // num_threads)
    psm_chunks = [psms[i:i + chunk_size] for i in range(0, len(psms), chunk_size)]
    
    worker = PSMProcessingWorker(model, flr_calculator, round_number)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for chunk in psm_chunks:
            future = executor.submit(worker.process_psm_batch, chunk)
            futures.append(future)
        
        # Wait for all tasks to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"PSM parallel processing error: {str(e)}")

def parallel_spectrum_matching(psms: List, config: Dict, num_threads: Optional[int] = None) -> List[Dict]:
    """
    Perform spectrum matching in parallel
    
    Args:
        psms: PSM list
        config: Configuration dictionary
        num_threads: Number of threads
        
    Returns:
        List of matching results
    """
    if num_threads is None:
        num_threads = os.cpu_count() or DEFAULT_NUM_THREADS
    
    # Split PSM list into chunks
    chunk_size = max(1, len(psms) // num_threads)
    psm_chunks = [psms[i:i + chunk_size] for i in range(0, len(psms), chunk_size)]
    
    worker = SpectrumMatchingWorker(config)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for chunk in psm_chunks:
            future = executor.submit(worker.process_psm_batch, chunk)
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                chunk_results = future.result()
                results.extend(chunk_results)
            except Exception as e:
                logger.error(f"Spectrum matching parallel processing error: {str(e)}")
    
    return results

def get_optimal_thread_count(num_items: int, min_threads: int = 1, max_threads: int = None) -> int:
    """
    Calculate optimal thread count based on data size
    
    Args:
        num_items: Number of data items
        min_threads: Minimum number of threads
        max_threads: Maximum number of threads
        
    Returns:
        Optimal thread count
    """
    if max_threads is None:
        max_threads = os.cpu_count() or DEFAULT_NUM_THREADS
    
    # Use fewer threads for small datasets
    if num_items < 10:
        return min(2, max_threads)
    elif num_items < 100:
        return min(4, max_threads)
    else:
        return max_threads 