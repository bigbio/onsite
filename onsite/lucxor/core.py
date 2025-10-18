"""
Core processing module for lucxor.

This module contains the main processing logic for scoring PSMs and calculating FLR.
"""

import logging
import numpy as np
import pyopenms
from typing import List, Dict, Any, Optional
from .psm import PSM
from .flr import FLRCalculator
from .config import LucXorConfig

logger = logging.getLogger(__name__)

class CoreProcessor:
    """Main processor for scoring PSMs and calculating FLR."""
    
    def __init__(self, config: LucXorConfig):
        """
        Initialize the core processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.psms = []
        self.flr_calculator = None
        
        # Performance optimization: pre-allocate lists with estimated capacity
        self._estimated_psm_count = config.get('estimated_psm_count', 10000)
        self.psms = []  # Will be converted to more efficient structure if needed
        
    def add_psm(self, psm: PSM) -> None:
        """
        Add a PSM to the processor.
        
        Args:
            psm: PSM object to add
        """
        self.psms.append(psm)
    
    def process_all_psms(self) -> None:
        """
        Process all PSMs using a two-stage workflow:
        Stage 1 (RN=0): Calculate FLR estimates
        Stage 2 (RN=1): Assign FLR values to PSMs
        """
        logger.info(f"Starting to process all PSMs, total PSM count: {len(self.psms)}")
        
        # Create FLR calculator
        self.flr_calculator = FLRCalculator(
            min_delta_score=self.config.get('min_delta_score', 0.1),
            min_psms_per_charge=self.config.get('min_psms_for_modeling', 50)
        )
        
        # Stage 1: Calculate FLR estimates (RN=0)
        logger.info("=== Stage 1: Calculate FLR estimates (RN=0) ===")
        self._process_psms_stage1()
        
        # Stage 2: Assign FLR values to PSMs (RN=1)
        logger.info("=== Stage 2: Assign FLR values to PSMs (RN=1) ===")
        self._process_psms_stage2()
        
        logger.info("All PSM processing completed")
    
    def _process_psms_stage1(self) -> None:
        """
        Stage 1 processing: Calculate FLR estimates
        """
        logger.info("Stage 1: Process all PSMs, generate real and decoy permutations")
        
        # Convert config once to avoid repeated calls
        config_dict = self.config.to_dict()
        
        # Process all PSMs, generate real and decoy permutations
        for i, psm in enumerate(self.psms):
            try:
                logger.debug(f"Stage 1 processing PSM {i+1}: {psm.peptide.peptide}")
                
                # Set FLR calculator
                psm.flr_calculator = self.flr_calculator
                
                # Process PSM (generate real and decoy permutations)
                psm.process(config_dict)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Stage 1 processed {i+1} PSMs")
                    
            except Exception as e:
                logger.error(f"Stage 1 processing PSM {i+1} error: {e}")
                continue
        
        # Check if FLR calculator is initialized
        if self.flr_calculator is None:
            logger.error("FLR calculator not initialized")
            return
        
        # Calculate FLR estimates
        logger.info("Stage 1: Calculate FLR estimates")
        self.flr_calculator.calculate_flr_estimates(self.psms)
        
        # Record FLR estimates
        logger.info("Stage 1: Record FLR estimates")
        self.flr_calculator.record_flr_estimates(self.psms)
        
        # Clear PSM scores
        logger.info("Stage 1: Clear PSM scores")
        self._clear_psm_scores()
        
        logger.info("Stage 1 processing completed")
    
    def _process_psms_stage2(self) -> None:
        """
        Stage 2 processing: Assign FLR values to PSMs
        """
        logger.info("Stage 2: Reprocess all PSMs, generate only real permutations")
        
        # Convert config once to avoid repeated calls
        config_dict = self.config.to_dict()
        
        # Reprocess all PSMs, generate only real permutations (no decoys)
        for i, psm in enumerate(self.psms):
            try:
                logger.debug(f"Stage 2 processing PSM {i+1}: {psm.peptide.peptide}")
                
                # Set FLR calculator
                psm.flr_calculator = self.flr_calculator
                
                # Process PSM (generate only real permutations, no decoys)
                psm.process_stage2(config_dict)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Stage 2 processed {i+1} PSMs")
                    
            except Exception as e:
                logger.error(f"Stage 2 processing PSM {i+1} error: {e}")
                continue
        
        # Assign FLR values to PSMs
        logger.info("Stage 2: Assign FLR values to PSMs")
        self.flr_calculator.assign_flr_to_psms(self.psms)
        
        logger.info("Stage 2 processing completed")
    
    def _clear_psm_scores(self) -> None:
        """
        Clear PSM scores
        """
        for psm in self.psms:
            psm.clear_scores()
    
    def get_results(self) -> List[PSM]:
        """
        Get processing results.
        
        Returns:
            List of processed PSMs
        """
        return self.psms

    def write_results(self, output_file: str, protein_ids=None, peptide_ids=None) -> None:
        """
        Write processing results to file.
        
        Args:
            output_file: Output file path
            protein_ids: Protein identification list (optional)
            peptide_ids: Peptide identification list (optional)
        """
        logger.info(f"Starting to write results to file: {output_file}")
        
        if not self.psms:
            raise ValueError("No PSM data available. Please call process_all_psms() method first.")
        
        file_extension = output_file.lower().split('.')[-1]
        
        if file_extension == 'idxml':
            # Update peptide identification data for all PSMs
            logger.info("Updating peptide identification data for PSMs")
            for psm in self.psms:
                psm.update_peptide_id()
            
            # If peptide_ids not provided, extract from PSMs
            if peptide_ids is None:
                peptide_ids = []
                for psm in self.psms:
                    if hasattr(psm, 'peptide_id'):
                        peptide_ids.append(psm.peptide_id)
            
            # If protein_ids not provided, create empty protein identification list
            if protein_ids is None:
                protein_ids = []
            
            # Write idXML file
            try:
                pyopenms.IdXMLFile().store(output_file, protein_ids, peptide_ids)
                logger.info(f"Results successfully written to: {output_file}")
            except Exception as e:
                logger.error(f"Error writing idXML file: {str(e)}")
                raise
                
        elif file_extension == 'csv':
            # Write CSV file
            with open(output_file, 'w', encoding='utf-8') as f:
                # Write header
                f.write("SpectrumID,Peptide,ModifiedPeptide,Score,DeltaScore,GlobalFLR,LocalFLR,IsDecoy\n")
                
                # Write PSM data
                for psm in self.psms:
                    f.write(f"{psm.scan_num},{psm.peptide.peptide},{psm.peptide.peptide},{psm.psm_score:.4f},{psm.delta_score:.4f},{psm.global_flr:.4f},{psm.local_flr:.4f},{1 if psm.is_decoy else 0}\n")
            
            logger.info(f"Results successfully written to: {output_file}")
            
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")