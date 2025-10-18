#!/usr/bin/env python3
"""
Command line interface for pyLuciPHOr2
"""

import argparse
import os
import sys
import logging
import time
import json
import pyopenms
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
from pyopenms import (
    IdXMLFile,
    MzMLFile,
    MSExperiment,
    PeptideIdentification, 
    ProteinIdentification,
    IDFilter
)

from .psm import PSM
from .peptide import Peptide
from .models import CIDModel, HCDModel
from .constants import (
    NTERM_MOD,
    CTERM_MOD,
    AA_MASSES,
    DEFAULT_CONFIG
)
from .spectrum import Spectrum
from .flr import FLRCalculator
from .parallel import parallel_psm_processing, get_optimal_thread_count

logger = logging.getLogger(__name__)

class PyLuciPHOr2:
    """Main class for pyLuciPHOr2 command line tool"""
    
    def __init__(self):
        """Initialize PyLuciPHOr2."""
        self.logger = logging.getLogger(__name__)
        # Parse arguments first
        self.args = self.parse_args()
        # Then setup logging
        self.setup_logging()
        # Initialize model
        self.model = None
        self.psms = []
        self.config = {}
        
    def setup_logging(self):
        """Setup logging configuration"""
        # Configure log format
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG if self.args.debug else logging.INFO)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Configure console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if self.args.debug else logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Only configure file handler in debug mode
        if self.args.debug:
            # Get output filename (without extension)
            output_base = os.path.splitext(self.args.output)[0]
            log_file = self.args.log_file or f"{output_base}_debug.log"
            
            # Configure file handler
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Set third-party library log levels
        logging.getLogger('numpy').setLevel(logging.WARNING)
        logging.getLogger('scipy').setLevel(logging.WARNING)
        
        self.logger = logging.getLogger(__name__)
        
    def parse_args(self) -> argparse.Namespace:
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description="Modification site localisation using pyLuciPHOr2"
        )
        
        # Required arguments
        parser.add_argument(
            "-in", "--input_spectrum",
            required=True,
            help="Input spectrum file (mzML)"
        )
        parser.add_argument(
            "-id", "--input_id",
            required=True,
            help="Input identification file (idXML)"
        )
        parser.add_argument(
            "-out", "--output",
            required=True,
            help="Output file (idXML)"
        )
        
        # Optional arguments
        parser.add_argument(
            "--fragment_method",
            choices=["CID", "HCD"],
            default="CID",
            help="Fragmentation method"
        )
        parser.add_argument(
            "--fragment_mass_tolerance",
            type=float,
            default=0.5,
            help="Tolerance of the peaks in the fragment spectrum"
        )
        parser.add_argument(
            "--fragment_error_units",
            choices=["Da", "ppm"],
            default="Da",
            help="Unit of fragment mass tolerance"
        )
        parser.add_argument(
            "--min_mz",
            type=float,
            default=150.0,
            help="Do not consider peaks below this value"
        )
        parser.add_argument(
            "--target_modifications",
            nargs="+",
            default=["Phospho (S)", "Phospho (T)", "Phospho (Y)"],
            help="List of target modifications (e.g., Phospho(S),Phospho(T),Phospho(Y),PhosphoDecoy(A))"
        )
        parser.add_argument(
            "--neutral_losses",
            nargs="+",
            default=["sty -H3PO4 -97.97690"],
            help="List of neutral losses"
        )
        parser.add_argument(
            "--decoy_mass",
            type=float,
            default=79.966331,
            help="Mass to add for decoy generation"
        )
        parser.add_argument(
            "--decoy_neutral_losses",
            nargs="+",
            default=["X -H3PO4 -97.97690"],
            help="List of decoy neutral losses"
        )
        parser.add_argument(
            "--max_charge_state",
            type=int,
            default=5,
            help="Maximum charge state to consider"
        )
        parser.add_argument(
            "--max_peptide_length",
            type=int,
            default=40,
            help="Maximum peptide length"
        )
        parser.add_argument(
            "--max_num_perm",
            type=int,
            default=16384,
            help="Maximum number of permutations"
        )
        parser.add_argument(
            "--modeling_score_threshold",
            type=float,
            default=0.95,
            help="Minimum score for modeling"
        )
        parser.add_argument(
            "--scoring_threshold",
            type=float,
            default=0.0,
            help="Minimum score threshold"
        )
        parser.add_argument(
            "--min_num_psms_model",
            type=int,
            default=50,
            help="Minimum number of PSMs for modeling"
        )
        parser.add_argument(
            "--threads",
            type=int,
            default=4,
            help="Number of threads to use (default: 4)"
        )
        parser.add_argument(
            "--rt_tolerance",
            type=float,
            default=0.01,
            help="Retention time tolerance"
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode"
        )
        parser.add_argument(
            "--log_file",
            type=str,
            help="Log file path (only used in debug mode, default: {output_base}_debug.log)"
        )
        
        return parser.parse_args()
        
    def load_input_files(self, args: argparse.Namespace) -> Tuple[List[PeptideIdentification], List[ProteinIdentification], MSExperiment]:
        """Load input files"""
        # Load identifications
        pep_ids = []
        prot_ids = []
        IdXMLFile().load(args.input_id, prot_ids, pep_ids)
        
        if not pep_ids:
            self.logger.warning("No peptide identifications found in input file")
            return [], [], None
            
        # Keep only best hits
        IDFilter().keepNBestHits(pep_ids, 1)
        
        # Load spectra
        exp = MSExperiment()
        MzMLFile().load(args.input_spectrum, exp)
        
        if exp.empty():
            self.logger.warning("No spectra found in input file")
            return [], [], None
            
        return pep_ids, prot_ids, exp
        
    def initialize_model(self, config: Dict) -> None:
        """Initialize scoring model"""
        fragment_method = config.get('fragment_method', 'HCD')
        
        if fragment_method == 'HCD':
            self.model = HCDModel(config)
            self.logger.info("HCD Model initialized")
        elif fragment_method == 'CID':
            self.model = CIDModel(config)
            self.logger.info("CID Model initialized")
        else:
            raise ValueError(f"Unsupported fragment method: {fragment_method}")
        
    def process_peptide(self, pep_id: PeptideIdentification, spectrum, config: Dict) -> Optional[PeptideIdentification]:
        """Process a single peptide identification"""
        # Get hit
        hit = pep_id.getHits()[0]
        
        # Get sequence and charge
        sequence = hit.getSequence().toString()
        charge = hit.getCharge()
        
        # Check for phosphorylation
        has_phospho = "(Phospho)" in sequence
        
        self.logger.info(f"Processing peptide - sequence: {sequence}, charge: {charge}, has phosphorylation: {has_phospho}")
        
        if not has_phospho:
            self.logger.info(f"Peptide {sequence} has no phosphorylation sites, skipping processing")
            # Create new peptide identification result
            new_pep_id = PeptideIdentification(pep_id)
            new_pep_id.setScoreType("Luciphor_delta_score")
            new_pep_id.setHigherScoreBetter(True)
            
            # Update hit score and metadata
            hit.setScore(-1.0)
            hit.setMetaValue("search_engine_sequence", sequence)
            hit.setMetaValue("Luciphor_pep_score", 0.0)
            hit.setMetaValue("Luciphor_global_flr", 0.0)
            hit.setMetaValue("Luciphor_local_flr", 0.0)
            
            new_pep_id.setHits([hit])
            new_pep_id.assignRanks()
            return new_pep_id
        
        # Create peptide object
        peptide = Peptide(sequence, {}, config, charge=charge)
        
        # Build ion ladders
        self.logger.info("Building ion ladders...")
        peptide.build_ion_ladders()
        
        # Create PSM object with spectrum
        self.logger.info("Creating PSM object...")
        psm = PSM(peptide, spectrum, config)
        
        # Initialize FLR calculator
        psm.flr_calculator = FLRCalculator()
        
        # Ensure config contains model
        if self.model is None:
            self.logger.warning("Model not initialized, skipping processing")
            return None
        config['model'] = self.model
        
        # Process PSM with config
        self.logger.info("Processing PSM...")
        psm.process(config)
        
        self.logger.debug(f"PSM processing result - Score: {psm.score}, Delta Score: {psm.delta_score}, Global FLR: {psm.global_flr}, Local FLR: {psm.local_flr}")
        
        # Update hit with results
        hit.setMetaValue("Luciphor_pep_score", psm.psm_score)
        hit.setMetaValue("Luciphor_global_flr", psm.global_flr)
        hit.setMetaValue("Luciphor_local_flr", psm.local_flr)
        hit.setMetaValue("search_engine_sequence", sequence)
        hit.setScore(psm.delta_score)
        
        # Create new peptide identification
        new_pep_id = PeptideIdentification(pep_id)
        new_pep_id.setScoreType("Luciphor_delta_score")
        new_pep_id.setHigherScoreBetter(True)
        new_pep_id.setHits([hit])
        new_pep_id.assignRanks()
        
        return new_pep_id
        
    def run(self):
        """
        LuciPHOr2 main workflow:
        1. Read input files and collect all PSMs.
        2. Train CID/HCD model with high-scoring PSMs.
        3. Score all PSMs with the trained model.
        4. Exit with error if insufficient high-scoring PSMs.
        """
        args = self.args  # Use already parsed arguments
        config = DEFAULT_CONFIG.copy()
        # Parse target_modifications to handle comma-separated format
        target_modifications = []
        for mod in args.target_modifications:
            if ',' in mod:
                # Split comma-separated modifications
                target_modifications.extend([m.strip() for m in mod.split(',')])
            else:
                target_modifications.append(mod.strip())
        
        config.update({
            "fragment_method": args.fragment_method,
            "fragment_mass_tolerance": args.fragment_mass_tolerance,
            "fragment_error_units": args.fragment_error_units,
            "min_mz": args.min_mz,
            "target_modifications": target_modifications,
            "neutral_losses": args.neutral_losses,
            "decoy_mass": args.decoy_mass,
            "decoy_neutral_losses": args.decoy_neutral_losses,
            "max_charge_state": args.max_charge_state,
            "max_peptide_length": args.max_peptide_length,
            "max_num_perm": args.max_num_perm,
            "modeling_score_threshold": args.modeling_score_threshold,
            "scoring_threshold": args.scoring_threshold,
            "min_num_psms_model": args.min_num_psms_model,
            "num_threads": args.threads,
            "rt_tolerance": args.rt_tolerance
        })
        
        # Start timing
        start_time = time.time()

        self.logger.info("Loading input files...")
        self.logger.debug(f"Debug mode: {args.debug}")
        self.logger.debug(f"Log level: {logging.getLogger().level}")
        pep_ids, prot_ids, exp = self.load_input_files(args)
        if not pep_ids or exp is None:
            self.logger.error("No valid peptide identification or spectrum data found, process terminated.")
            return

        # 1. Create scan number to spectrum mapping
        spectrum_map = {}
        for spectrum in exp:
            try:
                # Extract scan number from native ID
                native_id = spectrum.getNativeID()
                if 'scan=' in native_id:
                    scan_num = int(native_id.split('scan=')[-1])
                    spectrum_map[scan_num] = spectrum
                else:
                    # Try to extract scan number from other formats
                    for part in native_id.split():
                        if part.isdigit():
                            scan_num = int(part)
                            spectrum_map[scan_num] = spectrum
                            break
            except Exception as e:
                self.logger.warning(f"Cannot extract scan number from native ID: {native_id}, error: {str(e)}")

        # 2. Collect all PSM objects
        all_psms = []
        for i, pep_id in enumerate(pep_ids, 1):
            hit = pep_id.getHits()[0]
            sequence = hit.getSequence().toString()
            charge = hit.getCharge()
            rt = pep_id.getRT()
            
            # Try to get scan number
            spectrum = None
            scan_num = None
            
            # Get scan number from peptide identification
            if pep_id.metaValueExists('scan_number'):
                scan_num = pep_id.getMetaValue('scan_number')
            elif pep_id.metaValueExists('spectrum_reference'):
                spec_ref = pep_id.getMetaValue('spectrum_reference')
                if 'scan=' in spec_ref:
                    scan_num = int(spec_ref.split('scan=')[-1])
            
            # First try to find by scan number
            if scan_num is not None and scan_num in spectrum_map:
                spectrum = spectrum_map[scan_num]
                self.logger.debug(f"Found matching spectrum by scan number {scan_num}")
            else:
                # If scan number unavailable or no match found, try RT matching
                for spec in exp:
                    if abs(spec.getRT() - rt) <= args.rt_tolerance:
                        spectrum = spec
                        self.logger.debug(f"Found matching spectrum by RT {rt}")
                        break
            
            if spectrum:
                spectrum_dict = {
                    'mz': spectrum.get_peaks()[0],
                    'intensities': spectrum.get_peaks()[1],
                    'native_id': spectrum.getNativeID(),
                    'rt': spectrum.getRT()
                }
                peptide = Peptide(sequence, config, charge=charge)
                psm = PSM(peptide, spectrum_dict, config=config)
                
                # Set search_engine_sequence to the original sequence
                psm.search_engine_sequence = sequence
                
                # Automatically determine score type and assign values
                score_type = pep_id.getScoreType() if hasattr(pep_id, 'getScoreType') else None
                score = hit.getScore()
                higher_score_better = True
                if hasattr(pep_id, 'isHigherScoreBetter'):
                    higher_score_better = pep_id.isHigherScoreBetter()
                elif hasattr(pep_id, 'getHigherScoreBetter'):
                    higher_score_better = pep_id.getHigherScoreBetter()
                
                # If it's a PEP score (lower is better), convert to 1-PEP
                if (score_type and 'posterior error probability' in score_type.lower()) or (higher_score_better is False):
                    psm.psm_score = 1.0 - score
                    peptide.score = 1.0 - score
                else:
                    psm.psm_score = score
                    peptide.score = score
                all_psms.append(psm)
            else:
                self.logger.warning(f"No matching spectrum found - RT: {rt}, Scan: {scan_num if scan_num else 'N/A'}")

        if not all_psms:
            self.logger.error("No PSMs collected, process terminated.")
            return

        # 3. Train model with high-scoring PSMs
        modeling_score_threshold = config.get('modeling_score_threshold', 0.95)
        
        # First filter PSMs with modification sites
        phospho_psms = []
        for psm in all_psms:
            if hasattr(psm, 'peptide') and psm.peptide is not None:
                # Check if there are potential modification sites and reported modification sites
                num_pps = getattr(psm.peptide, 'num_pps', 0)
                num_rps = getattr(psm.peptide, 'num_rps', 0)
                
                # Must have potential modification sites and reported modification sites
                if num_pps > 0 and num_rps > 0:
                    phospho_psms.append(psm)
        
        # Then filter high-scoring PSMs from PSMs with modification sites
        high_score_psms = [psm for psm in phospho_psms if hasattr(psm, 'psm_score') and psm.psm_score >= modeling_score_threshold]
        if not high_score_psms:
            high_score_psms = [psm for psm in phospho_psms if hasattr(psm, 'peptide') and hasattr(psm.peptide, 'score') and psm.peptide.score >= modeling_score_threshold]
        
        self.logger.info(f"Total PSMs: {len(all_psms)}")
        self.logger.info(f"PSMs with modification sites: {len(phospho_psms)}")
        self.logger.info(f"High-scoring PSMs for training: {len(high_score_psms)}")
        
        if not high_score_psms or len(high_score_psms) < config.get('min_num_psms_model', 50):
            self.logger.error(f"Insufficient high-scoring PSMs for model training (need at least {config.get('min_num_psms_model', 50)}, actual {len(high_score_psms)}), process terminated.")
            raise RuntimeError("Not enough high-scoring PSMs for model training.")
        
        # Group statistics by charge state
        charge_stats = defaultdict(list)
        for psm in high_score_psms:
            charge = getattr(psm, 'charge', 0)
            charge_stats[charge].append(psm)
        
        # === End of additions ===

        # 4. Initialize and train model
        fragment_method = config.get('fragment_method', 'HCD')
        if fragment_method == 'HCD':
            self.model = HCDModel(config)
            self.logger.info("Training HCD model...")
        elif fragment_method == 'CID':
            self.model = CIDModel(config)
            self.logger.info("Training CID model...")
        else:
            raise ValueError(f"Unsupported fragment method: {fragment_method}")
        self.model.build(high_score_psms)
        self.logger.info("Model training completed.")

        # 5. Score all PSMs with trained model
        self.logger.info("Starting first round calculation (including decoy permutations)...")
        
        # Use multi-threading for PSM processing
        num_threads = get_optimal_thread_count(len(all_psms), max_threads=args.threads)
        self.logger.info(f"Using {num_threads} threads for PSM processing...")
        
        parallel_psm_processing(all_psms, model=self.model, round_number=0, num_threads=num_threads)

        # === Round 1: Execute FLR calculation and establish mapping relationships ===
        self.logger.info("Starting first round FLR calculation...")
        
        # Create a global FLR calculator
        global_flr_calculator = FLRCalculator(
            min_delta_score=config.get('min_delta_score', 0.1),
            min_psms_per_charge=config.get('min_num_psms_model', 50)
        )
        
        # Collect delta score data for all PSMs (first round results)
        for psm in all_psms:
            if hasattr(psm, 'delta_score') and not np.isnan(psm.delta_score):
                if psm.delta_score > global_flr_calculator.min_delta_score:
                    global_flr_calculator.add_psm(psm.delta_score, psm.is_decoy)
        
        self.logger.info(f"First round collected {global_flr_calculator.n_real} real PSMs and {global_flr_calculator.n_decoy} decoy PSMs")
        
        # Execute first round FLR calculation
        if global_flr_calculator.n_real > 0 and global_flr_calculator.n_decoy > 0:
            global_flr_calculator.calculate_flr_estimates(all_psms)
            global_flr_calculator.record_flr_estimates(all_psms)
            global_flr_calculator.assign_flr_to_psms(all_psms)
            
            # Save delta score to FLR mapping relationship
            global_flr_calculator.save_delta_score_flr_mapping()
            

            
            self.logger.info("First round FLR calculation completed")
        else:
            self.logger.warning("Insufficient PSMs in first round, cannot calculate FLR")
        
        # === Round 2: Recalculate delta score excluding decoys ===
        self.logger.info("Starting second round calculation (real permutations only)...")
        
        # Use multi-threading for second round PSM calculation
        parallel_psm_processing(all_psms, flr_calculator=global_flr_calculator, round_number=2, num_threads=num_threads)
        
        self.logger.info("Second round calculation completed")
        


        # 6. Write results to output file (using second round calculation results)
        new_pep_ids = []
        phospho_count = 0
        for psm in all_psms:
            idx = all_psms.index(psm)
            if idx < len(pep_ids):
                orig_pep_id = pep_ids[idx]
                hit = orig_pep_id.getHits()[0]
                
                # Use second round calculated delta score and FLR values
                hit.setScore(psm.delta_score)  # Second round calculated delta score
                
                # Set search_engine_sequence to the original sequence from the peptide
                if hasattr(psm, 'search_engine_sequence'):
                    hit.setMetaValue("search_engine_sequence", psm.search_engine_sequence)
                else:
                    # Fallback to the peptide sequence if search_engine_sequence is not available
                    hit.setMetaValue("search_engine_sequence", psm.peptide.peptide)
                
                hit.setMetaValue("Luciphor_pep_score", psm.psm_score)
                hit.setMetaValue("Luciphor_global_flr", psm.global_flr)  # Second round assigned FLR value
                hit.setMetaValue("Luciphor_local_flr", psm.local_flr)    # Second round assigned FLR value
                
                # Update the sequence to the best scoring sequence from permutations
                best_sequence = psm.get_best_sequence(include_decoys=False)  # Use second round (real permutations only)
                if best_sequence != psm.peptide.peptide:
                    # Create new AASequence with the best sequence
                    from pyopenms import AASequence
                    try:
                        best_aa_sequence = AASequence.fromString(best_sequence)
                        hit.setSequence(best_aa_sequence)
                        logger.debug(f"Updated sequence for PSM {psm.scan_num}: {psm.peptide.peptide} -> {best_sequence}")
                    except Exception as e:
                        logger.debug(f"Failed to create AASequence for {best_sequence}: {str(e)}, keeping original sequence")
                        # Keep original sequence if conversion fails
                        pass
                
                new_pep_id = PeptideIdentification(orig_pep_id)
                new_pep_id.setScoreType("Luciphor_delta_score")
                new_pep_id.setHigherScoreBetter(True)
                new_pep_id.setHits([hit])
                new_pep_id.assignRanks()
                new_pep_ids.append(new_pep_id)

                # Count phosphorylated peptides
                try:
                    seq_str = hit.getSequence().toString()
                    if "(Phospho)" in seq_str:
                        phospho_count += 1
                except Exception:
                    pass

        # 7. Save results
        IdXMLFile().store(args.output, prot_ids, new_pep_ids)
        self.logger.info(f"Results saved to: {args.output}")

        # 8. Processing completed - print run summary similar to Ascore
        elapsed = time.time() - start_time
        total = len(pep_ids)
        processed = len(new_pep_ids)
        errors = max(0, total - processed)

        print("\nProcessing Complete:")
        print(f"  Total identifications: {total}")
        print(f"  Successfully processed: {processed}")
        print(f"  Phosphorylated peptides: {phospho_count}")
        print(f"  Processing errors: {errors}")
        print(f"  Time elapsed: {elapsed:.2f} seconds")
        if elapsed > 0:
            print(f"  Processing speed: {processed/elapsed:.2f} IDs/second")

        if self.args.debug:
            self.logger.info("Processing completed successfully")
            self.logger.info({
                'total': total,
                'processed': processed,
                'phospho': phospho_count,
                'errors': errors,
                'elapsed_sec': round(elapsed, 2),
                'speed_ids_per_sec': round(processed/elapsed, 2) if elapsed > 0 else None
            })

        return 0
        


    def read_spectrum_data(self, spectrum_file: str) -> Dict[int, Spectrum]:
        """Read spectrum data.
        
        Args:
            spectrum_file: Spectrum file path
            
        Returns:
            Dict[int, Spectrum]: Mapping from scan number to spectrum object
        """
        spectra = {}
        
        try:
            # Choose appropriate reading method based on file extension
            if spectrum_file.lower().endswith('.mgf'):
                spectra = self._read_mgf(spectrum_file)
            elif spectrum_file.lower().endswith('.mzml'):
                spectra = self._read_mzml(spectrum_file)
            else:
                logger.error(f"Unsupported spectrum file format: {spectrum_file}")
                return {}
                
            # Validate spectrum data
            valid_spectra = {}
            for scan_num, spectrum in spectra.items():
                mz_array, intensity_array = spectrum.get_peaks()
                
                # Check if arrays are empty
                if len(mz_array) == 0 or len(intensity_array) == 0:
                    logger.warning(f"Spectrum data for scan {scan_num} is empty")
                    continue
                    
                # Check if there are valid intensity values
                if not np.any(intensity_array > 0):
                    logger.warning(f"Spectrum for scan {scan_num} has no valid intensity values")
                    continue
                    
                valid_spectra[scan_num] = spectrum
                
            logger.info(f"Found {len(valid_spectra)} valid spectra from {len(spectra)} spectra")
            return valid_spectra
            
        except Exception as e:
            logger.error(f"Error reading spectrum file: {str(e)}")
            return {}
            
    def _read_mgf(self, mgf_file: str) -> Dict[int, Spectrum]:
        """Read MGF format spectrum file.
        
        Args:
            mgf_file: MGF file path
            
        Returns:
            Dict[int, Spectrum]: Mapping from scan number to spectrum object
        """
        spectra = {}
        current_spectrum = None
        current_scan = None
        mz_list = []
        intensity_list = []
        
        try:
            with open(mgf_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    if line.startswith('BEGIN IONS'):
                        mz_list = []
                        intensity_list = []
                        current_scan = None
                        
                    elif line.startswith('TITLE='):
                        # Extract scan number from title
                        try:
                            scan_start = line.find('.') + 1
                            scan_end = line.find('.', scan_start)
                            current_scan = int(line[scan_start:scan_end])
                        except:
                            logger.warning(f"Cannot extract scan number from title: {line}")
                            
                    elif line.startswith('END IONS'):
                        if current_scan is not None and mz_list and intensity_list:
                            try:
                                # Create spectrum object
                                current_spectrum = Spectrum(
                                    np.array(mz_list),
                                    np.array(intensity_list)
                                )
                                spectra[current_scan] = current_spectrum
                            except Exception as e:
                                logger.warning(f"Error creating spectrum object for scan {current_scan}: {str(e)}")
                                
                        mz_list = []
                        intensity_list = []
                        current_scan = None
                        
                    elif line and line[0].isdigit():
                        # Parse peak data
                        try:
                            mz, intensity = map(float, line.split()[:2])
                            mz_list.append(mz)
                            intensity_list.append(intensity)
                        except:
                            logger.warning(f"Cannot parse peak data: {line}")
                            
            logger.info(f"Read {len(spectra)} spectra from MGF file")
            return spectra
            
        except Exception as e:
            logger.error(f"Error reading MGF file: {str(e)}")
            return {}
            
    def _read_mzml(self, mzml_file: str) -> Dict[int, Spectrum]:
        """Read mzML format spectrum file.
        
        Args:
            mzml_file: mzML file path
            
        Returns:
            Dict[int, Spectrum]: Mapping from scan number to spectrum object
        """
        try:
            
            spectra = {}
            exp = pyopenms.MSExperiment()
            pyopenms.MzMLFile().load(mzml_file, exp)

            for i, spectrum in enumerate(exp):
                try:
                    # Get scan number
                    scan_number = int(spectrum.getNativeID().split('scan=')[-1])
                    
                    # Get peak data
                    mz_array, intensity_array = spectrum.get_peaks()
                    
                    # Create spectrum object
                    spectra[scan_number] = Spectrum(mz_array, intensity_array)
                    
                except Exception as e:
                    logger.warning(f"Error processing mzML spectrum {i}: {str(e)}")
                    continue
                    
            logger.info(f"Read {len(spectra)} spectra from mzML file")
            return spectra
            
        except ImportError:
            logger.error("Cannot import pyopenms module, please ensure it is properly installed")
            return {}
        except Exception as e:
            logger.error(f"Error reading mzML file: {str(e)}")
            return {}
            
    def process_psms(self, config: Dict) -> List[PSM]:
        """Process PSMs.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List[PSM]: List of PSM objects
        """
        # Read spectrum data
        spectra = self.read_spectrum_data(config['spectrum_file'])
        if not spectra:
            logger.error("No valid spectrum data found")
            return []
            
        # Process PSMs
        processed_psms = []
        for psm in self.psms:
            # Get corresponding spectrum
            spectrum = spectra.get(psm.scan_number)
            if spectrum is None:
                logger.warning(f"Cannot find spectrum corresponding to scan number {psm.scan_number}")
                continue
                
            # Set spectrum data
            mz_array, intensity_array = spectrum.get_peaks()
            psm.set_spectrum(mz_array, intensity_array)
            
            # Process PSM
            if psm.process(config):
                processed_psms.append(psm)
                
        return processed_psms

def main():
    """Entry point"""
    tool = PyLuciPHOr2()
    sys.exit(tool.run())

if __name__ == "__main__":
    main()