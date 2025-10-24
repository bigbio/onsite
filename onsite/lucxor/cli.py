#!/usr/bin/env python3
"""
Command line interface for pyLuciPHOr2
"""

import click
import os
import sys
import logging
import time
import json
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from pyopenms import (
    IdXMLFile,
    MzMLFile,
    MSExperiment,
    PeptideIdentification,
    ProteinIdentification,
    IDFilter,
)

from .psm import PSM
from .peptide import Peptide
from .models import CIDModel, HCDModel
from .constants import NTERM_MOD, CTERM_MOD, AA_MASSES, DEFAULT_CONFIG
from .spectrum import Spectrum
from .flr import FLRCalculator
from .parallel import parallel_psm_processing, get_optimal_thread_count

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-in",
    "--input-spectrum",
    "input_spectrum",
    required=True,
    type=click.Path(exists=True),
    help="Input spectrum file (mzML)"
)
@click.option(
    "-id",
    "--input-id",
    "input_id",
    required=True,
    type=click.Path(exists=True),
    help="Input identification file (idXML)"
)
@click.option(
    "-out",
    "--output",
    "output",
    required=True,
    type=click.Path(),
    help="Output file (idXML)"
)
@click.option(
    "--fragment-method",
    type=click.Choice(["CID", "HCD"], case_sensitive=False),
    default="CID",
    help="Fragmentation method (default: CID)"
)
@click.option(
    "--fragment-mass-tolerance",
    type=float,
    default=0.5,
    help="Tolerance of the peaks in the fragment spectrum (default: 0.5)"
)
@click.option(
    "--fragment-error-units",
    type=click.Choice(["Da", "ppm"], case_sensitive=False),
    default="Da",
    help="Unit of fragment mass tolerance (default: Da)"
)
@click.option(
    "--min-mz",
    type=float,
    default=150.0,
    help="Do not consider peaks below this value (default: 150.0)"
)
@click.option(
    "--target-modifications",
    multiple=True,
    default=["Phospho (S)", "Phospho (T)", "Phospho (Y)"],
    help="List of target modifications (default: Phospho (S), Phospho (T), Phospho (Y))"
)
@click.option(
    "--neutral-losses",
    multiple=True,
    default=["sty -H3PO4 -97.97690"],
    help="List of neutral losses (default: sty -H3PO4 -97.97690)"
)
@click.option(
    "--decoy-mass",
    type=float,
    default=79.966331,
    help="Mass to add for decoy generation (default: 79.966331)"
)
@click.option(
    "--decoy-neutral-losses",
    multiple=True,
    default=["X -H3PO4 -97.97690"],
    help="List of decoy neutral losses (default: X -H3PO4 -97.97690)"
)
@click.option(
    "--max-charge-state",
    type=int,
    default=5,
    help="Maximum charge state to consider (default: 5)"
)
@click.option(
    "--max-peptide-length",
    type=int,
    default=40,
    help="Maximum peptide length (default: 40)"
)
@click.option(
    "--max-num-perm",
    type=int,
    default=16384,
    help="Maximum number of permutations (default: 16384)"
)
@click.option(
    "--modeling-score-threshold",
    type=float,
    default=0.95,
    help="Minimum score for modeling (default: 0.95)"
)
@click.option(
    "--scoring-threshold",
    type=float,
    default=0.0,
    help="Minimum score threshold (default: 0.0)"
)
@click.option(
    "--min-num-psms-model",
    type=int,
    default=50,
    help="Minimum number of PSMs for modeling (default: 50)"
)
@click.option(
    "--threads",
    type=int,
    default=4,
    help="Number of threads to use (default: 4)"
)
@click.option(
    "--rt-tolerance",
    type=float,
    default=0.01,
    help="Retention time tolerance (default: 0.01)"
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode"
)
@click.option(
    "--log-file",
    type=str,
    default=None,
    help="Log file path (only used in debug mode, default: {output_base}_debug.log)"
)
def lucxor(
    input_spectrum,
    input_id,
    output,
    fragment_method,
    fragment_mass_tolerance,
    fragment_error_units,
    min_mz,
    target_modifications,
    neutral_losses,
    decoy_mass,
    decoy_neutral_losses,
    max_charge_state,
    max_peptide_length,
    max_num_perm,
    modeling_score_threshold,
    scoring_threshold,
    min_num_psms_model,
    threads,
    rt_tolerance,
    debug,
    log_file,
):
    """
    Modification site localization using pyLuciPHOr2 algorithm.

    This tool processes MS/MS spectra and peptide identifications to localize
    post-translational modifications using the LuciPHOr2 algorithm with
    false localization rate (FLR) calculation.
    """
    try:
        # Setup logging first
        setup_logging(debug, log_file, output)

        # Create tool instance and run
        tool = PyLuciPHOr2()
        exit_code = tool.run(
            input_spectrum=input_spectrum,
            input_id=input_id,
            output=output,
            fragment_method=fragment_method,
            fragment_mass_tolerance=fragment_mass_tolerance,
            fragment_error_units=fragment_error_units,
            min_mz=min_mz,
            target_modifications=target_modifications,
            neutral_losses=neutral_losses,
            decoy_mass=decoy_mass,
            decoy_neutral_losses=decoy_neutral_losses,
            max_charge_state=max_charge_state,
            max_peptide_length=max_peptide_length,
            max_num_perm=max_num_perm,
            modeling_score_threshold=modeling_score_threshold,
            scoring_threshold=scoring_threshold,
            min_num_psms_model=min_num_psms_model,
            threads=threads,
            rt_tolerance=rt_tolerance,
            debug=debug,
        )
        
        sys.exit(exit_code)

    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        if debug:
            logger.error(f"Error: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
        sys.exit(1)


def setup_logging(debug, log_file, output):
    """Setup logging configuration"""
    # Configure log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Only configure file handler in debug mode
    if debug:
        # Get output filename (without extension)
        output_base = os.path.splitext(output)[0]
        log_file_path = log_file or f"{output_base}_debug.log"

        # Configure file handler
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set third-party library log levels
    logging.getLogger("numpy").setLevel(logging.WARNING)
    logging.getLogger("scipy").setLevel(logging.WARNING)


class PyLuciPHOr2:
    """Main class for pyLuciPHOr2 command line tool"""

    def __init__(self):
        """Initialize PyLuciPHOr2."""
        self.logger = logging.getLogger(__name__)
        # Initialize model
        self.model = None
        self.psms = []
        self.config = {}

    def load_input_files(
        self, input_id: str, input_spectrum: str
    ) -> Tuple[List[PeptideIdentification], List[ProteinIdentification], MSExperiment]:
        """Load input files"""
        # Load identifications
        pep_ids = []
        prot_ids = []
        IdXMLFile().load(input_id, prot_ids, pep_ids)

        if not pep_ids:
            self.logger.warning("No peptide identifications found in input file")
            return [], [], None

        # Keep only best hits
        IDFilter().keepNBestHits(pep_ids, 1)

        # Load spectra
        exp = MSExperiment()
        MzMLFile().load(input_spectrum, exp)
        
        if exp.empty():
            self.logger.warning("No spectra found in input file")
            return [], [], None
            
        return pep_ids, prot_ids, exp
        
    def initialize_model(self, config: Dict) -> None:
        """Initialize scoring model"""
        fragment_method = config.get("fragment_method", "HCD")
        
        if fragment_method == "HCD":
            self.model = HCDModel(config)
            self.logger.info("HCD Model initialized")
        elif fragment_method == "CID":
            self.model = CIDModel(config)
            self.logger.info("CID Model initialized")
        else:
            raise ValueError(f"Unsupported fragment method: {fragment_method}")
        
    def run(
        self,
        input_spectrum: str,
        input_id: str,
        output: str,
        fragment_method: str,
        fragment_mass_tolerance: float,
        fragment_error_units: str,
        min_mz: float,
        target_modifications: tuple,
        neutral_losses: tuple,
        decoy_mass: float,
        decoy_neutral_losses: tuple,
        max_charge_state: int,
        max_peptide_length: int,
        max_num_perm: int,
        modeling_score_threshold: float,
        scoring_threshold: float,
        min_num_psms_model: int,
        threads: int,
        rt_tolerance: float,
        debug: bool,
    ) -> int:
        """
        LuciPHOr2 main workflow:
        1. Read input files and collect all PSMs.
        2. Train CID/HCD model with high-scoring PSMs.
        3. Score all PSMs with the trained model.
        4. Exit with error if insufficient high-scoring PSMs.
        """
        config = DEFAULT_CONFIG.copy()

        # Parse target_modifications to handle comma-separated format
        parsed_target_modifications = []
        for mod in target_modifications:
            if "," in mod:
                # Split comma-separated modifications
                parsed_target_modifications.extend([m.strip() for m in mod.split(",")])
            else:
                parsed_target_modifications.append(mod.strip())

        config.update(
            {
                "fragment_method": fragment_method,
                "fragment_mass_tolerance": fragment_mass_tolerance,
                "fragment_error_units": fragment_error_units,
                "min_mz": min_mz,
                "target_modifications": parsed_target_modifications,
                "neutral_losses": list(neutral_losses),
                "decoy_mass": decoy_mass,
                "decoy_neutral_losses": list(decoy_neutral_losses),
                "max_charge_state": max_charge_state,
                "max_peptide_length": max_peptide_length,
                "max_num_perm": max_num_perm,
                "modeling_score_threshold": modeling_score_threshold,
                "scoring_threshold": scoring_threshold,
                "min_num_psms_model": min_num_psms_model,
                "num_threads": threads,
                "rt_tolerance": rt_tolerance,
            }
        )

        # Start timing
        start_time = time.time()

        self.logger.info("Loading input files...")
        self.logger.debug(f"Debug mode: {debug}")
        self.logger.debug(f"Log level: {logging.getLogger().level}")

        pep_ids, prot_ids, exp = self.load_input_files(input_id, input_spectrum)
        if not pep_ids or exp is None:
            self.logger.error("No valid peptide identification or spectrum data found, process terminated.")
            return 1

        # 1. Create scan number to spectrum mapping
        spectrum_map = {}
        for spectrum in exp:
            try:
                # Extract scan number from native ID
                native_id = spectrum.getNativeID()
                if "scan=" in native_id:
                    scan_num = int(native_id.split("scan=")[-1])
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
            if pep_id.metaValueExists("scan_number"):
                scan_num = pep_id.getMetaValue("scan_number")
            elif pep_id.metaValueExists("spectrum_reference"):
                spec_ref = pep_id.getMetaValue("spectrum_reference")
                if "scan=" in spec_ref:
                    scan_num = int(spec_ref.split("scan=")[-1])
            
            # First try to find by scan number
            if scan_num is not None and scan_num in spectrum_map:
                spectrum = spectrum_map[scan_num]
                self.logger.debug(f"Found matching spectrum by scan number {scan_num}")
            else:
                # If scan number unavailable or no match found, try RT matching
                for spec in exp:
                    if abs(spec.getRT() - rt) <= rt_tolerance:
                        spectrum = spec
                        self.logger.debug(f"Found matching spectrum by RT {rt}")
                        break
            
            if spectrum:
                spectrum_dict = {
                    "mz": spectrum.get_peaks()[0],
                    "intensities": spectrum.get_peaks()[1],
                    "native_id": spectrum.getNativeID(),
                    "rt": spectrum.getRT()
                }
                peptide = Peptide(sequence, config, charge=charge)
                psm = PSM(peptide, spectrum_dict, config=config)
                
                # Set search_engine_sequence to the original sequence
                psm.search_engine_sequence = sequence
                
                # Automatically determine score type and assign values
                score_type = pep_id.getScoreType() if hasattr(pep_id, "getScoreType") else None
                score = hit.getScore()
                higher_score_better = True
                if hasattr(pep_id, "isHigherScoreBetter"):
                    higher_score_better = pep_id.isHigherScoreBetter()
                elif hasattr(pep_id, "getHigherScoreBetter"):
                    higher_score_better = pep_id.getHigherScoreBetter()
                
                # If it"s a PEP score (lower is better), convert to 1-PEP
                if (score_type and "posterior error probability" in score_type.lower()) or (higher_score_better is False):
                    psm.psm_score = 1.0 - score
                    peptide.score = 1.0 - score
                else:
                    psm.psm_score = score
                    peptide.score = score
                all_psms.append(psm)
            else:
                self.logger.warning(f'No matching spectrum found - RT: {rt}, Scan: {scan_num if scan_num else "N/A"}')

        if not all_psms:
            self.logger.error("No PSMs collected, process terminated.")
            return 1

        # 3. Train model with high-scoring PSMs
        modeling_score_threshold_val = config.get("modeling_score_threshold", 0.95)
        
        # First filter PSMs with modification sites
        phospho_psms = []
        for psm in all_psms:
            if hasattr(psm, "peptide") and psm.peptide is not None:
                # Check if there are potential modification sites and reported modification sites
                num_pps = getattr(psm.peptide, "num_pps", 0)
                num_rps = getattr(psm.peptide, "num_rps", 0)
                
                # Must have potential modification sites and reported modification sites
                if num_pps > 0 and num_rps > 0:
                    phospho_psms.append(psm)
        
        # Then filter high-scoring PSMs from PSMs with modification sites
        high_score_psms = [psm for psm in phospho_psms if hasattr(psm, "psm_score") and psm.psm_score >= modeling_score_threshold_val]
        if not high_score_psms:
            high_score_psms = [psm for psm in phospho_psms if hasattr(psm, "peptide") and hasattr(psm.peptide, "score") and psm.peptide.score >= modeling_score_threshold_val]
        
        self.logger.info(f"Total PSMs: {len(all_psms)}")
        self.logger.info(f"PSMs with modification sites: {len(phospho_psms)}")
        self.logger.info(f"High-scoring PSMs for training: {len(high_score_psms)}")
        
        if not high_score_psms or len(high_score_psms) < config.get("min_num_psms_model", 50):
            self.logger.error(f'Insufficient high-scoring PSMs for model training (need at least {config.get("min_num_psms_model", 50)}, actual {len(high_score_psms)}), process terminated.')
            raise RuntimeError("Not enough high-scoring PSMs for model training.")
        
        # Group statistics by charge state
        charge_stats = defaultdict(list)
        for psm in high_score_psms:
            charge = getattr(psm, "charge", 0)
            charge_stats[charge].append(psm)

        # 4. Initialize and train model
        fragment_method_val = config.get("fragment_method", "HCD")
        if fragment_method_val == "HCD":
            self.model = HCDModel(config)
            self.logger.info("Training HCD model...")
        elif fragment_method_val == "CID":
            self.model = CIDModel(config)
            self.logger.info("Training CID model...")
        else:
            raise ValueError(f"Unsupported fragment method: {fragment_method_val}")
        self.model.build(high_score_psms)
        self.logger.info("Model training completed.")

        # 5. Score all PSMs with trained model
        self.logger.info("Starting first round calculation (including decoy permutations)...")
        
        # Use multi-threading for PSM processing
        num_threads = get_optimal_thread_count(len(all_psms), max_threads=threads)
        self.logger.info(f"Using {num_threads} threads for PSM processing...")
        
        parallel_psm_processing(all_psms, model=self.model, round_number=0, num_threads=num_threads)

        # === Round 1: Execute FLR calculation and establish mapping relationships ===
        self.logger.info("Starting first round FLR calculation...")
        
        # Create a global FLR calculator
        global_flr_calculator = FLRCalculator(
            min_delta_score=config.get("min_delta_score", 0.1),
            min_psms_per_charge=config.get("min_num_psms_model", 50)
        )
        
        # Collect delta score data for all PSMs (first round results)
        for psm in all_psms:
            if hasattr(psm, "delta_score") and not np.isnan(psm.delta_score):
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
                if hasattr(psm, "search_engine_sequence"):
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
        IdXMLFile().store(output, prot_ids, new_pep_ids)
        self.logger.info(f"Results saved to: {output}")

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

        if debug:
            self.logger.info("Processing completed successfully")
            self.logger.info({
                "total": total,
                "processed": processed,
                "phospho": phospho_count,
                "errors": errors,
                "elapsed_sec": round(elapsed, 2),
                "speed_ids_per_sec": round(processed/elapsed, 2) if elapsed > 0 else None
            })

        return 0

def main():
    """Entry point for standalone LucXor CLI."""
    lucxor()


if __name__ == "__main__":
    main()

