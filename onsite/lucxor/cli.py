#!/usr/bin/env python3
"""
Command line interface for pyLuciPHOr2
"""

import click
import os
import sys
import logging
import time
import random
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

import pandas as pd

from pyopenms import (
    MzMLFile,
    MSExperiment,
)

from onsite.idparquet import pyopenms_to_unimod_notation, save_dataframes
from .psm import PSM
from .peptide import Peptide
from .models import CIDModel, HCDModel
from .constants import DEFAULT_CONFIG
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
    help="Input identification file (idparquet directory)"
)
@click.option(
    "-out",
    "--output",
    "output",
    required=True,
    type=click.Path(),
    help="Output file (idparquet directory)"
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
    help="Target modifications (comma/space separated, e.g. 'Phospho(S),Phospho(T),Phospho(Y)' or 'PhosphoDecoy(A)'; default: Phospho (S), Phospho (T), Phospho (Y))"
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
    help="Score threshold for selecting high-quality PSMs for model training. "
         "If not specified (default 0.95), the threshold will be auto-adjusted based on detected score type: "
         "E-values (SpecEValue, expect) -> 0.01; "
         "Raw scores (RawScore, xcorr) -> 50th percentile (median); "
         "Probability scores (PEP) -> 0.95. "
         "For 'higher is better' scores: PSMs with score >= threshold are used. "
         "For 'lower is better' scores: PSMs with score <= threshold are used."
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
    default=1,
    help="Number of threads to use (default: 1)"
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="RNG seed for reproducible decoy permutations / model subsampling "
         "(default: 42). Guarantees deterministic, reproducible output for the "
         "default single-threaded run (--threads 1); with --threads > 1 the "
         "global RNG is shared across threads so exact reproducibility is not "
         "guaranteed.",
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
@click.option(
    "--disable-split-by-charge",
    is_flag=True,
    default=False,
    help="Disable splitting PSMs by charge state for model training (train a single global model)"
)
@click.option(
    "--compute-all-scores",
    "compute_all_scores",
    is_flag=True,
    default=False,
    help="Run all three algorithms (AScore, PhosphoRS, LucXor) and merge results",
)
@click.option(
    "--score-type",
    type=str,
    default=None,
    help="Score type to use for PSM filtering (default: auto-detect with priority: PEP > MSGF+ RawScore > Comet xcorr > SpecEValue)",
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
    seed,
    rt_tolerance,
    debug,
    log_file,
    disable_split_by_charge,
    compute_all_scores,
    score_type,
):
    """
    Modification site localization using pyLuciPHOr2 algorithm.

    This tool processes MS/MS spectra and peptide identifications to localize
    post-translational modifications using the LuciPHOr2 algorithm with
    false localization rate (FLR) calculation.
    """
    # If compute_all_scores is enabled, delegate to the unified handler
    if compute_all_scores:
        from onsite.onsitec import run_all_algorithms_from_single_cli
        # Determine if add_decoys should be True based on target_modifications
        add_decoys = any("PhosphoDecoy" in mod for mod in target_modifications)
        return run_all_algorithms_from_single_cli(
            in_file=input_spectrum,
            id_file=input_id,
            out_file=output,
            fragment_mass_tolerance=fragment_mass_tolerance,
            fragment_mass_unit=fragment_error_units,
            threads=threads,
            debug=debug,
            add_decoys=add_decoys,
        )
    
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
            seed=seed,
            rt_tolerance=rt_tolerance,
            debug=debug,
            disable_split_by_charge=disable_split_by_charge,
            score_type=score_type,
        )
        
        # Only call sys.exit if not being called from compute_all_scores
        if not compute_all_scores:
            sys.exit(exit_code)
        return exit_code

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
        
    def select_score_type(self, pep_ids, user_score_type: Optional[str] = None):
        """
        Select the best available score type based on priority.

        Priority order (context-aware):
        1. Posterior Error Probability (Percolator PEP) - always highest priority
        2. Search engine specific:
           - MSGF+: RawScore (MS:1002049) > SpecEValue (MS:1002052)
           - Comet: xcorr (COMET:xcorr) > expect

        Args:
            pep_ids: List of PeptideIdentification
            user_score_type: User-specified score type (optional)

        Returns:
            Tuple of (score_type, higher_score_better)
        """
        if not pep_ids or len(pep_ids) == 0:
            self.logger.warning("No peptide identifications to determine score type")
            return None, True

        if user_score_type:
            higher_better = True
            lower_better_scores = [
                "posterior error probability", "pep", "q-value", "qvalue",
                "expect", "evalue", "e-value", "specevalue", "ms:1002052", "ms:1002053",
                "ms:1002257",
                "ms:1001491", "ms:1001493",
            ]
            if any(kw in user_score_type.lower() for kw in lower_better_scores):
                higher_better = False
            self.logger.info(f"Using user-specified score type: {user_score_type} (higher_better={higher_better})")
            return user_score_type, higher_better

        # Get current score type from first peptide identification
        current_score_type = pep_ids[0].getScoreType() if hasattr(pep_ids[0], "getScoreType") else ""
        current_score_type_lower = current_score_type.lower()

        preferred_scores = [
            ("posterior error probability", False, 1),
            ("ms:1002049", True, 2),
            ("comet:xcorr", True, 2),
            ("specevalue", False, 3),
            ("ms:1002052", False, 3),
            ("expect", False, 3),
            ("ms:1002257", False, 3),
        ]

        current_priority = 999
        current_higher_better = True
        for score_pattern, higher_better, priority in preferred_scores:
            if score_pattern in current_score_type_lower:
                current_priority = priority
                current_higher_better = higher_better
                break

        best_userparam_score = None
        best_userparam_priority = 999
        best_userparam_higher_better = True

        if len(pep_ids[0].getHits()) > 0:
            hit = pep_ids[0].getHits()[0]

            for score_pattern, higher_better, priority in preferred_scores:
                score_exists = False
                actual_score_name = None

                if hit.metaValueExists(score_pattern):
                    score_exists = True
                    actual_score_name = score_pattern
                else:
                    variations = [
                        score_pattern,
                        score_pattern.upper(),
                        score_pattern.title(),
                        "COMET:xcorr" if score_pattern == "comet:xcorr" else None,
                        "MS:1002049" if score_pattern == "ms:1002049" else None,
                        "MS:1002052" if score_pattern == "ms:1002052" else None,
                        "MS:1002257" if score_pattern == "ms:1002257" else None,
                    ]
                    for v in variations:
                        if v and hit.metaValueExists(v):
                            score_exists = True
                            actual_score_name = v
                            break

                if score_exists and priority < best_userparam_priority:
                    best_userparam_score = actual_score_name
                    best_userparam_priority = priority
                    best_userparam_higher_better = higher_better

        if best_userparam_score and best_userparam_priority <= current_priority:
            return best_userparam_score, best_userparam_higher_better
        elif current_priority < 999:
            return current_score_type, current_higher_better

        higher_score_better = True
        if hasattr(pep_ids[0], "isHigherScoreBetter"):
            higher_score_better = pep_ids[0].isHigherScoreBetter()
        elif hasattr(pep_ids[0], "getHigherScoreBetter"):
            higher_score_better = pep_ids[0].getHigherScoreBetter()

        self.logger.warning(f"Using fallback score type: {current_score_type} (higher_better={higher_score_better})")
        return current_score_type, higher_score_better

    def get_psm_score(self, pep_id, hit, score_type: str, higher_score_better: bool) -> float:
        """
        Extract score from peptide hit based on score type.
        
        Args:
            pep_id: PeptideIdentification object
            hit: PeptideHit object
            score_type: Name of the score type to extract
            higher_score_better: Whether higher scores are better
            
        Returns:
            Normalized score (always higher is better, range 0-1 for probability-based scores)
        """
        score = hit.getScore()
        
        # Check if score exists as UserParam
        if hit.metaValueExists(score_type):
            score = float(hit.getMetaValue(score_type))
        
        # Normalize score based on type
        score_type_lower = (score_type or "").lower()
        is_probability_score = any(k in score_type_lower for k in [
            "posterior error probability", "pep", "q-value", "qvalue",
            "ms:1001493", "ms:1001491"
        ])
        
        if is_probability_score:
            # For probability-based scores (PEP, Q-value), convert to 1-score
            # These are already probabilities between 0 and 1
            if 0 <= score <= 1:
                return 1.0 - score
            else:
                self.logger.warning(f"PEP/Q-value out of range [0,1]: {score}, using as-is")
                return score
        
        # For E-values and other "lower is better" scores, keep as-is
        # Don't transform them - they will be handled differently in filtering
        if not higher_score_better:
            return score
        
        return score

    def load_input_files(
        self, input_id: str, input_spectrum: str, score_type: Optional[str] = None
    ):
        """Load input files from idParquet directory. Returns (pep_ids, prot_ids, exp)."""
        from pyopenms import AASequence, PeptideHit, PeptideIdentification
        from onsite.idparquet import load_dataframes, unimod_to_pyopenms_notation

        psms_df, proteins_df, _, _ = load_dataframes(input_id)
        self._psms_df_template = psms_df  # preserved for output schema
        if psms_df.empty:
            self.logger.warning("No peptide identifications found in input file")
            return [], [], None

        # Keep only best hits (hit_index == 0), then build pyOpenMS objects for internal use
        best_psms = psms_df[psms_df["hit_index"] == 0].copy()
        pep_ids = []
        self.logger.info(f"Loaded {len(best_psms)} PSMs from idParquet")

        # Build PeptideIdentification objects from DataFrame rows
        for _, row in best_psms.iterrows():
            pid = PeptideIdentification()
            peptidoform = str(row.get("peptidoform", row.get("sequence", "")))
            pyo_seq = unimod_to_pyopenms_notation(peptidoform)
            try:
                seq = AASequence.fromString(pyo_seq)
            except Exception:
                seq = AASequence.fromString(str(row.get("sequence", "")))
            hit = PeptideHit()
            hit.setSequence(seq)
            if pd.notna(row.get("precursor_charge")):
                hit.setCharge(int(row["precursor_charge"]))
            if pd.notna(row.get("posterior_error_probability")):
                hit.setScore(float(row["posterior_error_probability"]))
                hit.setMetaValue("Posterior Error Probability", float(row["posterior_error_probability"]))
            elif pd.notna(row.get("score")):
                hit.setScore(float(row["score"]))
            # Populate psm_metavalues as UserParams for score-type detection
            mv = row.get("psm_metavalues")
            if isinstance(mv, np.ndarray):
                for m in mv:
                    if isinstance(m, dict):
                        try:
                            val = m["value"]
                            vtype = str(m.get("value_type", "string"))
                            if vtype in ("double", "float"):
                                try: val = float(val)
                                except: pass
                            elif vtype in ("int", "integer"):
                                try: val = int(val)
                                except: pass
                            hit.setMetaValue(str(m["name"]), val)
                        except Exception:
                            pass
            # Populate additional_scores
            add_scores = row.get("additional_scores")
            if isinstance(add_scores, np.ndarray):
                for ascore in add_scores:
                    if isinstance(ascore, dict):
                        sname = ascore.get("score_name", "")
                        sval = ascore.get("score_value", "")
                        if sname and sval is not None:
                            try:
                                hit.setMetaValue(str(sname), float(sval))
                            except (ValueError, TypeError):
                                try:
                                    hit.setMetaValue(str(sname), str(sval))
                                except Exception:
                                    pass
            pid.setHits([hit])
            if pd.notna(row.get("rt")):
                pid.setRT(float(row["rt"]))
            if pd.notna(row.get("observed_mz")):
                pid.setMZ(float(row["observed_mz"]))
            elif pd.notna(row.get("calculated_mz")):
                pid.setMZ(float(row["calculated_mz"]))
            if pd.notna(row.get("spectrum_reference")):
                pid.setMetaValue("spectrum_reference", str(row["spectrum_reference"]))
            if pd.notna(row.get("scan")):
                pid.setMetaValue("scan_number", int(row["scan"]))
            pid.setScoreType(str(row.get("score_type", "")))
            pep_ids.append(pid)

        # Select best score type based on priority
        score_type_param, higher_score_better = self.select_score_type(pep_ids, user_score_type=score_type)

        # Override higher_score_better for probability scores after conversion
        if score_type_param and any(k in score_type_param.lower() for k in [
            "posterior error probability", "pep", "q-value", "qvalue",
            "ms:1001493", "ms:1001491"
        ]):
            higher_score_better = True

        self.score_type = score_type_param
        self.higher_score_better = higher_score_better

        # Load spectra
        exp = MSExperiment()
        MzMLFile().load(input_spectrum, exp)
        if exp.empty():
            self.logger.warning("No spectra found in input file")
            return [], [], None

        return pep_ids, proteins_df, exp
        
        # Store score info for later use
        self.score_type = score_type_param
        self.higher_score_better = higher_score_better

        # Load spectra
        exp = MSExperiment()
        MzMLFile().load(input_spectrum, exp)
        
        if exp.empty():
            self.logger.warning("No spectra found in input file")
            return [], [], None
            
        return pep_ids, proteins_df, exp
        
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
        disable_split_by_charge: bool = False,
        score_type: Optional[str] = None,
        seed: int = 42,
    ) -> int:
        """
        LuciPHOr2 main workflow:
        1. Read input files and collect all PSMs.
        2. Train CID/HCD model with high-scoring PSMs.
        3. Score all PSMs with the trained model.
        4. Exit with error if insufficient high-scoring PSMs.
        """
        # Seed the RNGs before any stochastic step (decoy permutation shuffling in
        # psm.py, model-subsampling np.random.choice in models.py) so a default
        # single-threaded run is fully reproducible. Done once here because both
        # the standalone CLI and the `onsite all` path funnel through run().
        random.seed(seed)
        np.random.seed(seed)
        self.logger.debug(f"Seeded RNGs (random, numpy) with seed={seed}")

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
                "disable_split_by_charge": disable_split_by_charge,
            }
        )

        # Start timing
        start_time = time.time()

        self.logger.info("Loading input files...")
        self.logger.debug(f"Debug mode: {debug}")
        self.logger.debug(f"Log level: {logging.getLogger().level}")

        pep_ids, prot_ids, exp = self.load_input_files(input_id, input_spectrum, score_type)
        if not pep_ids or exp is None:
            self.logger.error("No valid peptide identification or spectrum data found, process terminated.")
            return 1

        # 1. Create scan number to spectrum mapping AND RT-sorted index for fast lookup
        spectrum_map = {}
        rt_spectrum_list = []  # List of (RT, spectrum) tuples for binary search

        for spectrum in exp:
            rt = spectrum.getRT()
            rt_spectrum_list.append((rt, spectrum))
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

        # Sort by RT for binary search (O(n log n) once, then O(log n) per lookup)
        rt_spectrum_list.sort(key=lambda x: x[0])
        rt_values = np.array([rt for rt, _ in rt_spectrum_list])

        # 2. Collect all PSM objects
        all_psms = []
        for i, pep_id in enumerate(pep_ids, 1):
            hit = pep_id.getHits()[0]
            sequence = hit.getSequence().toString()
            # Strip pyOpenMS N-terminal dot notation like ".(Acetyl)"
            if sequence.startswith("."):
                end_paren = sequence.find(")")
                if end_paren >= 0:
                    sequence = sequence[end_paren + 1:]
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
                # If scan number unavailable or no match found, try RT matching with binary search
                # O(log n) instead of O(n) per lookup
                idx = np.searchsorted(rt_values, rt)

                # Check candidates near the insertion point
                candidates = []
                if idx > 0:
                    candidates.append(idx - 1)
                if idx < len(rt_values):
                    candidates.append(idx)
                if idx + 1 < len(rt_values):
                    candidates.append(idx + 1)

                for candidate_idx in candidates:
                    candidate_rt, candidate_spec = rt_spectrum_list[candidate_idx]
                    if abs(candidate_rt - rt) <= rt_tolerance:
                        spectrum = candidate_spec
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
                
                # Get score using the selected score type
                score = self.get_psm_score(pep_id, hit, self.score_type, self.higher_score_better)
                
                # Assign score to PSM and peptide
                psm.psm_score = score
                peptide.score = score
                
                # Store score metadata for filtering
                psm.score_type = self.score_type
                psm.higher_score_better = self.higher_score_better
                
                all_psms.append(psm)
            else:
                self.logger.warning(f'No matching spectrum found - RT: {rt}, Scan: {scan_num if scan_num else "N/A"}')

        if not all_psms:
            self.logger.error("No PSMs collected, process terminated.")
            return 1

        # 3. Train model with high-scoring PSMs
        modeling_score_threshold_val = config.get("modeling_score_threshold", 0.95)
        
        # Auto-adjust modeling_score_threshold based on score_type if using default value
        # Only adjust if user didn't explicitly set a non-default threshold
        user_set_threshold = modeling_score_threshold != 0.95  # Check if user changed from default
        
        if not user_set_threshold:
            # Auto-adjust threshold based on detected score type
            score_type_lower = self.score_type.lower()
            
            # Check if it's a probability score (case-insensitive)
            is_probability_score = any(keyword in score_type_lower for keyword in [
                "posterior error probability", "pep", "q-value", "qvalue",
                "ms:1001493", "ms:1001491"
            ])
            
            # Check if it's an E-value score
            is_evalue_score = any(keyword in score_type_lower for keyword in [
                "evalue", "e-value", "expect", "specevalue", "ms:1002052", "ms:1002257"
            ])
            
            # Check if it's a raw score
            is_raw_score = any(keyword in score_type_lower for keyword in [
                "rawscore", "ms:1002049", "xcorr", "comet:xcorr"
            ])
            
            if is_probability_score:
                # For PEP/Q-value (already transformed to 1-score, so higher is better)
                modeling_score_threshold_val = 0.95
                self.logger.info(f"Auto-adjusted modeling_score_threshold to {modeling_score_threshold_val} for probability-based score")
            elif is_evalue_score:
                # For E-values (lower is better)
                modeling_score_threshold_val = 0.01
                self.logger.info(f"Auto-adjusted modeling_score_threshold to {modeling_score_threshold_val} for E-value score type")
            elif is_raw_score:
                # For raw scores, use percentile-based approach
                all_scores = [psm.psm_score for psm in all_psms if hasattr(psm, "psm_score") and not np.isnan(psm.psm_score)]
                if all_scores:
                    # Use 50th percentile (median) as threshold for raw scores
                    # This balances quality and quantity, keeping top 50% of PSMs
                    modeling_score_threshold_val = np.percentile(all_scores, 50)
                    self.logger.info(f"Auto-adjusted modeling_score_threshold to {modeling_score_threshold_val:.2f} (50th percentile) for raw score type")
                else:
                    modeling_score_threshold_val = 0.0  # Fallback: accept all
                    self.logger.warning(f"No valid scores found, using threshold {modeling_score_threshold_val}")
            else:
                # Unknown score type, keep default
                modeling_score_threshold_val = 0.95
                self.logger.info(f"Using default modeling_score_threshold {modeling_score_threshold_val} for unknown score type")
            
            # Update config with adjusted threshold
            config["modeling_score_threshold"] = modeling_score_threshold_val
        
        # Log the score type and threshold being used
        self.logger.info(f"Using score type '{self.score_type}' for PSM filtering")
        self.logger.info(f"Score direction: {'higher is better' if self.higher_score_better else 'lower is better'}")
        self.logger.info(f"Final modeling score threshold: {modeling_score_threshold_val}")
        
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
        # Use correct comparison based on score direction
        high_score_psms = []
        for psm in phospho_psms:
            if hasattr(psm, "psm_score"):
                # For "higher is better" scores, use >= threshold
                # For "lower is better" scores (like E-values), use <= threshold
                if self.higher_score_better:
                    if psm.psm_score >= modeling_score_threshold_val:
                        high_score_psms.append(psm)
                else:
                    # For "lower is better" scores (E-values), use the modeling_score_threshold directly
                    # Lower threshold means more stringent filtering
                    if psm.psm_score <= modeling_score_threshold_val:
                        high_score_psms.append(psm)
            elif hasattr(psm, "peptide") and hasattr(psm.peptide, "score"):
                if self.higher_score_better:
                    if psm.peptide.score >= modeling_score_threshold_val:
                        high_score_psms.append(psm)
                else:
                    # For "lower is better" scores, use the modeling_score_threshold directly
                    if psm.peptide.score <= modeling_score_threshold_val:
                        high_score_psms.append(psm)
        
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

        # 6. Build output DataFrame (using second round calculation results)

        result_rows = []
        phospho_count = 0
        for pep_idx, psm in enumerate(all_psms):
            # Get original pep_id info
            orig_pep_id = pep_ids[pep_idx] if pep_idx < len(pep_ids) else None
            mz = orig_pep_id.getMZ() if orig_pep_id else 0.0
            rt = orig_pep_id.getRT() if orig_pep_id else 0.0
            spec_ref = (str(orig_pep_id.getMetaValue("spectrum_reference"))
                       if orig_pep_id and orig_pep_id.metaValueExists("spectrum_reference") else "")
            scan_num = (int(orig_pep_id.getMetaValue("scan_number"))
                       if orig_pep_id and orig_pep_id.metaValueExists("scan_number") else 0)
            ref_file = (str(orig_pep_id.getMetaValue("reference_file_name"))
                       if orig_pep_id and orig_pep_id.metaValueExists("reference_file_name") else "")

            # Best sequence
            best_sequence = psm.get_best_sequence(include_decoys=False)
            seq_str = best_sequence or psm.peptide.peptide
            base_seq = psm.peptide.peptide  # fallback unmodified

            # Use pyOpenMS to get unmodified string if possible
            try:
                from pyopenms import AASequence
                seq_obj = AASequence.fromString(seq_str) if seq_str else None
                if seq_obj:
                    base_seq = seq_obj.toUnmodifiedString()
            except Exception:
                pass

            # Preserve original peptidoform from input
            _orig_pf = ""
            _tmpl = getattr(self, "_psms_df_template", None)
            if _tmpl is not None:
                _best_pf = _tmpl[_tmpl["hit_index"] == 0]
                if pep_idx < len(_best_pf):
                    _orig_pf = str(_best_pf.iloc[pep_idx].get("peptidoform", ""))

            # Build psm_metavalues: preserve original + add Luciphor scores
            _lucxor_df = getattr(self, "_psms_df_template", None)
            if _lucxor_df is not None:
                _best = _lucxor_df[_lucxor_df["hit_index"] == 0]
                if pep_idx < len(_best):
                    _orig_mv = _best.iloc[pep_idx].get("psm_metavalues")
                    _orig_list = list(_orig_mv) if isinstance(_orig_mv, np.ndarray) else []
                else:
                    _orig_list = []
            else:
                _orig_list = []
            _lucxor_managed = {"search_engine_sequence", "Luciphor_pep_score", "Luciphor_global_flr",
                               "Luciphor_local_flr", "Luciphor_site_scores"}
            _filtered_orig = [m for m in _orig_list if isinstance(m, dict) and m.get("name") not in _lucxor_managed]
            metas = _filtered_orig[:]
            for key in ["search_engine_sequence", "Luciphor_pep_score", "Luciphor_global_flr",
                        "Luciphor_local_flr", "Luciphor_site_scores"]:
                val = None
                if key == "search_engine_sequence":
                    val = getattr(psm, "search_engine_sequence", psm.peptide.peptide)
                elif key == "Luciphor_pep_score":
                    val = float(getattr(psm, "psm_score", 0))
                elif key == "Luciphor_global_flr":
                    val = float(getattr(psm, "global_flr", 0))
                elif key == "Luciphor_local_flr":
                    val = float(getattr(psm, "local_flr", 0))
                elif key == "Luciphor_site_scores":
                    val = str(psm.get_site_scores())
                if val is not None:
                    vtype = "double" if isinstance(val, float) else "string"
                    metas.append({"name": key, "value": str(val), "value_type": vtype})

            score = float(getattr(psm, "delta_score", 0))
            is_decoy = getattr(psm, "is_decoy", False)

            result_rows.append({
                "sequence": base_seq,
                "peptidoform": _orig_pf or peptidoform,
                "precursor_charge": int(getattr(psm, "charge", 0)),
                "calculated_mz": mz,
                "observed_mz": mz,
                "is_decoy": bool(is_decoy),
                "score": score,
                "score_type": "Luciphor_delta_score",
                "higher_score_better": True,
                "rt": rt,
                "scan": scan_num,
                "spectrum_reference": spec_ref,
                "reference_file_name": ref_file,
                "hit_index": 0,
                "peptide_identification_index": pep_idx,
                "psm_metavalues": np.array(metas, dtype=object),
                "modifications": np.array([], dtype=object),
                "protein_accessions": np.array([], dtype=object),
                "additional_scores": np.array([], dtype=object),
                "run_identifier": "",
            })

            try:
                if "(Phospho)" in seq_str:
                    phospho_count += 1
            except Exception:
                pass

        # 7. Save results (idParquet format)
        out_df = pd.DataFrame(result_rows) if result_rows else pd.DataFrame()
        save_dataframes(output, out_df, prot_ids, template_df=self._psms_df_template, source_idparquet=input_id)
        self.logger.info(f"Results saved to: {output}")

        # 8. Processing completed - print run summary similar to Ascore
        elapsed = time.time() - start_time
        total = len(pep_ids)
        processed = len(result_rows)
        errors = max(0, total - processed)
        new_pep_ids = result_rows  # alias for backward compat in reporting

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
