#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import click
import traceback
import time
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyopenms import *
from onsite.ascore import AScore

@click.command()
@click.option('-in', '--in-file', 'in_file', required=True, 
              help='Input mzML file path', type=click.Path(exists=True))
@click.option('-id', '--id-file', 'id_file', required=True,
              help='Input idXML file path', type=click.Path(exists=True))
@click.option('-out', '--out-file', 'out_file', required=True,
              help='Output idXML file path', type=click.Path())
@click.option('--fragment-mass-tolerance', 'fragment_mass_tolerance', type=float, default=0.05,
              help='Fragment mass tolerance value (default: 0.05)')
@click.option('--fragment-mass-unit', 'fragment_mass_unit', type=click.Choice(['Da', 'ppm']), default='Da',
              help='Tolerance unit (default: Da)')
@click.option('--threads', 'threads', type=int, default=4,
              help='Number of parallel threads (default: 4)')
@click.option('--debug', 'debug', is_flag=True,
              help='Enable debug output and write debug log')
@click.option('--add-decoys', 'add_decoys', is_flag=True, default=False,
              help='Include A (PhosphoDecoy) as potential phosphorylation site')
def main(in_file, id_file, out_file, fragment_mass_tolerance, fragment_mass_unit, 
         threads, debug, add_decoys):
    """
    Phosphorylation site localization scoring tool using AScore algorithm.
    
    This tool processes MS/MS spectra and peptide identifications to localize
    phosphorylation sites using the AScore algorithm.
    """
    try:
        # Initialize processing pipeline
        exp = load_spectra(in_file)
        protein_ids, peptide_ids = load_identifications(id_file)
        # Pre-initialize spectrum cache once in the main thread
        if peptide_ids:
            _ = find_spectrum_by_mz(exp, peptide_ids[0].getMZ(), peptide_ids[0].getRT())
        
        # Initialize debug log (only when --debug)
        log_file = f"{out_file}.debug.log"
        logger = log_debug(log_file, debug)
        if debug:
            logger.info("PhosphoScoring Debug Log")
            logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Input file: {in_file}")
            logger.info(f"Identification file: {id_file}")
            logger.info(f"Output file: {out_file}")
            logger.info(f"Fragment mass tolerance: {fragment_mass_tolerance} {fragment_mass_unit}")
            logger.info(f"Threads: {threads}")
            logger.info(f"Add decoys: {add_decoys}")
        
        # Process peptide identifications
        start_time = time.time()
        
        if threads == 1:
            # Sequential processing
            click.echo(f"[{time.strftime('%H:%M:%S')}] Processing {len(peptide_ids)} peptide identifications sequentially...")
            for i, pid in enumerate(peptide_ids):
                if debug and i % 100 == 0:
                    logger.info(f"Processing peptide identification {i+1}/{len(peptide_ids)}")
                process_peptide_identification(pid, exp, logger, add_decoys)
        else:
            # Parallel processing
            click.echo(f"[{time.strftime('%H:%M:%S')}] Processing {len(peptide_ids)} peptide identifications using {threads} threads...")
            
            # Use ThreadPoolExecutor for I/O bound operations
            with ThreadPoolExecutor(max_workers=threads) as executor:
                # Submit all tasks
                future_to_pid = {
                    executor.submit(_worker_process_pid, (pid, in_file, fragment_mass_tolerance, fragment_mass_unit, add_decoys, debug)): pid
                    for pid in peptide_ids
                }
                
                # Process completed tasks
                completed = 0
                for future in as_completed(future_to_pid):
                    try:
                        result = future.result()
                        completed += 1
                        if debug and completed % 100 == 0:
                            logger.info(f"Completed {completed}/{len(peptide_ids)} peptide identifications")
                    except Exception as exc:
                        pid = future_to_pid[future]
                        click.echo(f'Peptide identification generated an exception: {exc}')
                        if debug:
                            logger.error(f"Error processing peptide identification: {exc}")
                            logger.error(traceback.format_exc())
        
        # Save results
        click.echo(f"[{time.strftime('%H:%M:%S')}] Saving results to {out_file}")
        save_identifications(out_file, protein_ids, peptide_ids)
        
        end_time = time.time()
        processing_time = end_time - start_time
        click.echo(f"[{time.strftime('%H:%M:%S')}] Processing completed in {processing_time:.2f} seconds")
        
        if debug:
            logger.info(f"Processing completed in {processing_time:.2f} seconds")
            logger.info("PhosphoScoring Debug Log End")
        
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        if debug:
            logger.error(f"Error: {str(e)}")
            logger.error(traceback.format_exc())
        sys.exit(1)

def load_spectra(mzml_file):
    """Load MS/MS spectra"""
    print(f"[{time.strftime('%H:%M:%S')}] Loading spectra from {mzml_file}")
    exp = MSExperiment()
    FileHandler().loadExperiment(mzml_file, exp)
    print(f"Loaded {exp.size()} spectra")
    return exp

def load_identifications(idxml_file):
    """Load identification results"""
    print(f"[{time.strftime('%H:%M:%S')}] Loading identifications from {idxml_file}")
    protein_ids = []
    peptide_ids = []
    IdXMLFile().load(idxml_file, protein_ids, peptide_ids)
    print(f"Loaded {len(peptide_ids)} peptide identifications")
    return protein_ids, peptide_ids

def save_identifications(out_file, protein_ids, peptide_ids):
    """Save identification results"""
    print(f"[{time.strftime('%H:%M:%S')}] Saving {len(peptide_ids)} peptide identifications to {out_file}")
    IdXMLFile().store(out_file, protein_ids, peptide_ids)

def log_debug(log_file, debug):
    """Setup debug logging"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    if debug:
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def find_spectrum_by_mz(exp, target_mz, rt=None, ppm_tolerance=10):
    """
    Find spectrum by m/z and optionally retention time.
    
    Args:
        exp: MSExperiment containing spectra
        target_mz: Target m/z value
        rt: Retention time (optional)
        ppm_tolerance: PPM tolerance for m/z matching
    
    Returns:
        MSSpectrum or None if not found
    """
    best_spectrum = None
    best_mz_diff = float('inf')
    
    for spec in exp:
        spec_mz = spec.getPrecursors()[0].getMZ() if spec.getPrecursors() else 0
        
        # Calculate m/z difference in ppm
        mz_diff_ppm = abs(spec_mz - target_mz) / target_mz * 1e6
        
        if mz_diff_ppm <= ppm_tolerance:
            if rt is not None:
                spec_rt = spec.getRT()
                rt_diff = abs(spec_rt - rt)
                if rt_diff > 0.1:  # 0.1 second tolerance
                    continue
            
            if mz_diff_ppm < best_mz_diff:
                best_mz_diff = mz_diff_ppm
                best_spectrum = spec
    
    return best_spectrum

def _worker_get_exp(mzml_file):
    """Worker function to load experiment data"""
    exp = MSExperiment()
    FileHandler().loadExperiment(mzml_file, exp)
    return exp

def _worker_process_pid(task):
    """Worker function to process peptide identification"""
    pid, mzml_file, fragment_mass_tolerance, fragment_mass_unit, add_decoys, debug = task
    
    # Load experiment data for this worker
    exp = _worker_get_exp(mzml_file)
    
    # Create a logger for this worker
    logger = logging.getLogger(f"{__name__}_worker")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    try:
        return process_peptide_identification(pid, exp, logger, add_decoys)
    except Exception as e:
        logger.error(f"Error in worker processing: {e}")
        raise

def process_peptide_hit(hit, spectrum, logger):
    """
    Process a single peptide hit with AScore.
    
    Args:
        hit: PeptideHit to process
        spectrum: Associated MSSpectrum
        logger: Logger instance
    
    Returns:
        Modified PeptideHit with AScore annotations
    """
    if not spectrum:
        logger.warning("No spectrum found for peptide hit")
        # Set default values for missing spectrum
        hit.setMetaValue("AScore_pep_score", -1.0)
        return hit
    
    try:
        # Get peptide sequence
        sequence = hit.getSequence()
        original_seq_str = sequence.toString()
        
        # Check if peptide has phosphorylation sites
        has_phospho = any('p' in str(seq.getModification()) for seq in sequence)
        if not has_phospho:
            logger.debug(f"No phosphorylation sites found in {original_seq_str}")
            hit.setMetaValue("AScore_pep_score", -1.0)
            return hit
        
        # Configure AScore parameters
        ascore = AScore()
        ascore.setFragmentMassTolerance(0.05)  # Default tolerance
        ascore.setFragmentMassUnit('Da')
        
        # Execute AScore calculation
        scored_hit = ascore.compute(sequence, spectrum)
        
        # Get the new sequence after AScore reassignment
        new_sequence = scored_hit.getSequence()
        new_seq_str = new_sequence.toString()
        
        # Update the hit with new sequence if changed
        if new_seq_str != original_seq_str:
            hit.setSequence(new_sequence)
            logger.debug(f"Sequence updated: {original_seq_str} -> {new_seq_str}")
        
        # Get existing AScore values
        site_scores = []
        rank = 1
        while scored_hit.metaValueExists(f"AScore_{rank}"):
            score = float(scored_hit.getMetaValue(f"AScore_{rank}"))
            site_scores.append(score)
            rank += 1
        
        # Get the AScore_pep_score (highest weighted peptide score)
        ascore_pep_score = float(scored_hit.getMetaValue("AScore_pep_score"))
        
        # Determine overall score (best AScore value, which is the minimum)
        best_ascore = min(site_scores) if site_scores else ascore_pep_score
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"AScore computation complete:")
            logger.debug(f"  Original sequence: {original_seq_str}")
            logger.debug(f"  New sequence: {new_seq_str}")
            logger.debug(f"  Site scores: {site_scores}")
            logger.debug(f"  Best AScore: {best_ascore}")
        
        # Update meta values
        meta_fields = []
        meta_fields.append(("AScore_pep_score", ascore_pep_score))
        
        for i, score in enumerate(site_scores):
            meta_fields.append((f"AScore_{i}", score))
        
        # Apply meta values to hit
        for field_name, value in meta_fields:
            hit.setMetaValue(field_name, value)
        
        # Set the score to the best AScore value
        hit.setScore(best_ascore)
        
        return hit
        
    except Exception as e:
        logger.error(f"Error processing peptide hit: {e}")
        logger.error(traceback.format_exc())
        # Set default values on error
        hit.setMetaValue("AScore_pep_score", -1.0)
        return hit

def process_peptide_identification(pid, exp, logger, add_decoys=False):
    """
    Process a peptide identification with all its hits.
    
    Args:
        pid: PeptideIdentification to process
        exp: MSExperiment containing spectra
        logger: Logger instance
        add_decoys: Whether to include decoy sites
    
    Returns:
        Modified PeptideIdentification
    """
    try:
        # Get spectrum for this peptide identification
        target_mz = pid.getMZ()
        target_rt = pid.getRT()
        spectrum = find_spectrum_by_mz(exp, target_mz, target_rt)
        
        if not spectrum:
            logger.warning(f"No spectrum found for m/z={target_mz}, RT={target_rt}")
            # Set default values for all hits
            for hit in pid.getHits():
                hit.setMetaValue("AScore_pep_score", -1.0)
            return pid
        
        # Process each hit
        for hit in pid.getHits():
            process_peptide_hit(hit, spectrum, logger)
        
        return pid
        
    except Exception as e:
        logger.error(f"Error processing peptide identification: {e}")
        logger.error(traceback.format_exc())
        return pid

def process_peptide_identification_legacy(pid, exp, logger, add_decoys=False):
    """
    Legacy processing function for peptide identification.
    This version maintains compatibility with existing code.
    """
    try:
        # Get spectrum for this peptide identification
        target_mz = pid.getMZ()
        target_rt = pid.getRT()
        spectrum = find_spectrum_by_mz(exp, target_mz, target_rt)
        
        if not spectrum:
            logger.warning(f"No spectrum found for m/z={target_mz}, RT={target_rt}")
            return pid
        
        # Process each hit
        for hit in pid.getHits():
            try:
                # Get peptide sequence
                sequence = hit.getSequence()
                original_seq_str = sequence.toString()
                
                # Check if peptide has phosphorylation sites
                has_phospho = any('p' in str(seq.getModification()) for seq in sequence)
                if not has_phospho:
                    logger.debug(f"No phosphorylation sites found in {original_seq_str}")
                    hit.setMetaValue("AScore_pep_score", -1.0)
                    continue
                
                # Configure AScore parameters
                ascore = AScore()
                
                # Execute AScore calculation
                scored_hit = ascore.compute(sequence, spectrum)
                
                # Get the new sequence after AScore reassignment
                new_sequence = scored_hit.getSequence()
                new_seq_str = new_sequence.toString()
                
                # Update the hit with new sequence if changed
                if new_seq_str != original_seq_str:
                    hit.setSequence(new_sequence)
                    logger.debug(f"Sequence updated: {original_seq_str} -> {new_seq_str}")
                
                # Get existing AScore values
                site_scores = []
                rank = 1
                while scored_hit.metaValueExists(f"AScore_{rank}"):
                    score = float(scored_hit.getMetaValue(f"AScore_{rank}"))
                    site_scores.append(score)
                    rank += 1
                
                # Get the AScore_pep_score (highest weighted peptide score)
                ascore_pep_score = float(scored_hit.getMetaValue("AScore_pep_score"))
                
                # Determine overall score (best AScore value, which is the minimum)
                best_ascore = min(site_scores) if site_scores else ascore_pep_score
                
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"AScore computation complete:")
                    logger.debug(f"  Original sequence: {original_seq_str}")
                    logger.debug(f"  New sequence: {new_seq_str}")
                    logger.debug(f"  Site scores: {site_scores}")
                    logger.debug(f"  Best AScore: {best_ascore}")
                
                # Update meta values
                meta_fields = []
                meta_fields.append(("AScore_pep_score", ascore_pep_score))
                
                for i, score in enumerate(site_scores):
                    meta_fields.append((f"AScore_{i}", score))
                
                # Apply meta values to hit
                for field_name, value in meta_fields:
                    hit.setMetaValue(field_name, value)
                
                # Set the score to the best AScore value
                hit.setScore(best_ascore)
                
            except Exception as e:
                logger.error(f"Error processing peptide hit: {e}")
                logger.error(traceback.format_exc())
                # Set default values on error
                hit.setMetaValue("AScore_pep_score", -1.0)
        
        return pid
        
    except Exception as e:
        logger.error(f"Error processing peptide identification: {e}")
        logger.error(traceback.format_exc())
        return pid

if __name__ == '__main__':
    main()