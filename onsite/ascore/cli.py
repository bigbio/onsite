#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import logging
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import click
from pyopenms import *

from .ascore import AScore


@click.command()
@click.option(
    "-in",
    "--in-file",
    "in_file",
    required=True,
    help="Input mzML file path",
    type=click.Path(exists=True),
)
@click.option(
    "-id",
    "--id-file",
    "id_file",
    required=True,
    help="Input idXML file path",
    type=click.Path(exists=True),
)
@click.option(
    "-out",
    "--out-file",
    "out_file",
    required=True,
    help="Output idXML file path",
    type=click.Path(),
)
@click.option(
    "--fragment-mass-tolerance",
    "fragment_mass_tolerance",
    type=float,
    default=0.05,
    help="Fragment mass tolerance value (default: 0.05)",
)
@click.option(
    "--fragment-mass-unit",
    "fragment_mass_unit",
    type=click.Choice(["Da", "ppm"]),
    default="Da",
    help="Tolerance unit (default: Da)",
)
@click.option(
    "--threads",
    "threads",
    type=int,
    default=4,
    help="Number of parallel threads (default: 4)",
)
@click.option(
    "--debug", "debug", is_flag=True, help="Enable debug output and write debug log"
)
@click.option(
    "--add-decoys",
    "add_decoys",
    is_flag=True,
    default=False,
    help="Include A (PhosphoDecoy) as potential phosphorylation site",
)
def ascore(
    in_file,
    id_file,
    out_file,
    fragment_mass_tolerance,
    fragment_mass_unit,
    threads,
    debug,
    add_decoys,
):
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
            logger.info(
                f"Fragment mass tolerance: {fragment_mass_tolerance} {fragment_mass_unit}"
            )
            logger.info(f"Threads: {threads}")
            logger.info(f"Add decoys: {add_decoys}")
            logger.info(f"Total spectra: {exp.size()}")
            logger.info(f"Total identifications: {len(peptide_ids)}")

        # Processing statistics
        stats = {
            "total": len(peptide_ids),
            "processed": 0,
            "phospho": 0,
            "errors": 0
        }

        # Main processing loop
        start_time = time.time()
        processed_peptide_ids = []

        # Process each PeptideIdentification (optionally in parallel)
        if max(1, int(threads)) == 1:
            if debug:
                logger.info("Starting sequential processing")
            
            click.echo(
                f"[{time.strftime('%H:%M:%S')}] Processing {len(peptide_ids)} peptide identifications sequentially..."
            )
            
            for i, pid in enumerate(peptide_ids):
                try:
                    if debug and i % 100 == 0:
                        logger.info(f"Processing identification {i+1}/{len(peptide_ids)}")
                    
                    # Process using the original function signature
                    result = process_peptide_identification(
                        pid, exp, fragment_mass_tolerance, fragment_mass_unit, add_decoys, logger
                    )
                    
                    if result["status"] == "success":
                        processed_peptide_ids.append(result["new_pid"])
                        stats["processed"] += 1
                        phospho_count = len([h for h in result["new_pid"].getHits() 
                                          if "(Phospho)" in h.getSequence().toString()])
                        stats["phospho"] += phospho_count
                        
                        if debug:
                            logger.info(f"Successfully processed identification {i+1} with {phospho_count} phosphorylated hits")
                    else:
                        stats["errors"] += 1
                        error_msg = f"Error processing identification: {result['reason']}"
                        if debug:
                            logger.error(error_msg)
                except Exception as e:
                    stats["errors"] += 1
                    error_msg = f"Error processing identification: {str(e)}"
                    if debug:
                        logger.error(error_msg)
                    traceback.print_exc()
        else:
            workers = max(1, int(threads))
            click.echo(
                f"[{time.strftime('%H:%M:%S')}] Parallel execution with {workers} processes"
            )
            
            if debug:
                logger.info(f"Starting parallel processing with {workers} workers")

            # Build serializable tasks in dict format
            params = {
                "fragment_mass_tolerance": fragment_mass_tolerance,
                "fragment_mass_unit": fragment_mass_unit,
                "add_decoys": add_decoys,
            }
            tasks = []
            for idx, pid in enumerate(peptide_ids):
                hit_payloads = []
                for hit in pid.getHits():
                    seq_str = hit.getSequence().toString()
                    proforma = hit.getMetaValue("ProForma") if hit.metaValueExists("ProForma") else None
                    hit_payloads.append({"sequence": seq_str, "proforma": proforma})
                tasks.append({
                    "idx": idx,
                    "mzml_path": in_file,
                    "params": params,
                    "pid": {
                        "mz": pid.getMZ(),
                        "rt": pid.getRT(),
                        "hits": hit_payloads,
                    }
                })

            if debug:
                logger.info(f"Created {len(tasks)} parallel tasks")

            indexed_results = {}
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_worker_process_pid, t): t["idx"] for t in tasks}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        indexed_results[idx] = fut.result()
                    except Exception as e:
                        indexed_results[idx] = {"status": "error", "reason": str(e)}

            if debug:
                logger.info("Parallel processing completed, rebuilding results")

            # Rebuild PeptideIdentification objects in order
            for idx in range(len(peptide_ids)):
                res = indexed_results.get(idx, {"status": "error", "reason": "unknown"})
                pid_src = peptide_ids[idx]
                if res["status"] == "success":
                    new_pid = PeptideIdentification(pid_src)
                    new_pid.setScoreType("PhosphoScore")
                    new_pid.setHigherScoreBetter(True)
                    new_pid.setSignificanceThreshold(0.0)

                    new_hits = []
                    for hit_src, hit_res in zip(pid_src.getHits(), res["hits"]):
                        new_hit = PeptideHit(hit_src)
                        new_hit.setSequence(AASequence.fromString(hit_res["new_sequence"]))
                        # Clear managed metas then set in order
                        for k in [
                            "search_engine_sequence", "regular_phospho_count", "phospho_decoy_count",
                            "AScore_pep_score"
                        ]:
                            if new_hit.metaValueExists(k):
                                try:
                                    new_hit.removeMetaValue(k)
                                except Exception:
                                    pass
                        i = 1
                        while new_hit.metaValueExists(f"AScore_{i}"):
                            try:
                                new_hit.removeMetaValue(f"AScore_{i}")
                            except Exception:
                                pass
                            i += 1

                        if new_hit.metaValueExists("ProForma"):
                            try:
                                new_hit.removeMetaValue("ProForma")
                            except Exception:
                                pass

                        for k, v in hit_res["meta_fields"]:
                            new_hit.setMetaValue(k, v)
                        new_hit.setScore(hit_res["best_ascore"])
                        new_hits.append(new_hit)

                    new_pid.setHits(new_hits)
                    processed_peptide_ids.append(new_pid)
                    stats["processed"] += 1
                    phospho_count = len([h for h in new_pid.getHits() if "(Phospho)" in h.getSequence().toString()])
                    stats["phospho"] += phospho_count
                    
                    if debug:
                        logger.info(f"Rebuilt identification {idx+1} with {phospho_count} phosphorylated hits")
                else:
                    stats["errors"] += 1
                    error_msg = f"Error processing identification: {res.get('reason', 'unknown')}"
                    if debug:
                        logger.error(error_msg)
        
        # Generate final report
        elapsed = time.time() - start_time
        click.echo(f"\nProcessing Complete:")
        click.echo(f"  Total identifications: {stats['total']}")
        click.echo(f"  Successfully processed: {stats['processed']}")
        click.echo(f"  Phosphorylated peptides: {stats['phospho']}")
        click.echo(f"  Processing errors: {stats['errors']}")
        click.echo(f"  Time elapsed: {elapsed:.2f} seconds")
        click.echo(f"  Processing speed: {stats['processed']/elapsed:.2f} IDs/second")
        if debug:
            click.echo(f"  Debug log saved to: {log_file}")
        
        if debug:
            logger.info("Processing completed successfully")
            logger.info(f"Final statistics: {stats}")
            logger.info(f"Total time: {elapsed:.2f} seconds")
            logger.info(f"Processing speed: {stats['processed']/elapsed:.2f} IDs/second")

        # Save results
        click.echo(f"[{time.strftime('%H:%M:%S')}] Saving results to {out_file}")
        save_identifications(out_file, protein_ids, processed_peptide_ids)

    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        if debug:
            logger.error(f"Error: {str(e)}")
            logger.error(traceback.format_exc())
        traceback.print_exc()
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
    """Save results"""
    try:
        # Ensure all critical metadata fields exist
        for pid in peptide_ids:
            # Set identification-level attributes
            pid.setScoreType("PhosphoScore")
            pid.setHigherScoreBetter(True)
            pid.setSignificanceThreshold(0.0)
            
            for hit in pid.getHits():
                seq_str = hit.getSequence().toString()
                
                # Ensure all required metadata fields exist
                if not hit.metaValueExists("search_engine_sequence"):
                    hit.setMetaValue("search_engine_sequence", seq_str)
                if not hit.metaValueExists("regular_phospho_count"):
                    hit.setMetaValue("regular_phospho_count", seq_str.count("(Phospho)"))
                if not hit.metaValueExists("phospho_decoy_count"):
                    hit.setMetaValue("phospho_decoy_count", seq_str.count("(PhosphoDecoy)"))
                
                # Then set AScore fields
                if not hit.metaValueExists("AScore_pep_score"):
                    hit.setMetaValue("AScore_pep_score", hit.getScore())

        # Write with XML format validation
        IdXMLFile().store(out_file, protein_ids, peptide_ids)
        print(f"Successfully saved {len(peptide_ids)} identifications to {out_file}")
    except Exception as e:
        print(f"Critical error saving results: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


def log_debug(log_file, enabled):
    """Initialize debug logging only when enabled"""
    logger = logging.getLogger("debug_logger")
    # Clear existing handlers to avoid duplicates on repeated runs
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    if enabled:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Ensure propagation is enabled
        logger.propagate = False
    else:
        # Disable logging output when not in debug mode
        logger.setLevel(logging.CRITICAL)
        # Add a null handler to prevent "No handler found" warnings
        logger.addHandler(logging.NullHandler())
    return logger


def find_spectrum_by_mz(exp, target_mz, rt=None, ppm_tolerance=10):
    """Optimized spectrum matching with caching"""
    # Binary search for optimized spectrum matching
    if not hasattr(find_spectrum_by_mz, "spectrum_cache"):
        find_spectrum_by_mz.spectrum_cache = {}
        find_spectrum_by_mz.spectrum_list = []
        
        # Preprocess all MS2 spectra
        for spec in exp:
            if spec.getMSLevel() == 2 and spec.getPrecursors():
                mz = spec.getPrecursors()[0].getMZ()
                find_spectrum_by_mz.spectrum_list.append((mz, spec))
        
        # Sort by m/z
        find_spectrum_by_mz.spectrum_list.sort(key=lambda x: x[0])
    
    # Binary search for the closest m/z
    left, right = 0, len(find_spectrum_by_mz.spectrum_list) - 1
    best_match = None
    min_diff = float("inf")
    
    while left <= right:
        mid = (left + right) // 2
        mz, spec = find_spectrum_by_mz.spectrum_list[mid]
        diff = abs(mz - target_mz)
        
        if diff < min_diff:
            min_diff = diff
            best_match = spec
        
        if mz < target_mz:
            left = mid + 1
        else:
            right = mid - 1
    
    return best_match


# ----------------------- Multiprocessing worker utilities -----------------------
_WORKER_EXP = None


def _worker_get_exp(mzml_file):
    global _WORKER_EXP
    if _WORKER_EXP is None:
        exp = MSExperiment()
        FileHandler().loadExperiment(mzml_file, exp)
        # Warm up spectrum index inside the worker for faster lookups
        if exp.size() > 0:
            # Rebuild local cache for find_spectrum_by_mz in this process
            if hasattr(find_spectrum_by_mz, "spectrum_list"):
                delattr(find_spectrum_by_mz, "spectrum_list")
            _ = find_spectrum_by_mz(exp, 0.0, None)
        _WORKER_EXP = exp
    return _WORKER_EXP


def _worker_process_pid(task):
    try:
        mzml_path = task["mzml_path"]
        pid_info = task["pid"]
        params = task["params"]

        exp = _worker_get_exp(mzml_path)

        # Find spectrum
        spectrum = find_spectrum_by_mz(exp, pid_info["mz"], pid_info.get("rt"))
        if spectrum is None:
            return {"status": "error", "reason": "spectrum_not_found"}

        # Configure AScore instance per task
        ascore = AScore()
        ascore.fragment_mass_tolerance_ = params["fragment_mass_tolerance"]
        ascore.fragment_tolerance_ppm_ = params["fragment_mass_unit"] == "ppm"
        ascore.setAddDecoys(params.get("add_decoys", False))

        results = []
        for hit_info in pid_info["hits"]:
            # Rebuild hit from sequence
            seq = AASequence.fromString(hit_info["sequence"])
            hit = PeptideHit()
            hit.setSequence(seq)
            if "proforma" in hit_info and hit_info["proforma"] is not None:
                hit.setMetaValue("ProForma", hit_info["proforma"])

            scored_hit = ascore.compute(hit, spectrum)

            # Extract metas in deterministic order
            new_seq_str = scored_hit.getSequence().toString()

            # Collect site scores
            site_scores = []
            rank = 1
            while scored_hit.metaValueExists(f"AScore_{rank}"):
                site_scores.append(float(scored_hit.getMetaValue(f"AScore_{rank}")))
                rank += 1

            ascore_pep_score = float(scored_hit.getMetaValue("AScore_pep_score")) if scored_hit.metaValueExists("AScore_pep_score") else -1.0
            best_ascore = min(site_scores) if site_scores else -1.0

            meta_fields = []
            meta_fields.append(("search_engine_sequence", hit_info["sequence"]))
            meta_fields.append(("regular_phospho_count", new_seq_str.count("(Phospho)")))
            meta_fields.append(("phospho_decoy_count", new_seq_str.count("(PhosphoDecoy)")))
            meta_fields.append(("AScore_pep_score", ascore_pep_score))
            for i, score in enumerate(site_scores, 1):
                meta_fields.append((f"AScore_{i}", score))
            if scored_hit.metaValueExists("ProForma"):
                meta_fields.append(("ProForma", scored_hit.getMetaValue("ProForma")))

            results.append({
                "new_sequence": new_seq_str,
                "best_ascore": best_ascore,
                "meta_fields": meta_fields
            })

        return {"status": "success", "hits": results}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def process_peptide_identification(pid, exp, fragment_mass_tolerance, fragment_mass_unit, add_decoys, logger):
    """Process a single peptide identification with error handling"""
    try:
        # Create new PeptideIdentification
        new_pid = PeptideIdentification(pid)
        new_pid.setScoreType("PhosphoScore")
        new_pid.setHigherScoreBetter(True)
        new_pid.setSignificanceThreshold(0.0)
        
        if logger and logger.isEnabledFor(logging.INFO):
            logger.info(f"Processing identification - MZ: {pid.getMZ():.4f}, RT: {pid.getRT():.2f}")
        
        # Find spectrum
        spectrum = find_spectrum_by_mz(exp, pid.getMZ(), pid.getRT())
        if not spectrum:
            error_msg = f"Error processing identification: spectrum_not_found for MZ {pid.getMZ()}"
            if logger and logger.isEnabledFor(logging.ERROR):
                logger.error(error_msg)
            return {"status": "error", "reason": "spectrum_not_found"}
        
        if logger and logger.isEnabledFor(logging.INFO):
            logger.info(f"Found spectrum: {spectrum.getNativeID()}")
        
        # Configure AScore instance
        ascore = AScore()
        ascore.fragment_mass_tolerance_ = fragment_mass_tolerance
        ascore.fragment_tolerance_ppm_ = (fragment_mass_unit == "ppm")
        ascore.setAddDecoys(add_decoys)
        
        # Process peptide hits
        scored_peptides = []
        for hit in pid.getHits():
            # Create new PeptideHit
            new_hit = PeptideHit(hit)
            
            # Store original sequence
            original_seq_str = new_hit.getSequence().toString()
            new_hit.setMetaValue("search_engine_sequence", original_seq_str)
            
            # Quick check for phosphorylation sites
            if "(Phospho)" not in original_seq_str and "(PhosphoDecoy)" not in original_seq_str:
                new_hit.setScore(-1.0)
                new_hit.setMetaValue("AScore_pep_score", -1.0)
                if logger and logger.isEnabledFor(logging.INFO):
                    logger.info(f"Skipping non-phosphorylated peptide: {original_seq_str}")
                scored_peptides.append(new_hit)
                continue
            
            if logger and logger.isEnabledFor(logging.INFO):
                logger.info(f"Processing phosphorylated peptide: {original_seq_str}")
            
            if logger and logger.isEnabledFor(logging.INFO):
                logger.info(f"Computing AScore for: {original_seq_str} (tolerance: {fragment_mass_tolerance} {fragment_mass_unit})")
            
            # Execute AScore calculation
            scored_hit = ascore.compute(new_hit, spectrum)
            
            # Get the new sequence after AScore reassignment
            new_seq_str = scored_hit.getSequence().toString()
            
            # Set the new sequence as the main sequence
            new_hit.setSequence(scored_hit.getSequence())
            
            # Count phosphorylation sites
            phospho_count = new_seq_str.count("(Phospho)")
            
            # Extract site-specific scores
            site_scores = []
            rank = 1
            
            # Get existing AScore values
            while scored_hit.metaValueExists(f"AScore_{rank}"):
                score = float(scored_hit.getMetaValue(f"AScore_{rank}"))
                site_scores.append(score)
                rank += 1
                
            # If we have fewer scores than phospho sites, add default scores
            while len(site_scores) < phospho_count:
                site_scores.append(1000.0)  # Default high confidence score
            
            # Get the AScore_pep_score (highest weighted peptide score)
            ascore_pep_score = float(scored_hit.getMetaValue("AScore_pep_score")) if scored_hit.metaValueExists("AScore_pep_score") else -1.0
            
            # Determine overall score (best AScore value, which is the minimum)
            best_ascore = min(site_scores) if site_scores else -1.0
            
            if logger and logger.isEnabledFor(logging.INFO):
                logger.info(f"AScore computation complete:")
                logger.info(f"  Original: {original_seq_str}")
                logger.info(f"  New: {new_seq_str}")
                logger.info(f"  Phospho sites: {phospho_count}")
                logger.info(f"  Site scores: {site_scores}")
                logger.info(f"  Best AScore: {best_ascore}")
            
            # Build meta fields in the required order
            meta_fields = []
            meta_fields.append(("search_engine_sequence", original_seq_str))
            meta_fields.append(("regular_phospho_count", phospho_count))
            meta_fields.append(("phospho_decoy_count", new_seq_str.count("(PhosphoDecoy)")))
            meta_fields.append(("AScore_pep_score", ascore_pep_score))
            for i, score in enumerate(site_scores, 1):
                meta_fields.append((f"AScore_{i}", score))
            if scored_hit.metaValueExists("ProForma"):
                meta_fields.append(("ProForma", scored_hit.getMetaValue("ProForma")))

            # Preserve existing ProForma
            proforma_value = None
            if scored_hit.metaValueExists("ProForma"):
                proforma_value = scored_hit.getMetaValue("ProForma")
            elif new_hit.metaValueExists("ProForma"):
                proforma_value = new_hit.getMetaValue("ProForma")

            # Remove managed fields to re-insert them
            for k in [
                "search_engine_sequence",
                "regular_phospho_count",
                "phospho_decoy_count",
                "AScore_pep_score",
            ]:
                if new_hit.metaValueExists(k):
                    try:
                        new_hit.removeMetaValue(k)
                    except Exception:
                        # Fallback: overwrite later if removal unsupported
                        pass

            # Remove any existing site-level AScore_* fields
            i = 1
            while new_hit.metaValueExists(f"AScore_{i}"):
                try:
                    new_hit.removeMetaValue(f"AScore_{i}")
                except Exception:
                    pass
                i += 1

            # Ensure ProForma will be appended last
            if new_hit.metaValueExists("ProForma"):
                try:
                    new_hit.removeMetaValue("ProForma")
                except Exception:
                    pass

            # Apply managed metadata (others are preserved)
            for key, value in meta_fields:
                new_hit.setMetaValue(key, value)
            # Re-append preserved ProForma if available
            if proforma_value is not None and not any(k == "ProForma" for k, _ in meta_fields):
                new_hit.setMetaValue("ProForma", proforma_value)
            
            # Set the score to the best AScore value
            new_hit.setScore(best_ascore)
            
            if logger and logger.isEnabledFor(logging.INFO):
                logger.info(f"Successfully processed: {original_seq_str} -> {new_seq_str}")
            
            scored_peptides.append(new_hit)
        
        # Set new hits
        new_pid.setHits(scored_peptides)
        
        if logger and logger.isEnabledFor(logging.INFO):
            logger.info(f"Successfully processed identification with {len(scored_peptides)} hits")
        
        return {"status": "success", "new_pid": new_pid}
        
    except Exception as e:
        error_msg = f"Error processing identification: {str(e)}"
        if logger and logger.isEnabledFor(logging.ERROR):
            logger.error(error_msg)
        return {"status": "error", "reason": str(e)}


def main():
    """Entry point for standalone AScore CLI."""
    ascore()


if __name__ == "__main__":
    main()
