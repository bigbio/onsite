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

from .phosphors import calculate_phospho_localization_compomics_style


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
    default=1,
    help="Number of parallel processes (default: 1)",
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
def phosphors(
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
    Phosphorylation site localization scoring tool using PhosphoRS algorithm.

    This tool processes MS/MS spectra and peptide identifications to localize
    phosphorylation sites using the PhosphoRS algorithm.
    """
    try:
        # Initialize processing pipeline
        exp = load_spectra(in_file)
        protein_ids, peptide_ids = load_identifications(id_file)
        # Prime spectrum cache
        if peptide_ids:
            _ = find_spectrum_by_mz(exp, peptide_ids[0].getMZ(), peptide_ids[0].getRT())

        # Initialize debug log (only when --debug)
        log_file = f"{out_file}.debug.log"
        logger = log_debug(log_file, debug)
        if debug:
            logger.info("PhosphoRSScoring Debug Log")
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

        start_time = time.time()
        processed_peptide_ids = []

        # Sequential or parallel processing
        if max(1, int(threads)) == 1:
            click.echo(
                f"[{time.strftime('%H:%M:%S')}] Processing {len(peptide_ids)} peptide identifications sequentially..."
            )
            
            for i, pid in enumerate(peptide_ids):
                try:
                    result = process_peptide_identification(
                        pid, exp, fragment_mass_tolerance, fragment_mass_unit, add_decoys, logger
                    )
                    if result["status"] == "success":
                        processed_peptide_ids.append(result["new_pid"])
                        stats["processed"] += 1
                        stats["phospho"] += len([h for h in result["new_pid"].getHits() 
                                              if "(Phospho)" in h.getSequence().toString()])
                    else:
                        stats["errors"] += 1
                        if debug:
                            logger.error(f"Error processing identification: {result['reason']}")
                except Exception as e:
                    stats["errors"] += 1
                    if debug:
                        logger.error(f"Error processing identification: {str(e)}")
                    traceback.print_exc()
        else:
            workers = max(1, int(threads))
            click.echo(
                f"[{time.strftime('%H:%M:%S')}] Parallel execution with {workers} processes"
            )
            if debug:
                logger.info(f"Starting parallel processing with {workers} workers")

            params = {
                "fragment_mass_tolerance": fragment_mass_tolerance,
                "fragment_mass_unit": fragment_mass_unit,
                "add_decoys": bool(add_decoys),
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

            indexed_results = {}
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_worker_process_pid, t): t["idx"] for t in tasks}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        indexed_results[idx] = fut.result()
                    except Exception as e:
                        indexed_results[idx] = {"status": "error", "reason": str(e)}

            # Rebuild results in order
            for idx in range(len(peptide_ids)):
                res = indexed_results.get(idx, {"status": "error", "reason": "unknown"})
                pid_src = peptide_ids[idx]
                if res["status"] == "success":
                    new_pid = PeptideIdentification(pid_src)
                    new_pid.setScoreType("PhosphoRSScore")
                    new_pid.setHigherScoreBetter(False)
                    new_pid.setSignificanceThreshold(0.0)

                    new_hits = []
                    for hit_src, hit_res in zip(pid_src.getHits(), res["hits"]):
                        if hit_res.get("status") != "success":
                            # Preserve original hit with -1 score when failed
                            failed_hit = PeptideHit(hit_src)
                            failed_hit.setScore(-1.0)
                            new_hits.append(failed_hit)
                            continue

                        new_hit = PeptideHit(hit_src)
                        new_hit.setSequence(
                            AASequence.fromString(hit_res["new_sequence"])
                        )
                        # Clear managed metas
                        for k in [
                            "search_engine_sequence",
                            "regular_phospho_count",
                            "phospho_decoy_count",
                            "PhosphoRS_pep_score",
                            "PhosphoRS_site_probs",
                            "SpecEValue_score",
                        ]:
                            if new_hit.metaValueExists(k):
                                try:
                                    new_hit.removeMetaValue(k)
                                except Exception:
                                    pass
                        if new_hit.metaValueExists("ProForma"):
                            try:
                                new_hit.removeMetaValue("ProForma")
                            except Exception:
                                pass
                        for k, v in hit_res["meta_fields"]:
                            new_hit.setMetaValue(k, v)
                        new_hit.setScore(float(hit_res["score"]))
                        new_hits.append(new_hit)

                    new_pid.setHits(new_hits)
                    processed_peptide_ids.append(new_pid)
                    stats["processed"] += 1
                    stats["phospho"] += len(
                        [
                            h
                            for h in new_pid.getHits()
                            if "(Phospho)" in h.getSequence().toString()
                        ]
                    )
                else:
                    stats["errors"] += 1
                    if debug:
                        logger.error(
                            f"Error processing identification: {res.get('reason', 'unknown')}"
                        )

        # Report
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
            logger.info("Processing completed successfully")
            logger.info(f"Final statistics: {stats}")
            logger.info(f"Total time: {elapsed:.2f} seconds")

        # Save results
        click.echo(f"[{time.strftime("%H:%M:%S")}] Saving results to {out_file}")
        save_identifications(out_file, protein_ids, processed_peptide_ids)

    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Fatal error: {str(e)}")
        if debug:
            logger.error(f"Error: {str(e)}")
            logger.error(traceback.format_exc())
        traceback.print_exc()
        sys.exit(1)

def load_spectra(mzml_file):
    """Load MS/MS spectra with progress feedback"""
    print(f"[{time.strftime('%H:%M:%S')}] Loading spectra from {mzml_file}")
    exp = MSExperiment()
    FileHandler().loadExperiment(mzml_file, exp)
    print(f"Loaded {exp.size()} spectra")
    return exp


def load_identifications(idxml_file):
    """Load identification results with metadata validation"""
    print(f"[{time.strftime('%H:%M:%S')}] Loading identifications from {idxml_file}")
    protein_ids = []
    peptide_ids = []
    IdXMLFile().load(idxml_file, protein_ids, peptide_ids)
    print(f"Loaded {len(peptide_ids)} peptide identifications")
    return protein_ids, peptide_ids


def save_identifications(out_file, protein_ids, peptide_ids):
    """Save results with complete metadata preservation"""
    try:
        for pid in peptide_ids:
            pid.setScoreType("PhosphoRSScore")
            pid.setHigherScoreBetter(False)
            pid.setSignificanceThreshold(0.0)
            for hit in pid.getHits():
                seq_str = hit.getSequence().toString()
                if not hit.metaValueExists("search_engine_sequence"):
                    hit.setMetaValue("search_engine_sequence", seq_str)
                if not hit.metaValueExists("regular_phospho_count"):
                    # Count S/T/Y(Phospho) occurrences
                    regular_count = sum(
                        seq_str.count(f"{aa}(Phospho)") for aa in ["S", "T", "Y"]
                    )
                    hit.setMetaValue("regular_phospho_count", regular_count)
                if not hit.metaValueExists("phospho_decoy_count"):
                    hit.setMetaValue(
                        "phospho_decoy_count", seq_str.count("(PhosphoDecoy)")
                    )
                if not hit.metaValueExists("PhosphoRS_pep_score"):
                    hit.setMetaValue("PhosphoRS_pep_score", float(hit.getScore()))
                if not hit.metaValueExists("PhosphoRS_site_probs"):
                    hit.setMetaValue("PhosphoRS_site_probs", "{}")
                if not hit.metaValueExists("SpecEValue_score") and hit.metaValueExists(
                    "MS:1002052"
                ):
                    hit.setMetaValue(
                        "SpecEValue_score", float(hit.getMetaValue("MS:1002052"))
                    )
        IdXMLFile().store(out_file, protein_ids, peptide_ids)
        print(f"Successfully saved {len(peptide_ids)} identifications to {out_file}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


def find_spectrum_by_mz(exp, target_mz, rt=None, ppm_tolerance=10):
    """Optimized spectrum matching with caching"""
    # Binary search for optimized spectrum matching
    cache = getattr(find_spectrum_by_mz, "spectrum_cache", {})
    exp_key = id(exp)
    if exp_key not in cache:
        cache[exp_key] = {"spectra": []}
        find_spectrum_by_mz.spectrum_cache = cache
        # Preprocess all MS2 spectra
        for spec in exp:
            if spec.getMSLevel() == 2 and spec.getPrecursors():
                mz = spec.getPrecursors()[0].getMZ()
                cache[exp_key]["spectra"].append((mz, spec))

        # Sort by m/z
        cache[exp_key]["spectra"].sort(key=lambda x: x[0])

    # Binary search for the closest m/z
    spectra = cache[exp_key]["spectra"]
    left, right = 0, len(spectra) - 1
    best_match = None
    min_diff = float("inf")

    while left <= right:
        mid = (left + right) // 2
        mz, spec = spectra[mid]
        diff = abs(mz - target_mz)

        if diff < min_diff:
            min_diff = diff
            best_match = spec

        if mz < target_mz:
            left = mid + 1
        else:
            right = mid - 1

    if best_match is None:
        return None
    # Enforce ppm tolerance and optional RT tolerance (~0.1 s default behavior)
    best_mz = best_match.getPrecursors()[0].getMZ()
    ppm = abs(best_mz - target_mz) / max(target_mz, 1e-12) * 1e6
    if ppm > ppm_tolerance:
        return None
    if rt is not None:
        if abs(best_match.getRT() - rt) > 0.1:
            return None
    return best_match


# ----------------------- Multiprocessing worker utilities -----------------------
_WORKER_EXP = None


def _worker_get_exp(mzml_file):
    global _WORKER_EXP
    if _WORKER_EXP is None:
        exp = MSExperiment()
        FileHandler().loadExperiment(mzml_file, exp)
        # Warm up spectrum cache in worker
        if exp.size() > 0:
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
        spectrum = find_spectrum_by_mz(exp, pid_info["mz"], pid_info.get("rt"))
        if spectrum is None:
            return {"status": "error", "reason": "spectrum_not_found"}

        results = []
        for hit_info in pid_info["hits"]:
            seq = AASequence.fromString(hit_info["sequence"])
            hit = PeptideHit()
            hit.setSequence(seq)
            if hit_info.get("proforma") is not None:
                hit.setMetaValue("ProForma", hit_info["proforma"])

            site_probs, isomer_list = calculate_phospho_localization_compomics_style(
                hit,
                spectrum,
                fragment_tolerance=params["fragment_mass_tolerance"],
                fragment_method_ppm=(params["fragment_mass_unit"] == "ppm"),
                add_decoys=params.get("add_decoys", False),
            )

            if site_probs is None or isomer_list is None:
                results.append({"status": "no_result"})
                continue

            best_isomer = min(isomer_list, key=lambda x: x[1])
            final_score = float(best_isomer[1])
            new_sequence = best_isomer[0]

            seq_str = hit.getSequence().toString()
            regular_count = sum(
                seq_str.count(f"{aa}(Phospho)") for aa in ["S", "T", "Y"]
            )
            decoy_count = seq_str.count("(PhosphoDecoy)")
            simple_site_probs = {k: float(v) for k, v in site_probs.items()}

            meta_fields = []
            meta_fields.append(("search_engine_sequence", seq_str))
            meta_fields.append(("regular_phospho_count", regular_count))
            meta_fields.append(("phospho_decoy_count", decoy_count))
            meta_fields.append(("PhosphoRS_pep_score", final_score))
            meta_fields.append(("PhosphoRS_site_probs", str(simple_site_probs)))

            # strict comparison removed
            if hit.metaValueExists("MS:1002052"):
                meta_fields.append(
                    ("SpecEValue_score", float(hit.getMetaValue("MS:1002052")))
                )
            if hit.metaValueExists("ProForma"):
                meta_fields.append(("ProForma", hit.getMetaValue("ProForma")))

            results.append(
                {
                    "status": "success",
                    "new_sequence": new_sequence,
                    "score": final_score,
                    "meta_fields": meta_fields,
                }
            )

        return {"status": "success", "hits": results}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def process_peptide_identification(pid, exp, fragment_mass_tolerance, fragment_mass_unit, add_decoys, logger):
    """Process a single peptide identification with error handling"""
    try:
        # Create new PeptideIdentification object
        new_pid = PeptideIdentification(pid)
        new_pid.setScoreType("PhosphoRSScore")
        new_pid.setHigherScoreBetter(False)  # Lower scores are better in PhosphoRS
        new_pid.setSignificanceThreshold(0.0)

        # Find corresponding spectrum
        spectrum = find_spectrum_by_mz(exp, pid.getMZ(), pid.getRT())
        if not spectrum:
            logger.error(
                f"Error processing identification: spectrum_not_found for MZ {pid.getMZ()}"
            )
            return {"status": "error", "reason": "spectrum_not_found"}

        # Process each peptide hit
        scored_peptides = []
        for hit in pid.getHits():
            # Create new PeptideHit object
            new_hit = PeptideHit(hit)
            
            # Store original sequence
            original_seq_str = new_hit.getSequence().toString()
            new_hit.setMetaValue("search_engine_sequence", original_seq_str)
            
            # Check for phosphorylation sites
            has_phospho = False
            for aa in ["S", "T", "Y", "A"]:
                if f"{aa}(Phospho)" in original_seq_str or f"{aa}(PhosphoDecoy)" in original_seq_str:
                    has_phospho = True
                    break
                    
            if not has_phospho:
                new_hit.setScore(-1.0)
                new_hit.setMetaValue("PhosphoRS_pep_score", float(-1.0))
                scored_peptides.append(new_hit)
                continue
            
            # Execute PhosphoRS calculation
            site_probs, isomer_list = calculate_phospho_localization_compomics_style(
                new_hit,
                spectrum,
                fragment_tolerance=fragment_mass_tolerance,
                fragment_method_ppm=(fragment_mass_unit == "ppm"),
                add_decoys=add_decoys
            )

            if site_probs is None or isomer_list is None:
                new_hit.setScore(-1.0)
                new_hit.setMetaValue("PhosphoRS_pep_score", float(-1.0))
                scored_peptides.append(new_hit)
                continue

            # Get the best scoring isomer
            best_isomer = min(isomer_list, key=lambda x: x[1])
            final_score = best_isomer[1]
            new_sequence = best_isomer[0]
            new_hit.setScore(final_score)
            
            # Count phosphorylation sites
            regular_count = sum(original_seq_str.count(f"{aa}(Phospho)") for aa in ["S","T","Y"])
            decoy_count = original_seq_str.count("(PhosphoDecoy)")
            
            # Convert site_probs to simple float values
            simple_site_probs = {k: float(v) for k, v in site_probs.items()}
            
            # Set all required metadata fields
            new_hit.setMetaValue("search_engine_sequence", original_seq_str)
            new_hit.setMetaValue("regular_phospho_count", regular_count)
            new_hit.setMetaValue("phospho_decoy_count", decoy_count)
            new_hit.setMetaValue("PhosphoRS_pep_score", float(final_score))
            new_hit.setMetaValue("PhosphoRS_site_probs", str(simple_site_probs))
            
            # Save MSGF score
            if new_hit.metaValueExists("MS:1002052"):
                new_hit.setMetaValue("SpecEValue_score", float(new_hit.getMetaValue("MS:1002052")))
            
            scored_peptides.append(new_hit)

        # Set new hits
        new_pid.setHits(scored_peptides)

        return {"status": "success", "new_pid": new_pid}

    except Exception as e:
        logger.error(f"Error processing identification: {str(e)}")
        return {"status": "error", "reason": str(e)}


def log_debug(log_file, enabled):
    """Initialize debug logging only when enabled"""
    logger = logging.getLogger("debug_logger")
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
        logger.propagate = False
    else:
        logger.setLevel(logging.CRITICAL)
        logger.addHandler(logging.NullHandler())
    return logger


def main():
    """Entry point for standalone PhosphoRS CLI."""
    phosphors()


if __name__ == "__main__":
    main()
