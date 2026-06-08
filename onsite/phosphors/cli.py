#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import logging
import traceback
from typing import Dict, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import click
import pandas as pd
from pyopenms import AASequence, MSExperiment, FileHandler, PeptideHit
from onsite.idparquet import unimod_to_pyopenms_notation, pyopenms_to_unimod_notation
from onsite.idparquet import save_dataframes as _save_df

from .phosphors import (
    calculate_phospho_localization_compomics_style,
    site_deltas_from_isomers,
)


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
    help="Input idparquet directory path",
    type=click.Path(exists=True),
)
@click.option(
    "-out",
    "--out-file",
    "out_file",
    required=True,
    help="Output idparquet directory path",
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
@click.option(
    "--compute-all-scores",
    "compute_all_scores",
    is_flag=True,
    default=False,
    help="Run all three algorithms (AScore, PhosphoRS, LucXor) and merge results",
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
    compute_all_scores,
):
    """
    Phosphorylation site localization scoring tool using PhosphoRS algorithm.

    This tool processes MS/MS spectra and peptide identifications to localize
    phosphorylation sites using the PhosphoRS algorithm.
    """
    # If compute_all_scores is enabled, delegate to the unified handler
    if compute_all_scores:
        from onsite.onsitec import run_all_algorithms_from_single_cli
        return run_all_algorithms_from_single_cli(
            in_file=in_file,
            id_file=id_file,
            out_file=out_file,
            fragment_mass_tolerance=fragment_mass_tolerance,
            fragment_mass_unit=fragment_mass_unit,
            threads=threads,
            debug=debug,
            add_decoys=add_decoys,
        )
    
    try:
        # Initialize processing pipeline
        exp = load_spectra(in_file)
        psms_df, proteins_df = load_identifications(id_file)

        # Build scan number to spectrum mapping for efficient lookup
        scan_map = build_scan_to_spectrum_map(exp)
        click.echo(f"Built scan number mapping: {len(scan_map)} MS2 spectra with scan numbers")

        # Prime spectrum cache
        if len(psms_df) > 0:

            row0 = psms_df.iloc[0]
            _ = find_spectrum_by_mz(
                exp,
                float(row0.get("observed_mz", row0.get("calculated_mz", 0.0))),
                float(row0.get("rt", 0.0)),
            )

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
            logger.info(f"Total identifications: {psms_df['peptide_identification_index'].nunique()}")

        grouped = list(psms_df.sort_values("peptide_identification_index").groupby("peptide_identification_index"))

        # Processing statistics
        stats = {"total": len(grouped), "processed": 0, "phospho": 0, "errors": 0}

        start_time = time.time()
        result_rows: List[Dict] = []

        # Sequential or parallel processing
        if max(1, int(threads)) == 1:
            click.echo(
                f"[{time.strftime('%H:%M:%S')}] Processing {len(grouped)} peptide identifications sequentially..."
            )

            for _, (_, group_df) in enumerate(grouped):
                try:
                    res = _process_psm_group(
                        group_df, exp, fragment_mass_tolerance, fragment_mass_unit, add_decoys, logger, scan_map
                    )
                    if res["status"] == "success":
                        result_rows.extend(res["rows"])
                        stats["processed"] += 1
                        stats["phospho"] += res["phospho_count"]
                    else:
                        stats["errors"] += 1
                except Exception as e:
                    stats["errors"] += 1
                    if debug:
                        logger.error(f"Error processing identification: {str(e)}")
                    traceback.print_exc()
        else:
            workers = max(1, int(threads))
            click.echo(
                f"[{time.strftime('%H:%M:%S')}] Parallel execution with {workers} threads"
            )
            if debug:
                logger.info(f"Starting parallel processing with {workers} workers")

            # Build tasks - with threads we can pass objects directly (shared memory)
            params = {
                "fragment_mass_tolerance": fragment_mass_tolerance,
                "fragment_mass_unit": fragment_mass_unit,
                "add_decoys": bool(add_decoys),
            }
            tasks = []
            for idx, (_, group_df) in enumerate(grouped):
                hit_payloads = []
                for _, row in group_df.iterrows():
                    raw_seq = str(row.get("peptidoform", row.get("sequence", "")))
                    seq_str = unimod_to_pyopenms_notation(raw_seq)
                    proforma = None
                    mv = row.get("psm_metavalues")
                    if isinstance(mv, np.ndarray):
                        for m in mv:
                            if isinstance(m, dict) and m.get("name") == "ProForma":
                                proforma = m.get("value")
                                break
                    hit_payloads.append({"sequence": seq_str, "proforma": proforma})
                row0 = group_df.iloc[0]
                tasks.append({
                    "idx": idx,
                    "exp": exp,
                    "scan_map": scan_map,
                    "params": params,
                    "mz": float(row0.get("observed_mz", row0.get("calculated_mz", 0.0))),
                    "rt": float(row0.get("rt", 0.0)),
                    "spectrum_reference": str(row0.get("spectrum_reference", "")) if pd.notna(row0.get("spectrum_reference")) else None,
                    "hits": hit_payloads,
                })

            indexed_results = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_worker_process_pid_threaded, t): t["idx"] for t in tasks}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        indexed_results[idx] = fut.result()
                    except Exception as e:
                        indexed_results[idx] = {"status": "error", "reason": str(e)}

            # Rebuild results in order
            for idx in range(len(grouped)):
                res = indexed_results.get(idx, {"status": "error", "reason": "unknown"})
                if res["status"] == "success":
                    result_rows.extend(res["rows"])
                    stats["processed"] += 1
                    stats["phospho"] += res["phospho_count"]
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
        click.echo(f"[{time.strftime('%H:%M:%S')}] Saving results to {out_file}")
        if result_rows:
            out_df = pd.DataFrame(result_rows)
            save_identifications(out_file, out_df, proteins_df, template_df=psms_df, source_idparquet=id_file)
        else:
            click.echo("Warning: No results to save")

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


def load_identifications(idparquet_path):
    """Load identification results from an idParquet directory as DataFrames."""
    from onsite.idparquet import load_dataframes as _load_df
    psms, prots, _, _ = _load_df(idparquet_path)
    return psms, prots


def save_identifications(out_file, psms_df, proteins_df=None, template_df=None, source_idparquet=None):
    """Save results directly as DataFrames to an idParquet directory."""
    # Ensure metadata fields on the DataFrame rows
    if "psm_metavalues" in psms_df.columns:
        for idx, row in psms_df.iterrows():
            metas = list(row["psm_metavalues"]) if isinstance(row["psm_metavalues"], np.ndarray) else []
            existing_names = {m["name"] for m in metas if isinstance(m, dict)}

            seq_str = str(row.get("peptidoform", row.get("sequence", "")))
            pyo_seq = unimod_to_pyopenms_notation(seq_str)

            if "search_engine_sequence" not in existing_names:
                metas.append({"name": "search_engine_sequence", "value": pyo_seq, "value_type": "string"})
            if "regular_phospho_count" not in existing_names:
                cnt = sum(pyo_seq.count(f"{aa}(Phospho)") for aa in ["S", "T", "Y"])
                metas.append({"name": "regular_phospho_count", "value": str(cnt), "value_type": "int"})
            if "phospho_decoy_count" not in existing_names:
                cnt = pyo_seq.count("(PhosphoDecoy)")
                metas.append({"name": "phospho_decoy_count", "value": str(cnt), "value_type": "int"})
            if "PhosphoRS_pep_score" not in existing_names:
                metas.append({"name": "PhosphoRS_pep_score", "value": str(row.get("score", -1.0)), "value_type": "double"})

            psms_df.at[idx, "psm_metavalues"] = np.array(metas, dtype=object)

    _save_df(out_file, psms_df, proteins_df, template_df=template_df, source_idparquet=source_idparquet)


def extract_scan_number_from_reference(spectrum_reference):
    """Extract scan number from spectrum_reference string.
    
    Examples:
        "controllerType=0 controllerNumber=1 scan=4886" -> 4886
        "scan=5093" -> 5093
    """
    if not spectrum_reference:
        return None
    try:
        if "scan=" in spectrum_reference:
            # Extract number after "scan="
            parts = spectrum_reference.split("scan=")
            if len(parts) > 1:
                scan_str = parts[-1].split()[0] if " " in parts[-1] else parts[-1]
                return int(scan_str)
    except (ValueError, IndexError):
        pass
    return None


def build_scan_to_spectrum_map(exp):
    """Build a mapping from scan number to spectrum for efficient lookup."""
    scan_map = {}
    for spec in exp:
        if spec.getMSLevel() == 2 and spec.getPrecursors():
            native_id = spec.getNativeID()
            scan_num = extract_scan_number_from_reference(native_id)
            if scan_num is not None:
                scan_map[scan_num] = spec
    return scan_map


def find_spectrum_by_scan(exp, scan_number, scan_map=None):
    """Find spectrum by scan number."""
    if scan_map is None:
        scan_map = build_scan_to_spectrum_map(exp)
    return scan_map.get(scan_number)


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


# Managed PhosphoRS metadata keys, written on scored hits and stripped from
# hits PhosphoRS does not score (so stale values can't leak downstream).
_MANAGED_PHOSPHORS_METAS = (
    "search_engine_sequence",
    "regular_phospho_count",
    "phospho_decoy_count",
    "PhosphoRS_pep_score",
    "PhosphoRS_site_probs",
    "PhosphoRS_site_delta",
    "SpecEValue_score",
    "ProForma",
)


def make_unscored_hit(hit_src):
    """Build the output hit for a peptide PhosphoRS does not score (no phospho
    site, or scoring returned no result): the original hit with score -1 and
    all managed PhosphoRS metadata removed.

    Both the serial (threads=1) and threaded (threads>1) paths route their
    skip branches through this so the two produce byte-identical output.
    """
    h = PeptideHit(hit_src)
    h.setScore(-1.0)
    for k in _MANAGED_PHOSPHORS_METAS:
        if h.metaValueExists(k):
            try:
                h.removeMetaValue(k)
            except Exception:
                pass
    return h


def _has_localizable_phospho(seq_str):
    """True if the (normalized) sequence carries an explicit (Phospho)/
    (PhosphoDecoy) on a localizable residue (S/T/Y/A).

    Both the serial (threads=1) and threaded (threads>1) paths gate on this
    BEFORE calling the scorer, so they skip exactly the same hits. Without a
    shared gate the threaded worker would fall through to the scorer's
    mass-based modification inference (phosphors.py: abs_tol=0.1 Da around the
    phospho mass) and could (mis)score a non-phospho modification that is
    near-isobaric with phospho (e.g. Sulfation, +79.9568, 0.0095 Da away) which
    the serial path skips - diverging threads=1 vs threads>1 output. See the
    PR #41 review (bigbio/onsite#40).
    """
    for aa in ("S", "T", "Y", "A"):
        if f"{aa}(Phospho)" in seq_str or f"{aa}(PhosphoDecoy)" in seq_str:
            return True
    return False


# ----------------------- Threading worker utilities -----------------------
# Note: Using ThreadPoolExecutor instead of ProcessPoolExecutor allows threads
# to share the spectrum data (exp object) directly without reloading the file.
# This provides significant performance improvement for parallel processing.


def _worker_process_pid_threaded(task):
    """Thread-safe worker that uses shared spectrum data.

    Unlike process-based workers, threads share memory so we can pass
    the exp and scan_map objects directly without serialization or file reloading.
    """
    try:
        exp = task["exp"]  # Shared spectrum object - no file reload needed
        scan_map = task["scan_map"]  # Shared scan map - no rebuild needed
        pid_info = task["pid"]
        params = task["params"]

        # First, try to find by scan number from spectrum_reference
        spectrum = None
        scan_number = None

        if pid_info.get("spectrum_reference"):
            scan_number = extract_scan_number_from_reference(pid_info["spectrum_reference"])
            if scan_number is not None:
                spectrum = find_spectrum_by_scan(exp, scan_number, scan_map)

        # Fallback to m/z and RT matching if scan number lookup failed
        if spectrum is None:
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

            # Same localizable-phospho gate as the serial path, so the two paths
            # skip identical hits (the rebuild routes "no_result" through
            # make_unscored_hit, mirroring the serial make_unscored_hit branch).
            if not _has_localizable_phospho(seq.toString()):
                results.append({"status": "no_result"})
                continue

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
            simple_site_probs = {int(k) + 1: float(v) for k, v in site_probs.items()}

            meta_fields = []
            meta_fields.append(("search_engine_sequence", seq_str))
            meta_fields.append(("regular_phospho_count", regular_count))
            meta_fields.append(("phospho_decoy_count", decoy_count))
            meta_fields.append(("PhosphoRS_pep_score", final_score))
            meta_fields.append(("PhosphoRS_site_probs", str(simple_site_probs)))
            meta_fields.append(
                ("PhosphoRS_site_delta", str(site_deltas_from_isomers(isomer_list)))
            )

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


def _metas_list_from_hit_result(seq_str, new_sequence, final_score, site_probs, isomer_list, original_metavalues=None):
    """Build metas list from a PhosphoRS scoring result."""
    regular_count = sum(seq_str.count(f"{aa}(Phospho)") for aa in ["S", "T", "Y"])
    decoy_count = seq_str.count("(PhosphoDecoy)")
    simple_site_probs = {int(k) + 1: float(v) for k, v in site_probs.items()}

    meta_fields = [
        ("search_engine_sequence", seq_str),
        ("regular_phospho_count", regular_count),
        ("phospho_decoy_count", decoy_count),
        ("PhosphoRS_pep_score", final_score),
        ("PhosphoRS_site_probs", str(simple_site_probs)),
        ("PhosphoRS_site_delta", str(site_deltas_from_isomers(isomer_list))),
    ]
    return [
        {"name": k, "value": str(v), "value_type": "double" if isinstance(v, float) else "string"}
        for k, v in meta_fields
    ]


def _make_phosphors_row(seq, raw_seq, charge, mz, rt, scan_num, spectrum_reference,
                        ref_file, hit_idx, pep_idx, score, metas, row, row0):
    return {
        "sequence": seq.toUnmodifiedString(),
        "peptidoform": raw_seq,
        "precursor_charge": charge,
        "calculated_mz": mz, "observed_mz": mz,
        "is_decoy": False, "score": score,
        "score_type": "PhosphoRSScore",
        "higher_score_better": False,
        "rt": rt, "scan": scan_num,
        "spectrum_reference": spectrum_reference or "",
        "reference_file_name": ref_file,
        "hit_index": hit_idx, "peptide_identification_index": pep_idx,
        "psm_metavalues": np.array(_preserve_plus_new(row, metas, _PHOSPHORS_MANAGED), dtype=object),
        "modifications": np.array([], dtype=object),
        "protein_accessions": np.array([], dtype=object),
        "additional_scores": np.array([], dtype=object),
        "run_identifier": str(row0.get("run_identifier", "")) if pd.notna(row0.get("run_identifier", "")) else "",
    }


_EMPTY_PHOSPHORS_METAS = [
    {"name": "search_engine_sequence", "value": "", "value_type": "string"},
    {"name": "regular_phospho_count", "value": "0", "value_type": "int"},
    {"name": "phospho_decoy_count", "value": "0", "value_type": "int"},
    {"name": "PhosphoRS_pep_score", "value": "-1.0", "value_type": "double"},
    {"name": "PhosphoRS_site_probs", "value": "{}", "value_type": "string"},
    {"name": "PhosphoRS_site_delta", "value": "{}", "value_type": "string"},
]

_PHOSPHORS_MANAGED = {"search_engine_sequence", "regular_phospho_count", "phospho_decoy_count",
                      "PhosphoRS_pep_score", "PhosphoRS_site_probs", "PhosphoRS_site_delta",
                      "SpecEValue_score"}


def _find_spectrum(spectrum_reference, mz, rt, exp, scan_map=None):
    spectrum = None
    if spectrum_reference:
        sn = extract_scan_number_from_reference(spectrum_reference)
        if sn is not None:
            if scan_map is None:
                scan_map = build_scan_to_spectrum_map(exp)
            spectrum = find_spectrum_by_scan(exp, sn, scan_map)
    if spectrum is None:
        spectrum = find_spectrum_by_mz(exp, mz, rt)
    return spectrum


def _process_psm_group(group_df, exp, fragment_mass_tolerance, fragment_mass_unit, add_decoys, logger, scan_map=None):
    try:
        row0 = group_df.iloc[0]
        mz = float(row0.get("observed_mz", row0.get("calculated_mz", 0.0)))
        rt = float(row0.get("rt", 0.0))
        spectrum_reference = str(row0.get("spectrum_reference", "")) if pd.notna(row0.get("spectrum_reference")) else None
        scan_num = int(row0.get("scan", 0)) if pd.notna(row0.get("scan")) else 0
        ref_file = str(row0.get("reference_file_name", "")) if pd.notna(row0.get("reference_file_name")) else ""

        spectrum = _find_spectrum(spectrum_reference, mz, rt, exp, scan_map)
        if not spectrum:
            return {"status": "error", "reason": "spectrum_not_found"}

        psm_rows = []
        phospho_count = 0
        pep_idx = int(row0["peptide_identification_index"])

        for hit_idx, (_, row) in enumerate(group_df.iterrows()):
            raw_seq = str(row.get("peptidoform", row.get("sequence", "")))
            seq_str = unimod_to_pyopenms_notation(raw_seq)
            charge = int(row.get("precursor_charge", 0)) if pd.notna(row.get("precursor_charge")) else 0

            seq = AASequence.fromString(seq_str)
            hit = PeptideHit()
            hit.setSequence(seq)
            if charge:
                hit.setCharge(charge)

            if not _has_localizable_phospho(seq_str):
                metas = _EMPTY_PHOSPHORS_METAS.copy()
                metas[0]["value"] = seq_str
                psm_rows.append(_make_phosphors_row(
                    seq, raw_seq, charge, mz, rt, scan_num, spectrum_reference,
                    ref_file, hit_idx, pep_idx, -1.0, metas, row, row0))
                continue

            site_probs, isomer_list = calculate_phospho_localization_compomics_style(
                hit, spectrum,
                fragment_tolerance=fragment_mass_tolerance,
                fragment_method_ppm=(fragment_mass_unit == "ppm"),
                add_decoys=add_decoys,
            )

            if site_probs is None or isomer_list is None:
                metas = _EMPTY_PHOSPHORS_METAS.copy()
                metas[0]["value"] = seq_str
                psm_rows.append(_make_phosphors_row(
                    seq, raw_seq, charge, mz, rt, scan_num, spectrum_reference,
                    ref_file, hit_idx, pep_idx, -1.0, metas, row, row0))
                continue

            best_isomer = min(isomer_list, key=lambda x: x[1])
            final_score = float(best_isomer[1])
            new_sequence = best_isomer[0]
            # Re-attach N-terminal modification dropped by isomer generation
            if seq_str.startswith("(") and not new_sequence.startswith("("):
                end = seq_str.find(")", 1)
                if end > 0 and not any(seq_str.startswith(t) for t in ("(Phospho", "(PhosphoDecoy")):
                    new_sequence = seq_str[:end + 1] + new_sequence
            new_peptidoform = pyopenms_to_unimod_notation(new_sequence)

            metas_list = _metas_list_from_hit_result(seq_str, new_sequence, final_score, site_probs, isomer_list)
            phospho_count += 1

            psm_rows.append(_make_phosphors_row(
                seq, new_peptidoform, charge, mz, rt, scan_num, spectrum_reference,
                ref_file, hit_idx, pep_idx, float(final_score), metas_list, row, row0))

        return {"status": "success", "rows": psm_rows, "phospho_count": phospho_count}

    except Exception as e:
        if logger:
            logger.error(f"Error: {e}")
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



def _preserve_plus_new(row, new_metas: list, managed_keys: set) -> list:
    """Merge original psm_metavalues from row with new metas, dropping managed keys."""
    orig = row.get("psm_metavalues")
    orig_list = list(orig) if isinstance(orig, np.ndarray) else []
    return [m for m in orig_list if isinstance(m, dict) and m.get("name") not in managed_keys] + new_metas

def main():
    """Entry point for standalone PhosphoRS CLI."""
    phosphors()


if __name__ == "__main__":
    main()
