#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import ast
import time
import logging
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import click
import numpy as np
import pandas as pd
from pyopenms import AASequence, PeptideHit, MSExperiment, FileHandler, SpectrumLookup
from onsite.idparquet import (
    load_dataframes,
    save_dataframes,
    pyopenms_to_unimod_notation,
    unimod_to_pyopenms_notation,
)

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
    help="Number of parallel threads (default: 1)",
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
def ascore(
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
    """Phosphorylation site localization scoring tool using AScore algorithm."""
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
        exp = load_spectra(in_file)
        psms_df, proteins_df, _, _ = load_dataframes(id_file)

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
            logger.info(f"Total spectra: {exp.size()}")
            logger.info(f"Total identifications: {psms_df['peptide_identification_index'].nunique()}")

        grouped = list(psms_df.sort_values("peptide_identification_index").groupby("peptide_identification_index"))
        stats = {"total": len(grouped), "processed": 0, "phospho": 0, "errors": 0}
        start_time = time.time()
        result_rows: List[Dict] = []
        lookup = SpectrumLookup()
        if "spectrum=" in exp.getSpectrum(0).getNativeID():
            lookup.readSpectra(exp, "spectrum=(?<SCAN>\\d+)")
        else:
            lookup.readSpectra(exp, "scan=(?<SCAN>\\d+)")

        if max(1, int(threads)) == 1:
            click.echo(f"[{time.strftime('%H:%M:%S')}] Processing {len(grouped)} peptide identifications sequentially...")
            for _, (_, group_df) in enumerate(grouped):
                try:
                    res = process_psm_group(
                        group_df, exp, lookup, fragment_mass_tolerance, fragment_mass_unit, add_decoys, logger, debug
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
                        logger.error(f"Error: {e}")
                    traceback.print_exc()
        else:
            workers = max(1, int(threads))
            click.echo(f"[{time.strftime('%H:%M:%S')}] Parallel execution with {workers} threads")
            params = {
                "fragment_mass_tolerance": fragment_mass_tolerance,
                "fragment_mass_unit": fragment_mass_unit,
                "add_decoys": add_decoys,
            }
            tasks = []
            for idx, (_, group_df) in enumerate(grouped):
                hit_payloads = []
                for _, row in group_df.iterrows():
                    orig_mv = row.get("psm_metavalues")
                    hit_payloads.append({
                        "sequence": unimod_to_pyopenms_notation(str(row.get("peptidoform", row.get("sequence", "")))),
                        "charge": int(row.get("precursor_charge", 0)) if pd.notna(row.get("precursor_charge")) else 0,
                        "orig_metas": list(orig_mv) if isinstance(orig_mv, np.ndarray) else [],
                    })
                tasks.append({
                    "idx": idx,
                    "exp": exp,
                    "lookup": lookup,
                    "params": params,
                    "scan": int(group_df.iloc[0].get("scan")),
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

            for idx in range(len(grouped)):
                res = indexed_results.get(idx, {"status": "error", "reason": "unknown"})
                if res["status"] == "success":
                    result_rows.extend(res["rows"])
                    stats["processed"] += 1
                    stats["phospho"] += res["phospho_count"]
                else:
                    stats["errors"] += 1

        # Count how many PSMs had their modification sites reassigned
        relocated_count = 0
        orig_pf_map = {}
        for _, group_df in grouped:
            pep_idx = group_df.iloc[0].get("peptide_identification_index")
            orig_pf = str(group_df.iloc[0].get("peptidoform", ""))
            if orig_pf:
                orig_pf_map[pep_idx] = orig_pf
        for row in result_rows:
            pep_idx = row.get("peptide_identification_index")
            orig_pf = orig_pf_map.get(pep_idx, "")
            out_pf = str(row.get("peptidoform", ""))
            if orig_pf and orig_pf != out_pf:
                relocated_count += 1

        elapsed = time.time() - start_time
        click.echo(f"\nProcessing Complete:")
        click.echo(f"  Total identifications: {stats['total']}")
        click.echo(f"  Successfully processed: {stats['processed']}")
        click.echo(f"  Modification sites reassigned: {relocated_count}")
        click.echo(f"  Phosphorylated peptides: {stats['phospho']}")
        click.echo(f"  Processing errors: {stats['errors']}")
        click.echo(f"  Time elapsed: {elapsed:.2f} seconds")
        click.echo(f"  Processing speed: {stats['processed']/elapsed:.2f} IDs/second")

        if result_rows:
            out_df = pd.DataFrame(result_rows)
            click.echo(f"[{time.strftime('%H:%M:%S')}] Saving results to {out_file}")
            save_dataframes(out_file, out_df, proteins_df, template_df=psms_df, source_idparquet=id_file)
        else:
            click.echo("Warning: No results to save")

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
    print(f"[{time.strftime('%H:%M:%S')}] Loading spectra from {mzml_file}")
    exp = MSExperiment()
    FileHandler().loadExperiment(mzml_file, exp)
    print(f"Loaded {exp.size()} spectra")
    return exp


def log_debug(log_file, enabled):
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


def find_spectrum_by_mz(exp, target_mz, rt=None, ppm_tolerance=10):
    if not hasattr(find_spectrum_by_mz, "spectrum_cache"):
        find_spectrum_by_mz.spectrum_cache = {}
        find_spectrum_by_mz.spectrum_list = []
        for spec in exp:
            if spec.getMSLevel() == 2 and spec.getPrecursors():
                mz = spec.getPrecursors()[0].getMZ()
                find_spectrum_by_mz.spectrum_list.append((mz, spec))
        find_spectrum_by_mz.spectrum_list.sort(key=lambda x: x[0])

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


def _metas_from_scored_hit(scored_hit, original_seq_str: str) -> tuple:
    """Extract meta_fields and best_ascore from a scored PeptideHit."""
    new_seq_str = scored_hit.getSequence().toString()
    phospho_count = new_seq_str.count("(Phospho)")

    site_scores = _extract_ascore_list(scored_hit, phospho_count)

    ascore_pep_score = (
        float(scored_hit.getMetaValue("AScore_pep_score"))
        if scored_hit.metaValueExists("AScore_pep_score")
        else -1.0
    )
    best_ascore = min(site_scores) if site_scores else -1.0

    meta_fields = [
        ("search_engine_sequence", original_seq_str),
        ("regular_phospho_count", phospho_count),
        ("phospho_decoy_count", new_seq_str.count("(PhosphoDecoy)")),
        ("AScore_pep_score", ascore_pep_score),
    ]
    for i, score in enumerate(site_scores, 1):
        meta_fields.append((f"AScore_{i}", score))
    if scored_hit.metaValueExists("ProForma"):
        meta_fields.append(("ProForma", scored_hit.getMetaValue("ProForma")))
    if scored_hit.metaValueExists("AScore_site_scores"):
        raw = scored_hit.getMetaValue("AScore_site_scores")
        try:
            d = ast.literal_eval(raw)
            d_1based = str({int(k) + 1: v for k, v in d.items()})
        except Exception:
            d_1based = raw
        meta_fields.append(("AScore_site_scores", d_1based))

    return meta_fields, best_ascore, new_seq_str, phospho_count


def _extract_ascore_list(scored_hit, phospho_count: int):
    site_scores = []
    rank = 1

    while scored_hit.metaValueExists(f"AScore_{rank}"):
        site_scores.append(float(scored_hit.getMetaValue(f"AScore_{rank}")))
        rank += 1

    if len(site_scores) < phospho_count:
        site_scores.extend([1000.0] * (phospho_count - len(site_scores)))

    return site_scores


def _process_hit_serial(hit, fragment_mass_tolerance, fragment_mass_unit, add_decoys, spectrum):
    """Process a single hit in serial mode. Returns meta_fields, best_ascore, new_seq_str, phospho_count."""
    new_hit = PeptideHit(hit)
    original_seq_str = new_hit.getSequence().toString()
    new_hit.setMetaValue("search_engine_sequence", original_seq_str)

    if "(Phospho)" not in original_seq_str and "(PhosphoDecoy)" not in original_seq_str:
        new_hit.setScore(-1.0)
        new_hit.setMetaValue("AScore_pep_score", -1.0)
        meta_fields = [
            ("search_engine_sequence", original_seq_str),
            ("regular_phospho_count", 0),
            ("phospho_decoy_count", 0),
            ("AScore_pep_score", -1.0),
        ]
        return meta_fields, -1.0, original_seq_str, 0

    ascore = AScore()
    ascore.fragment_mass_tolerance_ = fragment_mass_tolerance
    ascore.fragment_tolerance_ppm_ = (fragment_mass_unit == "ppm")
    ascore.setAddDecoys(add_decoys)

    scored_hit = ascore.compute(new_hit, spectrum)
    meta_fields, best_ascore, new_seq_str, phospho_count = _metas_from_scored_hit(scored_hit, original_seq_str)
    return meta_fields, best_ascore, new_seq_str, phospho_count


def process_psm_group(group_df, exp, lookup, fragment_mass_tolerance, fragment_mass_unit, add_decoys, logger, debug=False):
    """Process a group of PSM rows (one identification, possibly multiple hits)."""
    try:
        row0 = group_df.iloc[0]
        scan_num = int(row0.get("scan"))
        index = lookup.findByScanNumber(scan_num)
        spectrum = exp.getSpectrum(index)
        score_type = "PhosphoScore"
        higher_better = True
        pep_idx = int(row0["peptide_identification_index"])

        psm_rows = []
        phospho_count = 0

        for hit_idx, (_, row) in enumerate(group_df.iterrows()):
            seq_str = unimod_to_pyopenms_notation(str(row.get("peptidoform")))
            charge = int(row.get("precursor_charge", 0))

            # Build a PeptideHit for the algorithm (internal)
            seq = AASequence.fromString(seq_str)
            hit = PeptideHit()
            hit.setSequence(seq)
            hit.setScore(float(row["score"]))
            hit.setCharge(charge)

            meta_fields, best_ascore, new_seq_str, ph_count = _process_hit_serial(
                hit, fragment_mass_tolerance, fragment_mass_unit, add_decoys, spectrum
            )
            phospho_count += 1 if ph_count > 0 else 0
            peptidoform_out = pyopenms_to_unimod_notation(new_seq_str)
            metas_list = [{"name": k, "value": str(v), "value_type": "double" if isinstance(v, float) else "string"} for k, v in meta_fields]
            _ascore_prefixes = {"search_engine_sequence", "regular_phospho_count", "phospho_decoy_count",
                                "AScore_pep_score", "AScore_site_scores", "ProForma", "AScore_"}
            combined_metas = _preserve_plus_new(row, metas_list, _ascore_prefixes)

            psm_rows.append({
                "sequence": hit.getSequence().toUnmodifiedString(), "peptidoform": peptidoform_out,
                "precursor_charge": charge, "observed_mz": float(row0.get("observed_mz")),
                "is_decoy": False, "score": float(best_ascore),
                "score_type": score_type, "higher_score_better": higher_better,
                "rt": float(row0.get("rt")), "scan": scan_num,
                "spectrum_reference": str(row0.get("spectrum_reference")),
                "reference_file_name": str(row0.get("reference_file_name")),
                "hit_index": hit_idx, "peptide_identification_index": pep_idx,
                "psm_metavalues": np.array(combined_metas, dtype=object),
                "modifications": np.array([], dtype=object),
                "protein_accessions": np.array([], dtype=object),
                "additional_scores": np.array([], dtype=object),
                "run_identifier": str(row0.get("run_identifier"))
            })

        return {"status": "success", "rows": psm_rows, "phospho_count": phospho_count}

    except Exception as e:
        if debug and logger:
            logger.error(f"Error: {e}")
        return {"status": "error", "reason": str(e)}


def _worker_process_pid_threaded(task):
    """Thread worker. task has: exp, lookup, params, hits=[{sequence, charge}]."""
    try:
        lookup = task["lookup"]
        params = task["params"]
        index = lookup.findByScanNumber(task["scan"])
        spectrum = task["exp"].getSpectrum(index)

        rows = []
        for hit_info in task["hits"]:
            seq = AASequence.fromString(hit_info["sequence"])
            hit = PeptideHit()
            hit.setSequence(seq)
            if hit_info.get("charge"):
                hit.setCharge(hit_info["charge"])

            ascore = AScore()
            ascore.fragment_mass_tolerance_ = params["fragment_mass_tolerance"]
            ascore.fragment_tolerance_ppm_ = params["fragment_mass_unit"] == "ppm"
            ascore.setAddDecoys(params.get("add_decoys", False))

            scored_hit = ascore.compute(hit, spectrum)
            meta_fields, best_ascore, new_seq_str, _ = _metas_from_scored_hit(scored_hit, hit_info["sequence"])

            metas_list = [{"name": k, "value": str(v), "value_type": "double" if isinstance(v, float) else "string"} for k, v in meta_fields]
            _ascore_prefixes = {"search_engine_sequence", "regular_phospho_count", "phospho_decoy_count",
                                "AScore_pep_score", "AScore_site_scores", "ProForma", "AScore_"}
            orig = hit_info.get("orig_metas", [])
            filtered_orig = [m for m in orig if isinstance(m, dict) and m.get("name") not in _ascore_prefixes
                            and not m.get("name", "").startswith("AScore_")]
            combined = filtered_orig + metas_list
            rows.append({
                "new_sequence": new_seq_str,
                "best_ascore": best_ascore,
                "meta_fields": meta_fields,
                "metas_list": combined,
            })

        return {"status": "success", "hits": rows}
    except Exception as e:
        return {"status": "error", "reason": str(e)}



def _preserve_plus_new(row, new_metas: list, managed_prefixes: set) -> list:
    """Merge original psm_metavalues from row with new metas, dropping managed keys."""
    orig = row.get("psm_metavalues")
    orig_list = list(orig) if isinstance(orig, np.ndarray) else []

    def _is_managed(name):
        return name in managed_prefixes or any(name.startswith(p) for p in managed_prefixes if p.endswith("_"))
    return [m for m in orig_list if isinstance(m, dict) and not _is_managed(m.get("name", ""))] + new_metas

def main():
    ascore()


if __name__ == "__main__":
    main()
