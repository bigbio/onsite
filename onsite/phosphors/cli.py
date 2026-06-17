#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import logging
import traceback
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import click
import pandas as pd
from pyopenms import AASequence, MSExperiment, FileHandler, PeptideHit, SpectrumLookup
from onsite.idparquet import unimod_to_pyopenms_notation, pyopenms_to_unimod_notation
from onsite.idparquet import save_dataframes as _save_df

from .phosphors import (
    calculate_phospho_localization_compomics_style,
    site_deltas_from_isomers,
)

_EMPTY_META_KEYS = [
    "search_engine_sequence", "regular_phospho_count", "phospho_decoy_count",
    "PhosphoRS_pep_score", "PhosphoRS_site_probs", "PhosphoRS_site_delta",
]

_EMPTY_META_VALUES = lambda s: [s, "0", "0", "-1.0", "{}", "{}"]
_EMPTY_META_TYPES = lambda: ["string", "int", "int", "double", "string", "string"]


def _fresh_empty_metas(seq_str=""):
    return [{"name": n, "value": v, "value_type": t}
            for n, v, t in zip(_EMPTY_META_KEYS, _EMPTY_META_VALUES(seq_str), _EMPTY_META_TYPES())]


_PHOSPHORS_MANAGED = {"search_engine_sequence", "regular_phospho_count", "phospho_decoy_count",
                        "PhosphoRS_pep_score", "PhosphoRS_site_probs", "PhosphoRS_site_delta",
                        "SpecEValue_score"}


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
        lookup = build_scan_to_spectrum_map(exp)

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
                        group_df, exp, fragment_mass_tolerance, fragment_mass_unit, add_decoys, logger, lookup
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
                    "lookup": lookup,
                    "params": params,
                    "mz": float(row0.get("observed_mz", row0.get("calculated_mz", 0.0))),
                    "rt": float(row0.get("rt", 0.0)),
                    "spectrum_reference": str(row0.get("scan")),
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

        # Report
        elapsed = time.time() - start_time
        click.echo(f"\nProcessing Complete:")
        click.echo(f"  Total identifications: {stats['total']}")
        click.echo(f"  Successfully processed: {stats['processed']}")
        click.echo(f"  Modification sites reassigned: {relocated_count}")
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
    for idx, row in psms_df.iterrows():
        metas = list(row["psm_metavalues"])
        existing_names = {m["name"] for m in metas}

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


def build_scan_to_spectrum_map(exp):
    """Build a mapping from scan number to spectrum for efficient lookup."""
    lookup = SpectrumLookup()
    if "spectrum=" in exp.getSpectrum(0).getNativeID():
        lookup.readSpectra(exp, "spectrum=(?<SCAN>\\d+)")
    else:
        lookup.readSpectra(exp, "scan=(?<SCAN>\\d+)")

    return lookup


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
        lookup = task["lookup"]  # Shared scan map - no rebuild needed
        pid_info = task["pid"]
        params = task["params"]

        # First, try to find by scan number from spectrum_reference

        scan_number = pid_info["scan"]
        index = lookup.findByScanNumber(scan_number)
        spectrum = exp.getSpectrum(index)


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


@dataclass
class _RowCtx:
    mz: float
    rt: float
    scan_num: int
    spectrum_reference: str
    ref_file: str
    pep_idx: int
    run_identifier: str


def _make_phosphors_row(ctx: _RowCtx, seq, raw_seq, charge, hit_idx, score, metas, row):
    return {
        "sequence": seq.toUnmodifiedString(),
        "peptidoform": raw_seq,
        "precursor_charge": charge,
        "observed_mz": ctx.mz,
        "is_decoy": False, "score": score,
        "score_type": "PhosphoRSScore",
        "higher_score_better": False,
        "rt": ctx.rt, "scan": ctx.scan_num,
        "spectrum_reference": ctx.spectrum_reference or "",
        "reference_file_name": ctx.ref_file,
        "hit_index": hit_idx, "peptide_identification_index": ctx.pep_idx,
        "psm_metavalues": np.array(_preserve_plus_new(row, metas, _PHOSPHORS_MANAGED), dtype=object),
        "modifications": np.array([], dtype=object),
        "protein_accessions": np.array([], dtype=object),
        "additional_scores": np.array([], dtype=object),
        "run_identifier": ctx.run_identifier,
    }


def _process_psm_group(group_df, exp, fragment_mass_tolerance, fragment_mass_unit, add_decoys, logger, lookup):

    row0 = group_df.iloc[0]
    ctx = _RowCtx(
        mz=float(row0.get("observed_mz")),
        rt=float(row0.get("rt")),
        scan_num=int(row0.get("scan")),
        spectrum_reference=str(row0.get("spectrum_reference")),
        ref_file=str(row0.get("reference_file_name")),
        pep_idx=int(row0["peptide_identification_index"]),
        run_identifier=str(row0.get("run_identifier")),
    )

    index = lookup.findByScanNumber(ctx.scan_num)
    spectrum = exp.getSpectrum(index)

    psm_rows = []
    phospho_count = 0

    for hit_idx, (_, row) in enumerate(group_df.iterrows()):
        hit_result = _process_single_hit(
            hit_idx, row, ctx, spectrum, fragment_mass_tolerance, fragment_mass_unit, add_decoys)
        psm_rows.append(hit_result["row"])
        phospho_count += hit_result["phospho_count"]

    return {"status": "success", "rows": psm_rows, "phospho_count": phospho_count}


def _restore_nterm_mod(seq_str: str, new_sequence: str) -> str:
    """Re-attach N-terminal modification dropped by isomer generation, if any."""
    if not seq_str.startswith("(") or new_sequence.startswith("("):
        return new_sequence
    end = seq_str.find(")", 1)
    if end <= 0:
        return new_sequence
    if any(seq_str.startswith(t) for t in ("(Phospho", "(PhosphoDecoy")):
        return new_sequence
    return seq_str[:end + 1] + new_sequence


def _process_single_hit(hit_idx, row, ctx, spectrum, fragment_mass_tolerance, fragment_mass_unit, add_decoys):
    """Process a single peptide hit for phospho localization. Returns dict with 'row' and 'phospho_count'."""
    raw_seq = str(row.get("peptidoform"))
    seq_str = unimod_to_pyopenms_notation(raw_seq)
    charge = int(row.get("precursor_charge"))
    seq = AASequence.fromString(seq_str)
    hit = PeptideHit()
    hit.setSequence(seq)
    hit.setCharge(charge)

    if not _has_localizable_phospho(seq_str):
        return {"row": _make_phosphors_row(ctx, seq, raw_seq, charge, hit_idx, -1.0, _fresh_empty_metas(seq_str), row),
                "phospho_count": 0}

    site_probs, isomer_list = calculate_phospho_localization_compomics_style(
        hit, spectrum, fragment_tolerance=fragment_mass_tolerance,
        fragment_method_ppm=(fragment_mass_unit == "ppm"), add_decoys=add_decoys,
    )

    if site_probs is None or isomer_list is None:
        return {"row": _make_phosphors_row(ctx, seq, raw_seq, charge, hit_idx, -1.0, _fresh_empty_metas(seq_str), row),
                "phospho_count": 0}

    best_isomer = min(isomer_list, key=lambda x: x[1])
    final_score = float(best_isomer[1])
    new_sequence = _restore_nterm_mod(seq_str, best_isomer[0])
    new_peptidoform = pyopenms_to_unimod_notation(new_sequence)

    metas_list = _metas_list_from_hit_result(seq_str, new_sequence, final_score, site_probs, isomer_list)
    return {"row": _make_phosphors_row(ctx, seq, new_peptidoform, charge, hit_idx, final_score, metas_list, row),
            "phospho_count": 1}


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
