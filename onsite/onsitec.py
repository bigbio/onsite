#!/usr/bin/env python3
"""
OnSite CLI - Integrated command line interface for phosphorylation site localization tools.

This module provides a unified interface for accessing AScore, PhosphoRS, and LucXor algorithms.
"""

import click
import sys
import os
import tempfile
import time
from typing import Dict

import numpy as np
from onsite.lucxor.cli import lucxor
from onsite.phosphors.cli import phosphors
from onsite.ascore.cli import ascore
import pandas as pd
from onsite.idparquet import load_dataframes, save_dataframes, unimod_to_pyopenms_notation

@click.group()
@click.version_option(version="0.0.3")
def cli():
    """
    OnSite: Mass spectrometry post-translational modification localization tool

    Available algorithms:
      ascore      AScore algorithm for phosphorylation site localization
      phosphors   PhosphoRS algorithm for phosphorylation site localization
      lucxor      LucXor (LuciPHOr2) algorithm for PTM localization
      all         Run all three algorithms and merge results

    Examples:
      onsite ascore -in spectra.mzML -id identifications.idparquet -out results.idparquet
      onsite phosphors -in spectra.mzML -id identifications.idparquet -out results.idparquet
      onsite lucxor -in spectra.mzML -id identifications.idparquet -out results.idparquet
      onsite all -in spectra.mzML -id identifications.idparquet -out results.idparquet --add-decoys
    """
    pass


cli.add_command(ascore)
cli.add_command(phosphors)
cli.add_command(lucxor)


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
    help="Include A (PhosphoDecoy) as potential phosphorylation site for all algorithms",
)
def all(
    in_file,
    id_file,
    out_file,
    fragment_mass_tolerance,
    fragment_mass_unit,
    threads,
    debug,
    add_decoys,
):
    """Run all three algorithms (AScore, PhosphoRS, LucXor) and merge results."""
    try:
        start_time = time.time()
        click.echo(f"[{time.strftime('%H:%M:%S')}] Starting OnSite with all algorithms")
        click.echo(f"  Input spectrum: {in_file}")
        click.echo(f"  Input ID: {id_file}")
        click.echo(f"  Output: {out_file}")

        with tempfile.TemporaryDirectory() as tmpdir:
            ascore_out = os.path.join(tmpdir, "ascore_result.idparquet")
            phosphors_out = os.path.join(tmpdir, "phosphors_result.idparquet")
            lucxor_out = os.path.join(tmpdir, "lucxor_result.idparquet")

            # Run AScore
            click.echo(f"\n{'='*60}")
            click.echo(f"[{time.strftime('%H:%M:%S')}] Running AScore...")
            click.echo(f"{'='*60}")
            from onsite.ascore.cli import ascore as ascore_func
            ctx = click.Context(ascore_func)
            ctx.invoke(
                ascore_func,
                in_file=in_file, id_file=id_file, out_file=ascore_out,
                fragment_mass_tolerance=fragment_mass_tolerance,
                fragment_mass_unit=fragment_mass_unit,
                threads=threads, debug=debug, add_decoys=add_decoys,
            )

            # Run PhosphoRS
            click.echo(f"\n{'='*60}")
            click.echo(f"[{time.strftime('%H:%M:%S')}] Running PhosphoRS...")
            click.echo(f"{'='*60}")
            from onsite.phosphors.cli import phosphors as phosphors_func
            ctx = click.Context(phosphors_func)
            ctx.invoke(
                phosphors_func,
                in_file=in_file, id_file=id_file, out_file=phosphors_out,
                fragment_mass_tolerance=fragment_mass_tolerance,
                fragment_mass_unit=fragment_mass_unit,
                threads=threads, debug=debug, add_decoys=add_decoys,
            )

            # Run LucXor
            click.echo(f"\n{'='*60}")
            click.echo(f"[{time.strftime('%H:%M:%S')}] Running LucXor...")
            click.echo(f"{'='*60}")
            from onsite.lucxor.cli import lucxor as lucxor_func

            if add_decoys:
                target_mods = ("Phospho (S)", "Phospho (T)", "Phospho (Y)", "PhosphoDecoy (A)")
            else:
                target_mods = ("Phospho (S)", "Phospho (T)", "Phospho (Y)")

            ctx = click.Context(lucxor_func)
            ctx.invoke(
                lucxor_func,
                input_spectrum=in_file, input_id=id_file, output=lucxor_out,
                fragment_method="CID",
                fragment_mass_tolerance=fragment_mass_tolerance,
                fragment_error_units=fragment_mass_unit,
                min_mz=150.0,
                target_modifications=target_mods,
                neutral_losses=("sty -H3PO4 -97.97690",),
                decoy_mass=79.966331,
                decoy_neutral_losses=("X -H3PO4 -97.97690",),
                max_charge_state=5, max_peptide_length=40, max_num_perm=16384,
                modeling_score_threshold=0.95, scoring_threshold=0.0,
                min_num_psms_model=50, threads=threads, rt_tolerance=0.01,
                debug=debug, log_file=None, disable_split_by_charge=False,
            )

            # Merge results
            click.echo(f"\n{'='*60}")
            click.echo(f"[{time.strftime('%H:%M:%S')}] Merging results...")
            click.echo(f"{'='*60}")
            merge_algorithm_results(ascore_out, phosphors_out, lucxor_out, out_file)

        elapsed = time.time() - start_time
        click.echo(f"\n{'='*60}")
        click.echo(f"All algorithms completed successfully! Time: {elapsed:.2f}s")
        click.echo(f"  Output: {os.path.abspath(out_file)}")
        click.echo(f"{'='*60}")

    except KeyboardInterrupt:
        click.echo("\nCancelled")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        import traceback; traceback.print_exc()
        sys.exit(1)


def _is_new_valid_ref(ref, common, seen):
    """True when ref is non-empty, present in all three tool maps, and not yet seen."""
    return bool(ref) and ref in common and ref not in seen


def _seqs_across_tools_match(ascore_df, phosphors_df, lucxor_df,
                                a_map, p_map, l_map, ref):
    """True when the base sequence for ref is identical across all three tools."""
    ai, pi = a_map[ref], p_map[ref]
    s_a = _get_ref_seq(ascore_df, ai)
    s_p = _get_ref_seq(phosphors_df, pi)
    s_l = _get_ref_seq(lucxor_df, l_map[ref])
    return bool(s_a) and bool(s_p) and bool(s_l) and s_a == s_p == s_l


def _get_ref_seq(df, idx):
    row = df.iloc[idx]
    return unimod_to_pyopenms_notation(str(row.get("sequence")))


def _by_ref(df):
    idx = df[df["hit_index"] == 0]["spectrum_reference"]
    return {str(ref): i for i, ref in idx.items() if ref and str(ref).strip()}


def _join_psms_by_ref(ascore_df, phosphors_df, lucxor_df):
    """
    Match PSM rows across three tool DataFrames by spectrum_reference.

    Returns (merged_rows, stats).
    """
    a_map = _by_ref(ascore_df)
    p_map = _by_ref(phosphors_df)
    l_map = _by_ref(lucxor_df)

    common = set(a_map) & set(p_map) & set(l_map)

    triples = []
    seen = set()
    seq_mismatch = 0
    # Preserve LucXor order
    lucxor_sorted = lucxor_df[lucxor_df["hit_index"] == 0].copy()
    for _, row in lucxor_sorted.iterrows():
        ref = str(row.get("spectrum_reference", ""))
        if not _is_new_valid_ref(ref, common, seen):
            continue
        seen.add(ref)
        if not _seqs_across_tools_match(
            ascore_df, phosphors_df, lucxor_df, a_map, p_map, l_map, ref
        ):
            seq_mismatch += 1
            continue
        ai, pi = a_map[ref], p_map[ref]
        triples.append((ai, pi, l_map[ref]))

    stats = {
        "ascore_dropped": len(a_map) - len(common),
        "phosphors_dropped": len(p_map) - len(common),
        "lucxor_dropped": len(l_map) - len(common),
        "seq_mismatch": seq_mismatch,
        "merged": len(triples),
    }
    return triples, stats


def _get_metas_dict(df, row_idx: int) -> Dict[str, str]:
    """Get psm_metavalues as a dict for a given row index."""
    row = df.iloc[row_idx]
    mv = row.get("psm_metavalues")
    if mv is None:
        return {}
    items = list(mv) if isinstance(mv, np.ndarray) else []
    return {str(m["name"]): str(m["value"]) for m in items if isinstance(m, dict)}


def merge_algorithm_results(ascore_file, phosphors_file, lucxor_file, output_file, input_idparquet):
    """Merge results from all three algorithms into a single idparquet directory."""
    ascore_df, _, _, _ = load_dataframes(ascore_file)
    phosphors_df, _, _, _ = load_dataframes(phosphors_file)
    lucxor_df, proteins_df, _, _ = load_dataframes(lucxor_file)

    triples, stats = _join_psms_by_ref(ascore_df, phosphors_df, lucxor_df)
    for tool in ("ascore", "phosphors", "lucxor"):
        if stats[f"{tool}_dropped"]:
            click.echo(f"  Note: {stats[f'{tool}_dropped']} {tool} PSM(s) not in all tools")
    if stats["seq_mismatch"]:
        click.echo(f"  Warning: {stats['seq_mismatch']} PSM(s) skipped (seq mismatch)")

    merged_rows = []

    for ai, pi, li in triples:
        a_metas = _get_metas_dict(ascore_df, ai)
        p_metas = _get_metas_dict(phosphors_df, pi)
        l_metas = _get_metas_dict(lucxor_df, li)

        # Get the actual row data from LucXor for base properties
        for hit_idx in range(3):
            a_hit_row = ascore_df[(ascore_df["peptide_identification_index"] == ai) & (ascore_df["hit_index"] == hit_idx)]
            p_hit_row = phosphors_df[(phosphors_df["peptide_identification_index"] == pi) & (phosphors_df["hit_index"] == hit_idx)]
            l_hit_row = lucxor_df[(lucxor_df["peptide_identification_index"] == li) & (lucxor_df["hit_index"] == hit_idx)]

            if l_hit_row.empty:
                break

            l_hit = l_hit_row.iloc[0]
            seq = str(l_hit.get("sequence", ""))
            peptidoform = str(l_hit.get("peptidoform", ""))
            charge = int(l_hit.get("precursor_charge", 0)) if pd.notna(l_hit.get("precursor_charge")) else 0
            mz = float(l_hit.get("observed_mz", l_hit.get("calculated_mz", 0.0)))
            rt = float(l_hit.get("rt", 0.0))
            score = float(l_hit.get("score", 0.0))
            scan = int(l_hit.get("scan", 0)) if pd.notna(l_hit.get("scan")) else 0
            spec_ref = str(l_hit.get("spectrum_reference", ""))
            ref_file = str(l_hit.get("reference_file_name", ""))

            # Build merged metas
            merged_metas = []

            # Preserve selected metas from LucXor
            for k in ["target_decoy", "q-value", "Posterior Error Probability_score"]:
                v = l_metas.get(k)
                if v:
                    merged_metas.append({"name": k, "value": v, "value_type": "string"})

            # AScore metas
            if not a_hit_row.empty:
                a_hit = a_hit_row.iloc[0]
                a_peptidoform = str(a_hit.get("peptidoform", ""))
                merged_metas.append({"name": "AScore_sequence", "value": a_peptidoform, "value_type": "string"})
                merged_metas.append({"name": "AScore_best_score", "value": str(a_hit.get("score", -1)), "value_type": "double"})
                for k, v in a_metas.items():
                    if k.startswith("AScore_") or k == "AScore_pep_score":
                        merged_metas.append({"name": k, "value": v, "value_type": "double" if k.endswith("score") else "string"})

            # PhosphoRS metas
            if not p_hit_row.empty:
                p_hit = p_hit_row.iloc[0]
                p_peptidoform = str(p_hit.get("peptidoform", ""))
                merged_metas.append({"name": "PhosphoRS_sequence", "value": p_peptidoform, "value_type": "string"})
                merged_metas.append({"name": "PhosphoRS_score", "value": str(p_hit.get("score", -1)), "value_type": "double"})
                for k in ["PhosphoRS_pep_score", "PhosphoRS_site_probs"]:
                    v = p_metas.get(k)
                    if v:
                        merged_metas.append({"name": k, "value": v, "value_type": "double" if "score" in k else "string"})

            # Luciphor metas
            merged_metas.append({"name": "Luciphor_sequence", "value": peptidoform, "value_type": "string"})
            merged_metas.append({"name": "Luciphor_delta_score", "value": str(score), "value_type": "double"})
            for k in ["Luciphor_pep_score", "Luciphor_global_flr", "Luciphor_local_flr", "Luciphor_site_scores", "search_engine_sequence"]:
                v = l_metas.get(k)
                if v:
                    merged_metas.append({"name": k, "value": v, "value_type": "double" if "score" in k or "flr" in k.lower() else "string"})

            merged_rows.append({
                "sequence": seq,
                "peptidoform": peptidoform,
                "precursor_charge": charge,
                "calculated_mz": mz, "observed_mz": mz,
                "is_decoy": str(l_hit.get("is_decoy", "False")).lower() == "true",
                "score": float(score),
                "score_type": "onsite_combined_score",
                "higher_score_better": True,
                "rt": rt, "scan": scan,
                "spectrum_reference": spec_ref,
                "reference_file_name": ref_file,
                "hit_index": hit_idx,
                "peptide_identification_index": li,
                "psm_metavalues": np.array(merged_metas, dtype=object),
                "modifications": np.array([], dtype=object),
                "protein_accessions": np.array([], dtype=object),
                "additional_scores": np.array([], dtype=object),
                "run_identifier": str(l_hit.get("run_identifier", "")),
            })

    out_df = pd.DataFrame(merged_rows)

    full_df = lucxor_df.copy()

    out_df = out_df.set_index(["peptide_identification_index", "hit_index"])
    full_df = full_df.set_index(["peptide_identification_index", "hit_index"])

    for col in out_df.columns:
        if col in full_df.columns and out_df[col].dtype != full_df[col].dtype:
            try:
                out_df[col] = out_df[col].astype(full_df[col].dtype)
            except Exception as e:
                click.echo(
                    f"Could not convert column '{col}' from {out_df[col].dtype} to {full_df[col].dtype}: {e}"
                )

    full_df.update(out_df)

    missing_mask = ~full_df.index.isin(out_df.index)
    full_df.loc[missing_mask, "score"] = np.nan
    full_df.loc[missing_mask, "score_type"] = "onsite_combined_score"

    out_df = full_df.reset_index()

    save_dataframes(output_file, out_df, proteins_df, template_df=lucxor_df, source_idparquet=input_idparquet)
    click.echo(f"Successfully merged {stats['merged']} peptide identifications")
    click.echo("Each peptide contains scores from all three algorithms")


def run_all_algorithms_from_single_cli(
    in_file, id_file, out_file,
    fragment_mass_tolerance, fragment_mass_unit,
    threads, debug, add_decoys,
):
    """Run all three algorithms when --compute-all-scores is specified."""
    try:
        start_time = time.time()
        click.echo(f"[{time.strftime('%H:%M:%S')}] --compute-all-scores: Running all algorithms")

        with tempfile.TemporaryDirectory() as tmpdir:
            ascore_out = os.path.join(tmpdir, "ascore_result.idparquet")
            phosphors_out = os.path.join(tmpdir, "phosphors_result.idparquet")

            # Run AScore
            from onsite.ascore.cli import ascore as ascore_func
            ctx = click.Context(ascore_func)
            ctx.invoke(ascore_func,
                in_file=in_file, id_file=id_file, out_file=ascore_out,
                fragment_mass_tolerance=fragment_mass_tolerance,
                fragment_mass_unit=fragment_mass_unit,
                threads=threads, debug=debug, add_decoys=add_decoys,
                compute_all_scores=False,
            )

            # Run PhosphoRS
            from onsite.phosphors.cli import phosphors as phosphors_func
            ctx = click.Context(phosphors_func)
            ctx.invoke(phosphors_func,
                in_file=in_file, id_file=id_file, out_file=phosphors_out,
                fragment_mass_tolerance=fragment_mass_tolerance,
                fragment_mass_unit=fragment_mass_unit,
                threads=threads, debug=debug, add_decoys=add_decoys,
                compute_all_scores=False,
            )

            # Run LucXor
            if add_decoys:
                target_mods = ("Phospho (S)", "Phospho (T)", "Phospho (Y)", "PhosphoDecoy (A)")
            else:
                target_mods = ("Phospho (S)", "Phospho (T)", "Phospho (Y)")

            from onsite.lucxor.cli import PyLuciPHOr2, setup_logging as lucxor_setup_logging
            lucxor_out = os.path.join(tmpdir, "lucxor_result.idparquet")
            lucxor_setup_logging(debug, None, lucxor_out)
            tool = PyLuciPHOr2()
            exit_code = tool.run(
                input_spectrum=in_file, input_id=id_file, output=lucxor_out,
                fragment_method="CID",
                fragment_mass_tolerance=fragment_mass_tolerance,
                fragment_error_units=fragment_mass_unit,
                min_mz=150.0, target_modifications=target_mods,
                neutral_losses=("sty -H3PO4 -97.97690",),
                decoy_mass=79.966331,
                decoy_neutral_losses=("X -H3PO4 -97.97690",),
                max_charge_state=5, max_peptide_length=40, max_num_perm=16384,
                modeling_score_threshold=0.95, scoring_threshold=0.0,
                min_num_psms_model=50, threads=threads, rt_tolerance=0.01,
                debug=debug, disable_split_by_charge=False,
            )
            if exit_code != 0:
                raise RuntimeError(f"LucXor failed with exit code {exit_code}")

            merge_algorithm_results(ascore_out, phosphors_out, lucxor_out, out_file, id_file)

        elapsed = time.time() - start_time
        click.echo(f"All algorithms completed in {elapsed:.2f}s")
        return 0

    except KeyboardInterrupt:
        click.echo("\nCancelled")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        import traceback; traceback.print_exc()
        sys.exit(1)


cli.add_command(all)


def main():
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nCancelled")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
