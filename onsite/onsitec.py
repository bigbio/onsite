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
from pyopenms import IdXMLFile, PeptideIdentification, PeptideHit, PeptideIdentificationList
from onsite.lucxor.cli import lucxor
from onsite.phosphors.cli import phosphors
from onsite.ascore.cli import ascore
from onsite.mzid_adapter import (
    load_identifications, store_identifications, has_alanine, validate_spectrum_refs,
)


@click.group()
@click.version_option(version="0.0.1")
def cli():
    """
    OnSite: Mass spectrometry post-translational modification localization tool

    Available algorithms:
      ascore      AScore algorithm for phosphorylation site localization
      phosphors   PhosphoRS algorithm for phosphorylation site localization
      lucxor      LucXor (LuciPHOr2) algorithm for PTM localization
      all         Run all three algorithms and merge results

    Examples:
      onsite ascore -in spectra.mzML -id identifications.idXML -out results.idXML
      onsite phosphors -in spectra.mzML -id identifications.idXML -out results.idXML
      onsite lucxor -in spectra.mzML -id identifications.idXML -out results.idXML
      onsite all -in spectra.mzML -id identifications.idXML -out results.idXML --add-decoys
    """
    pass


# Add the individual CLI commands to the main CLI group
cli.add_command(ascore)
cli.add_command(phosphors)
cli.add_command(lucxor)


def run_all_localizers(
    in_file,
    id_file,
    out_file,
    fragment_mass_tolerance=0.05,
    fragment_mass_unit="Da",
    threads=1,
    add_decoys=False,
    debug=False,
):
    """Run AScore + PhosphoRS + LucXor into a tempdir and merge into out_file (idXML).

    Extracted from the `all` command so other entry points (e.g. the mzid
    adapter) can reuse the exact same orchestration. Behavior is unchanged.
    """
    # Load identifications once for the decoy guard + spectrum-reference validation.
    _prot, _pep = load_identifications(id_file)
    validate_spectrum_refs(_pep, in_file)
    if add_decoys and not has_alanine(_pep):
        click.echo("Warning: --add-decoys set but no Alanine candidate present; "
                   "computing the no-decoy score pack instead.")
        add_decoys = False

    with tempfile.TemporaryDirectory() as tmpdir:
        ascore_out = os.path.join(tmpdir, "ascore_result.idXML")
        phosphors_out = os.path.join(tmpdir, "phosphors_result.idXML")
        lucxor_out = os.path.join(tmpdir, "lucxor_result.idXML")

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
            click.echo("  Using target modifications with decoys: Phospho(S), Phospho(T), Phospho(Y), PhosphoDecoy(A)")
        else:
            target_mods = ("Phospho (S)", "Phospho (T)", "Phospho (Y)")
        ctx = click.Context(lucxor_func)
        try:
            ctx.invoke(
                lucxor_func,
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
                debug=debug, log_file=None, disable_split_by_charge=False,
            )
        except SystemExit as exc:
            if exc.code != 0:
                raise RuntimeError(f"LucXor failed with exit code {exc.code}") from exc

        # Merge results
        click.echo(f"\n{'='*60}")
        click.echo(f"[{time.strftime('%H:%M:%S')}] Merging results from all algorithms...")
        click.echo(f"{'='*60}")
        merge_algorithm_results(ascore_out, phosphors_out, lucxor_out, out_file)


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
    """
    Run all three algorithms (AScore, PhosphoRS, LucXor) and merge results.
    
    This command runs all three phosphorylation site localization algorithms
    sequentially and merges their scores and site assignments into a single
    output file. Each algorithm's specific scores and metadata are preserved.
    
    When --add-decoys is specified:
    - AScore and PhosphoRS: Include A (PhosphoDecoy) as potential site
    - LucXor: Use target modifications "Phospho(S), Phospho(T), Phospho(Y), PhosphoDecoy(A)"
    """
    try:
        start_time = time.time()
        click.echo(f"[{time.strftime('%H:%M:%S')}] Starting OnSite with all algorithms")
        click.echo(f"  Input spectrum: {in_file}")
        click.echo(f"  Input ID: {id_file}")
        click.echo(f"  Output: {out_file}")
        click.echo(f"  Fragment tolerance: {fragment_mass_tolerance} {fragment_mass_unit}")
        click.echo(f"  Threads: {threads}")
        click.echo(f"  Add decoys: {add_decoys}")
        click.echo(f"  Debug: {debug}")
        
        run_all_localizers(
            in_file, id_file, out_file,
            fragment_mass_tolerance, fragment_mass_unit, threads, add_decoys, debug,
        )

        elapsed = time.time() - start_time
        abs_out_file = os.path.abspath(out_file)
        click.echo(f"\n{'='*60}")
        click.echo(f"All algorithms completed successfully!")
        click.echo(f"  Total time: {elapsed:.2f} seconds")
        click.echo(f"  Output saved to: {abs_out_file}")
        click.echo(f"{'='*60}")
        
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _join_psms_by_ref(ascore_pep_ids, phosphors_pep_ids, lucxor_pep_ids):
    """
    Match PeptideIdentifications across the three tools by spectrum_reference.

    Merging by list position is unsafe: the tools drop/reorder PSMs
    independently (LucXor keepNBestHits / spectrum-miss / min-PSM abort; AScore
    error-drops), so a single drop shifts every later index and would fuse
    scores from different peptides.

    Returns (triples, stats), where triples is a list of
    (ascore_pid, phosphors_pid, lucxor_pid) for spectra present in ALL three
    tools — in LucXor order, with the unmodified backbone verified identical —
    and stats reports the per-tool exclusions.
    """
    def by_ref(pep_ids):
        d = {}
        for pid in pep_ids:
            if pid.metaValueExists("spectrum_reference"):
                d[pid.getMetaValue("spectrum_reference")] = pid
        return d

    a_map, p_map, l_map = (
        by_ref(ascore_pep_ids),
        by_ref(phosphors_pep_ids),
        by_ref(lucxor_pep_ids),
    )
    common = set(a_map) & set(p_map) & set(l_map)

    def unmod(pid):
        hits = pid.getHits()
        return hits[0].getSequence().toUnmodifiedString() if hits else None

    triples = []
    seen = set()
    seq_mismatch = 0
    for lpid in lucxor_pep_ids:  # preserve LucXor order
        if not lpid.metaValueExists("spectrum_reference"):
            continue
        ref = lpid.getMetaValue("spectrum_reference")
        if ref not in common or ref in seen:
            continue
        seen.add(ref)
        apid, ppid = a_map[ref], p_map[ref]
        if not (unmod(apid) == unmod(ppid) == unmod(lpid)):
            seq_mismatch += 1
            continue
        triples.append((apid, ppid, lpid))

    stats = {
        "ascore_dropped": len(a_map) - len(common),
        "phosphors_dropped": len(p_map) - len(common),
        "lucxor_dropped": len(l_map) - len(common),
        "seq_mismatch": seq_mismatch,
        "merged": len(triples),
    }
    return triples, stats


def merge_algorithm_results(ascore_file, phosphors_file, lucxor_file, output_file):
    """
    Merge results from all three algorithms into a single idXML file.
    
    This function combines the scores and metadata from AScore, PhosphoRS, and LucXor
    into a single output file. Each peptide hit will contain:
    - All algorithm-specific scores and metadata
    - The best sequence assignment from each algorithm
    - A combined score (using LucXor's delta score as primary)
    """
    # Load all three result files
    ascore_prot_ids, ascore_pep_ids = [], PeptideIdentificationList()
    phosphors_prot_ids, phosphors_pep_ids = [], PeptideIdentificationList()
    lucxor_prot_ids, lucxor_pep_ids = [], PeptideIdentificationList()

    IdXMLFile().load(ascore_file, ascore_prot_ids, ascore_pep_ids)
    IdXMLFile().load(phosphors_file, phosphors_prot_ids, phosphors_pep_ids)
    IdXMLFile().load(lucxor_file, lucxor_prot_ids, lucxor_pep_ids)
    
    # Match PSMs across tools by spectrum_reference (NOT list position).
    triples, stats = _join_psms_by_ref(ascore_pep_ids, phosphors_pep_ids, lucxor_pep_ids)
    for tool in ("ascore", "phosphors", "lucxor"):
        if stats[f"{tool}_dropped"]:
            click.echo(
                f"  Note: {stats[f'{tool}_dropped']} {tool} PSM(s) not present in all tools; excluded from merge"
            )
    if stats["seq_mismatch"]:
        click.echo(
            f"  Warning: {stats['seq_mismatch']} PSM(s) skipped due to backbone-sequence mismatch across tools"
        )

    # Create merged results (typed container required by IdXMLFile().store)
    merged_pep_ids = PeptideIdentificationList()

    for ascore_pid, phosphors_pid, lucxor_pid in triples:
        # Create new PeptideIdentification based on LucXor result (as it has FLR)
        merged_pid = PeptideIdentification(lucxor_pid)
        merged_pid.setScoreType("onsite_combined_score")
        merged_pid.setHigherScoreBetter(True)
        
        # Merge hits
        merged_hits = []
        for j in range(min(len(ascore_pid.getHits()), len(phosphors_pid.getHits()), len(lucxor_pid.getHits()))):
            ascore_hit = ascore_pid.getHits()[j]
            phosphors_hit = phosphors_pid.getHits()[j]
            lucxor_hit = lucxor_pid.getHits()[j]
            
            # Create a new merged hit based on LucXor hit (for basic properties)
            merged_hit = PeptideHit(lucxor_hit)
            
            # Clear all existing algorithm-specific metadata to rebuild in correct order
            # First, collect non-algorithm metadata to preserve
            preserved_meta = {}
            meta_keys_to_preserve = ["target_decoy", "consensus_support", "Posterior Error Probability_score", "q-value"]
            for key in meta_keys_to_preserve:
                if merged_hit.metaValueExists(key):
                    preserved_meta[key] = merged_hit.getMetaValue(key)
            
            # Remove all metadata
            keys_to_remove = []
            merged_hit.getKeys(keys_to_remove)
            for key in keys_to_remove:
                try:
                    merged_hit.removeMetaValue(key)
                except:
                    pass
            
            # Restore preserved metadata first (in original order)
            for key in meta_keys_to_preserve:
                if key in preserved_meta:
                    merged_hit.setMetaValue(key, preserved_meta[key])
            
            # Add AScore metadata (in order)
            merged_hit.setMetaValue("AScore_sequence", ascore_hit.getSequence().toString())
            merged_hit.setMetaValue("AScore_best_score", float(ascore_hit.getScore()))
            if ascore_hit.metaValueExists("AScore_pep_score"):
                merged_hit.setMetaValue("AScore_pep_score", float(ascore_hit.getMetaValue("AScore_pep_score")))
            
            # Copy all AScore site scores
            if ascore_hit.metaValueExists("AScore_site_scores"):
                merged_hit.setMetaValue("AScore_site_scores", ascore_hit.getMetaValue("AScore_site_scores"))
            rank = 1
            while ascore_hit.metaValueExists(f"AScore_{rank}"):
                merged_hit.setMetaValue(f"AScore_{rank}", float(ascore_hit.getMetaValue(f"AScore_{rank}")))
                rank += 1
            
            # Add PhosphoRS metadata (in order)
            merged_hit.setMetaValue("PhosphoRS_sequence", phosphors_hit.getSequence().toString())
            merged_hit.setMetaValue("PhosphoRS_score", float(phosphors_hit.getScore()))
            if phosphors_hit.metaValueExists("PhosphoRS_pep_score"):
                merged_hit.setMetaValue("PhosphoRS_pep_score", float(phosphors_hit.getMetaValue("PhosphoRS_pep_score")))
            if phosphors_hit.metaValueExists("PhosphoRS_site_probs"):
                merged_hit.setMetaValue("PhosphoRS_site_probs", phosphors_hit.getMetaValue("PhosphoRS_site_probs"))
            if phosphors_hit.metaValueExists("PhosphoRS_site_delta"):
                merged_hit.setMetaValue("PhosphoRS_site_delta", phosphors_hit.getMetaValue("PhosphoRS_site_delta"))
            
            # Add Luciphor metadata (in order)
            merged_hit.setMetaValue("Luciphor_sequence", lucxor_hit.getSequence().toString())
            merged_hit.setMetaValue("Luciphor_delta_score", float(lucxor_hit.getScore()))
            if lucxor_hit.metaValueExists("Luciphor_pep_score"):
                merged_hit.setMetaValue("Luciphor_pep_score", float(lucxor_hit.getMetaValue("Luciphor_pep_score")))
            if lucxor_hit.metaValueExists("Luciphor_global_flr"):
                merged_hit.setMetaValue("Luciphor_global_flr", float(lucxor_hit.getMetaValue("Luciphor_global_flr")))
            if lucxor_hit.metaValueExists("Luciphor_local_flr"):
                merged_hit.setMetaValue("Luciphor_local_flr", float(lucxor_hit.getMetaValue("Luciphor_local_flr")))
            if lucxor_hit.metaValueExists("Luciphor_site_scores"):
                merged_hit.setMetaValue("Luciphor_site_scores", lucxor_hit.getMetaValue("Luciphor_site_scores"))
            
            # Set combined score (use LucXor delta score as primary)
            combined_score = float(lucxor_hit.getScore())
            merged_hit.setScore(combined_score)
            
            merged_hits.append(merged_hit)
        
        merged_pid.setHits(merged_hits)
        merged_pep_ids.push_back(merged_pid)
    
    # Save merged results (format-agnostic: idXML or mzIdentML by extension)
    store_identifications(output_file, lucxor_prot_ids, merged_pep_ids)
    click.echo(f"Successfully merged {len(merged_pep_ids)} peptide identifications")
    click.echo(f"  Each peptide contains scores from all three algorithms:")
    click.echo(f"    - AScore: site-specific scores")
    click.echo(f"    - PhosphoRS: site probabilities")
    click.echo(f"    - LucXor: delta scores and FLR values")


def run_all_algorithms_from_single_cli(
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
    Run all three algorithms when --compute-all-scores is specified.
    This function is called from individual algorithm CLIs.
    """
    try:
        start_time = time.time()
        click.echo(f"[{time.strftime('%H:%M:%S')}] --compute-all-scores enabled: Running all three algorithms")
        click.echo(f"  Input spectrum: {in_file}")
        click.echo(f"  Input ID: {id_file}")
        click.echo(f"  Output: {out_file}")
        click.echo(f"  Fragment tolerance: {fragment_mass_tolerance} {fragment_mass_unit}")
        click.echo(f"  Threads: {threads}")
        click.echo(f"  Add decoys: {add_decoys}")
        
        # Create temporary directory for intermediate results
        with tempfile.TemporaryDirectory() as tmpdir:
            ascore_out = os.path.join(tmpdir, "ascore_result.idXML")
            phosphors_out = os.path.join(tmpdir, "phosphors_result.idXML")
            lucxor_out = os.path.join(tmpdir, "lucxor_result.idXML")
            
            # Run AScore
            click.echo(f"\n{'='*60}")
            click.echo(f"[{time.strftime('%H:%M:%S')}] Running AScore...")
            click.echo(f"{'='*60}")
            from onsite.ascore.cli import ascore as ascore_func
            ctx = click.Context(ascore_func)
            ctx.invoke(
                ascore_func,
                in_file=in_file,
                id_file=id_file,
                out_file=ascore_out,
                fragment_mass_tolerance=fragment_mass_tolerance,
                fragment_mass_unit=fragment_mass_unit,
                threads=threads,
                debug=debug,
                add_decoys=add_decoys,
                compute_all_scores=False,  # Prevent recursion
            )
            
            # Run PhosphoRS
            click.echo(f"\n{'='*60}")
            click.echo(f"[{time.strftime('%H:%M:%S')}] Running PhosphoRS...")
            click.echo(f"{'='*60}")
            from onsite.phosphors.cli import phosphors as phosphors_func
            ctx = click.Context(phosphors_func)
            ctx.invoke(
                phosphors_func,
                in_file=in_file,
                id_file=id_file,
                out_file=phosphors_out,
                fragment_mass_tolerance=fragment_mass_tolerance,
                fragment_mass_unit=fragment_mass_unit,
                threads=threads,
                debug=debug,
                add_decoys=add_decoys,
                compute_all_scores=False,  # Prevent recursion
            )
            
            # Run LucXor with appropriate target modifications
            click.echo(f"\n{'='*60}")
            click.echo(f"[{time.strftime('%H:%M:%S')}] Running LucXor...")
            click.echo(f"{'='*60}")
            
            # Configure target modifications based on add_decoys flag
            if add_decoys:
                target_mods = ("Phospho (S)", "Phospho (T)", "Phospho (Y)", "PhosphoDecoy (A)")
                click.echo("  Using target modifications with decoys: Phospho(S), Phospho(T), Phospho(Y), PhosphoDecoy(A)")
            else:
                target_mods = ("Phospho (S)", "Phospho (T)", "Phospho (Y)")
            
            # Import and call LucXor directly to avoid sys.exit() issues
            from onsite.lucxor.cli import PyLuciPHOr2, setup_logging as lucxor_setup_logging
            
            # Setup logging for LucXor
            lucxor_setup_logging(debug, None, lucxor_out)
            
            # Create tool instance and run
            tool = PyLuciPHOr2()
            exit_code = tool.run(
                input_spectrum=in_file,
                input_id=id_file,
                output=lucxor_out,
                fragment_method="CID",
                fragment_mass_tolerance=fragment_mass_tolerance,
                fragment_error_units=fragment_mass_unit,
                min_mz=150.0,
                target_modifications=target_mods,
                neutral_losses=("sty -H3PO4 -97.97690",),
                decoy_mass=79.966331,
                decoy_neutral_losses=("X -H3PO4 -97.97690",),
                max_charge_state=5,
                max_peptide_length=40,
                max_num_perm=16384,
                modeling_score_threshold=0.95,
                scoring_threshold=0.0,
                min_num_psms_model=50,
                threads=threads,
                rt_tolerance=0.01,
                debug=debug,
                disable_split_by_charge=False,
            )
            
            if exit_code != 0:
                raise RuntimeError(f"LucXor failed with exit code {exit_code}")
            
            # Merge results
            click.echo(f"\n{'='*60}")
            click.echo(f"[{time.strftime('%H:%M:%S')}] Merging results from all algorithms...")
            click.echo(f"{'='*60}")
            merge_algorithm_results(ascore_out, phosphors_out, lucxor_out, out_file)
            
        elapsed = time.time() - start_time
        abs_out_file = os.path.abspath(out_file)
        click.echo(f"\n{'='*60}")
        click.echo(f"All algorithms completed successfully!")
        click.echo(f"  Total time: {elapsed:.2f} seconds")
        click.echo(f"  Output saved to: {abs_out_file}")
        click.echo(f"{'='*60}")
        
        return 0  # Success
        
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# Add the 'all' command to the CLI group
cli.add_command(all)


def main():
    """Main entry point for OnSite CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
