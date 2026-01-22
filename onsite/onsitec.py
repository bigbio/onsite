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
            )
            
            # Run LucXor with appropriate target modifications
            click.echo(f"\n{'='*60}")
            click.echo(f"[{time.strftime('%H:%M:%S')}] Running LucXor...")
            click.echo(f"{'='*60}")
            from onsite.lucxor.cli import lucxor as lucxor_func
            
            # Configure target modifications based on add_decoys flag
            if add_decoys:
                target_mods = ("Phospho (S)", "Phospho (T)", "Phospho (Y)", "PhosphoDecoy (A)")
                click.echo("  Using target modifications with decoys: Phospho(S), Phospho(T), Phospho(Y), PhosphoDecoy(A)")
            else:
                target_mods = ("Phospho (S)", "Phospho (T)", "Phospho (Y)")
            
            ctx = click.Context(lucxor_func)
            ctx.invoke(
                lucxor_func,
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
                log_file=None,
                disable_split_by_charge=False,
            )
            
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
        
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


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
    
    if len(ascore_pep_ids) != len(phosphors_pep_ids) or len(ascore_pep_ids) != len(lucxor_pep_ids):
        click.echo(f"Warning: Different number of peptide IDs in results:")
        click.echo(f"  AScore: {len(ascore_pep_ids)}")
        click.echo(f"  PhosphoRS: {len(phosphors_pep_ids)}")
        click.echo(f"  LucXor: {len(lucxor_pep_ids)}")
    
    # Create merged results
    merged_pep_ids = []
    
    for i in range(min(len(ascore_pep_ids), len(phosphors_pep_ids), len(lucxor_pep_ids))):
        ascore_pid = ascore_pep_ids[i]
        phosphors_pid = phosphors_pep_ids[i]
        lucxor_pid = lucxor_pep_ids[i]
        
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
            
            # Add Luciphor metadata (in order)
            merged_hit.setMetaValue("Luciphor_sequence", lucxor_hit.getSequence().toString())
            merged_hit.setMetaValue("Luciphor_delta_score", float(lucxor_hit.getScore()))
            if lucxor_hit.metaValueExists("Luciphor_pep_score"):
                merged_hit.setMetaValue("Luciphor_pep_score", float(lucxor_hit.getMetaValue("Luciphor_pep_score")))
            if lucxor_hit.metaValueExists("Luciphor_global_flr"):
                merged_hit.setMetaValue("Luciphor_global_flr", float(lucxor_hit.getMetaValue("Luciphor_global_flr")))
            if lucxor_hit.metaValueExists("Luciphor_local_flr"):
                merged_hit.setMetaValue("Luciphor_local_flr", float(lucxor_hit.getMetaValue("Luciphor_local_flr")))
            
            # Set combined score (use LucXor delta score as primary)
            combined_score = float(lucxor_hit.getScore())
            merged_hit.setScore(combined_score)
            
            merged_hits.append(merged_hit)
        
        merged_pid.setHits(merged_hits)
        merged_pep_ids.append(merged_pid)
    
    # Save merged results
    IdXMLFile().store(output_file, lucxor_prot_ids, merged_pep_ids)
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


@click.command()
@click.option(
    "-in",
    "--in-file",
    "in_files",
    multiple=True,
    required=True,
    type=click.Path(exists=True),
    help="Input mzML file(s). Specify multiple files with repeated -in options.",
)
@click.option(
    "-id",
    "--id-file",
    "id_files",
    multiple=True,
    required=True,
    type=click.Path(exists=True),
    help="Input idXML file(s). Specify multiple files with repeated -id options.",
)
@click.option(
    "-out",
    "--out-file",
    "out_files",
    multiple=True,
    required=False,
    type=click.Path(),
    help="Output idXML file(s). If not specified, generates <input>_localized.idXML.",
)
@click.option(
    "--global-flr-report",
    "global_flr_report",
    type=click.Path(),
    help="Output path for global FLR summary report (TSV).",
)
@click.option(
    "--global-flr-threshold",
    "global_flr_threshold",
    type=float,
    default=0.01,
    help="Global FLR threshold for filtering (default: 0.01).",
)
@click.option(
    "--global-flr-plot",
    "global_flr_plot",
    type=click.Path(),
    help="Output path for cumulative FLR curve plot (requires matplotlib).",
)
@click.option(
    "--update-with-global-flr",
    "update_with_global_flr",
    is_flag=True,
    help="Update idXML files with cross-file global FLR metadata.",
)
@click.option(
    "--fragment-mass-tolerance",
    "fragment_mass_tolerance",
    type=float,
    default=0.05,
    help="Fragment mass tolerance value (default: 0.05).",
)
@click.option(
    "--fragment-mass-unit",
    "fragment_mass_unit",
    type=click.Choice(["Da", "ppm"]),
    default="Da",
    help="Tolerance unit (default: Da).",
)
@click.option(
    "--threads",
    "threads",
    type=int,
    default=1,
    help="Number of parallel threads (default: 1).",
)
@click.option(
    "--debug",
    "debug",
    is_flag=True,
    help="Enable debug output.",
)
@click.option(
    "--add-decoys",
    "add_decoys",
    is_flag=True,
    default=False,
    help="Include A (PhosphoDecoy) as potential phosphorylation site for FLR estimation.",
)
def batch(
    in_files,
    id_files,
    out_files,
    global_flr_report,
    global_flr_threshold,
    global_flr_plot,
    update_with_global_flr,
    fragment_mass_tolerance,
    fragment_mass_unit,
    threads,
    debug,
    add_decoys,
):
    """
    Process multiple file pairs with all algorithms and calculate global FLR.

    This command processes multiple mzML/idXML file pairs, running all three
    phosphorylation site localization algorithms (AScore, PhosphoRS, LucXor)
    on each pair, then calculates global FLR across all files.

    Examples:
      # Process 2 file pairs
      onsite batch -in file1.mzML -id file1.idXML -in file2.mzML -id file2.idXML

      # Process with global FLR report
      onsite batch -in file1.mzML -id file1.idXML -in file2.mzML -id file2.idXML \\
        --global-flr-report summary.tsv --add-decoys

      # Process with specific output files
      onsite batch -in file1.mzML -id file1.idXML -out file1_out.idXML \\
        -in file2.mzML -id file2.idXML -out file2_out.idXML
    """
    from onsite.validation import FileValidator, FileValidationError
    from onsite.global_flr import GlobalFLRCalculator

    try:
        start_time = time.time()
        click.echo(f"[{time.strftime('%H:%M:%S')}] Starting OnSite batch processing")
        click.echo(f"  Input files: {len(in_files)} mzML, {len(id_files)} idXML")
        click.echo(f"  Fragment tolerance: {fragment_mass_tolerance} {fragment_mass_unit}")
        click.echo(f"  Threads: {threads}")
        click.echo(f"  Add decoys: {add_decoys}")
        click.echo(f"  Debug: {debug}")

        # Step 1: Validate file pairs
        click.echo(f"\n[{time.strftime('%H:%M:%S')}] Validating file pairs...")
        try:
            file_pairs = FileValidator.validate_file_pairs(in_files, id_files)
        except FileValidationError as e:
            click.echo(f"Error: {str(e)}", err=True)
            sys.exit(1)

        # Step 2: Generate output paths if needed
        try:
            output_paths = FileValidator.validate_output_paths(
                out_files, in_files, len(file_pairs)
            )
        except FileValidationError as e:
            click.echo(f"Error: {str(e)}", err=True)
            sys.exit(1)

        click.echo(f"  Validated {len(file_pairs)} file pair(s)")

        # Step 3: Process each file pair
        processed_outputs = []
        for i, ((mzml_path, idxml_path), out_path) in enumerate(
            zip(file_pairs, output_paths), 1
        ):
            click.echo(f"\n{'='*60}")
            click.echo(f"[{time.strftime('%H:%M:%S')}] Processing file {i}/{len(file_pairs)}: {os.path.basename(mzml_path)}")
            click.echo(f"{'='*60}")

            # Create temporary directory for intermediate results
            with tempfile.TemporaryDirectory() as tmpdir:
                ascore_out = os.path.join(tmpdir, "ascore_result.idXML")
                phosphors_out = os.path.join(tmpdir, "phosphors_result.idXML")
                lucxor_out = os.path.join(tmpdir, "lucxor_result.idXML")

                # Run AScore
                click.echo(f"\n[{time.strftime('%H:%M:%S')}] Running AScore...")
                from onsite.ascore.cli import ascore as ascore_func
                ctx = click.Context(ascore_func)
                ctx.invoke(
                    ascore_func,
                    in_file=mzml_path,
                    id_file=idxml_path,
                    out_file=ascore_out,
                    fragment_mass_tolerance=fragment_mass_tolerance,
                    fragment_mass_unit=fragment_mass_unit,
                    threads=threads,
                    debug=debug,
                    add_decoys=add_decoys,
                )

                # Run PhosphoRS
                click.echo(f"\n[{time.strftime('%H:%M:%S')}] Running PhosphoRS...")
                from onsite.phosphors.cli import phosphors as phosphors_func
                ctx = click.Context(phosphors_func)
                ctx.invoke(
                    phosphors_func,
                    in_file=mzml_path,
                    id_file=idxml_path,
                    out_file=phosphors_out,
                    fragment_mass_tolerance=fragment_mass_tolerance,
                    fragment_mass_unit=fragment_mass_unit,
                    threads=threads,
                    debug=debug,
                    add_decoys=add_decoys,
                )

                # Run LucXor
                click.echo(f"\n[{time.strftime('%H:%M:%S')}] Running LucXor...")
                from onsite.lucxor.cli import PyLuciPHOr2, setup_logging as lucxor_setup_logging

                # Setup logging for LucXor
                lucxor_setup_logging(debug, None, lucxor_out)

                # Configure target modifications based on add_decoys flag
                if add_decoys:
                    target_mods = ("Phospho (S)", "Phospho (T)", "Phospho (Y)", "PhosphoDecoy (A)")
                else:
                    target_mods = ("Phospho (S)", "Phospho (T)", "Phospho (Y)")

                # Create tool instance and run
                tool = PyLuciPHOr2()
                exit_code = tool.run(
                    input_spectrum=mzml_path,
                    input_id=idxml_path,
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
                    click.echo(f"Warning: LucXor failed for {mzml_path} with exit code {exit_code}")
                    continue

                # Merge results
                click.echo(f"\n[{time.strftime('%H:%M:%S')}] Merging algorithm results...")
                merge_algorithm_results(ascore_out, phosphors_out, lucxor_out, out_path)
                processed_outputs.append(out_path)

            click.echo(f"[{time.strftime('%H:%M:%S')}] Completed file {i}/{len(file_pairs)}: {out_path}")

        # Step 4: Calculate global FLR if requested
        if global_flr_report or global_flr_plot or update_with_global_flr:
            click.echo(f"\n{'='*60}")
            click.echo(f"[{time.strftime('%H:%M:%S')}] Calculating global FLR across all files...")
            click.echo(f"{'='*60}")

            global_flr_calc = GlobalFLRCalculator(min_delta_score=0.1)

            # Extract data from all processed files
            for out_path in processed_outputs:
                click.echo(f"  Extracting data from: {os.path.basename(out_path)}")
                global_flr_calc.extract_from_idxml(out_path)

            # Calculate global FLR
            global_flr_calc.calculate_global_flr()

            # Get counts at specified threshold
            counts = global_flr_calc.get_counts_at_threshold(global_flr_threshold)
            total_counts = global_flr_calc.get_total_counts()

            click.echo(f"\n  Global FLR Results (threshold={global_flr_threshold}):")
            click.echo(f"    Total PSMs: {total_counts['total_psms']}")
            click.echo(f"    Target PSMs: {total_counts['target_psms']}")
            click.echo(f"    Decoy PSMs: {total_counts['decoy_psms']}")
            click.echo(f"    Passing at {global_flr_threshold*100:.1f}% FLR: {counts['TOTAL']['passing']}")

            # Generate report if requested
            if global_flr_report:
                click.echo(f"\n  Generating FLR report: {global_flr_report}")
                global_flr_calc.generate_summary_report(global_flr_report, global_flr_threshold)

            # Generate plot if requested
            if global_flr_plot:
                click.echo(f"\n  Generating FLR plot: {global_flr_plot}")
                try:
                    global_flr_calc.plot_cumulative_flr_curve(global_flr_plot)
                except ImportError as e:
                    click.echo(f"  Warning: {str(e)}")

            # Update idXML files if requested
            if update_with_global_flr:
                click.echo(f"\n  Updating idXML files with cross-file global FLR...")
                global_flr_calc.update_idxml_files(processed_outputs)

        # Print summary
        elapsed = time.time() - start_time
        click.echo(f"\n{'='*60}")
        click.echo(f"Batch processing completed!")
        click.echo(f"  Files processed: {len(processed_outputs)}/{len(file_pairs)}")
        click.echo(f"  Total time: {elapsed:.2f} seconds")
        click.echo(f"{'='*60}")

        for i, out_path in enumerate(processed_outputs, 1):
            click.echo(f"  Output {i}: {os.path.abspath(out_path)}")

    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# Add the 'batch' command to the CLI group
cli.add_command(batch)


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
