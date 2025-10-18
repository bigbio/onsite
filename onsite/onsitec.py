#!/usr/bin/env python3
"""
OnSite CLI - Integrated command line interface for phosphorylation site localization tools.

This module provides a unified interface for accessing AScore, PhosphoRS, and LucXor algorithms.
"""

import click
import sys
import os
from typing import List, Optional

@click.group()
@click.version_option(version='0.0.1')
def cli():
    """
    OnSite: Mass spectrometry post-translational modification localization tool
    
    Available algorithms:
      ascore      AScore algorithm for phosphorylation site localization
      phosphors   PhosphoRS algorithm for phosphorylation site localization  
      lucxor      LucXor (LuciPHOr2) algorithm for PTM localization
    
    Examples:
      onsite ascore -in spectra.mzML -id identifications.idXML -out results.idXML
      onsite phosphors -in spectra.mzML -id identifications.idXML -out results.idXML
      onsite lucxor -in spectra.mzML -id identifications.idXML -out results.idXML
    """
    pass

@cli.command()
@click.option('-in', '--in-file', 'in_file', required=True, 
              help='Input mzML file path', type=click.Path(exists=True))
@click.option('-id', '--id-file', 'id_file', required=True,
              help='Input idXML file path', type=click.Path(exists=True))
@click.option('-out', '--out-file', 'out_file', required=True,
              help='Output idXML file path', type=click.Path())
@click.option('--fragment-mass-tolerance', type=float, default=0.05,
              help='Fragment mass tolerance value (default: 0.05)')
@click.option('--fragment-mass-unit', type=click.Choice(['Da', 'ppm']), default='Da',
              help='Tolerance unit (default: Da)')
@click.option('--threads', type=int, default=4,
              help='Number of parallel threads (default: 4)')
@click.option('--debug', is_flag=True,
              help='Enable debug output and write debug log')
@click.option('--add-decoys', is_flag=True, default=False,
              help='Include A (PhosphoDecoy) as potential phosphorylation site')
def ascore(in_file, id_file, out_file, fragment_mass_tolerance, fragment_mass_unit, 
           threads, debug, add_decoys):
    """AScore algorithm for phosphorylation site localization."""
    run_ascore(in_file, id_file, out_file, fragment_mass_tolerance, fragment_mass_unit, 
               threads, debug, add_decoys)

@cli.command()
@click.option('-in', '--in-file', 'in_file', required=True,
              help='Input mzML file path', type=click.Path(exists=True))
@click.option('-id', '--id-file', 'id_file', required=True,
              help='Input idXML file path', type=click.Path(exists=True))
@click.option('-out', '--out-file', 'out_file', required=True,
              help='Output idXML file path', type=click.Path())
@click.option('--fragment-mass-tolerance', type=float, default=0.05,
              help='Fragment mass tolerance value (default: 0.05)')
@click.option('--fragment-mass-unit', type=click.Choice(['Da', 'ppm']), default='Da',
              help='Tolerance unit (default: Da)')
@click.option('--threads', type=int, default=1,
              help='Number of parallel processes (default: 1)')
@click.option('--debug', is_flag=True,
              help='Enable debug output and write debug log')
@click.option('--add-decoys', is_flag=True, default=False,
              help='Include A (PhosphoDecoy) as potential phosphorylation site')
def phosphors(in_file, id_file, out_file, fragment_mass_tolerance, fragment_mass_unit,
              threads, debug, add_decoys):
    """PhosphoRS algorithm for phosphorylation site localization."""
    run_phosphors(in_file, id_file, out_file, fragment_mass_tolerance, fragment_mass_unit,
                  threads, debug, add_decoys)

@cli.command()
@click.option('-in', '--input-spectrum', 'input_spectrum', required=True,
              help='Input spectrum file (mzML)', type=click.Path(exists=True))
@click.option('-id', '--input-id', 'input_id', required=True,
              help='Input identification file (idXML)', type=click.Path(exists=True))
@click.option('-out', '--output', 'output', required=True,
              help='Output file (idXML)', type=click.Path())
@click.option('--fragment-method', type=click.Choice(['CID', 'HCD']), default='CID',
              help='Fragmentation method (default: CID)')
@click.option('--fragment-mass-tolerance', type=float, default=0.5,
              help='Tolerance of the peaks in the fragment spectrum (default: 0.5)')
@click.option('--fragment-error-units', type=click.Choice(['Da', 'ppm']), default='Da',
              help='Unit of fragment mass tolerance (default: Da)')
@click.option('--min-mz', type=float, default=150.0,
              help='Do not consider peaks below this value (default: 150.0)')
@click.option('--target-modifications', multiple=True,
              default=["Phospho (S)", "Phospho (T)", "Phospho (Y)"],
              help='List of target modifications (default: Phospho (S) Phospho (T) Phospho (Y))')
@click.option('--neutral-losses', multiple=True,
              default=["sty -H3PO4 -97.97690"],
              help='List of neutral losses (default: sty -H3PO4 -97.97690)')
@click.option('--decoy-mass', type=float, default=79.966331,
              help='Mass to add for decoy generation (default: 79.966331)')
@click.option('--decoy-neutral-losses', multiple=True,
              default=["X -H3PO4 -97.97690"],
              help='List of decoy neutral losses (default: X -H3PO4 -97.97690)')
@click.option('--max-charge-state', type=int, default=5,
              help='Maximum charge state to consider (default: 5)')
@click.option('--max-peptide-length', type=int, default=40,
              help='Maximum peptide length (default: 40)')
@click.option('--max-num-perm', type=int, default=16384,
              help='Maximum number of permutations (default: 16384)')
@click.option('--modeling-score-threshold', type=float, default=0.95,
              help='Minimum score for modeling (default: 0.95)')
@click.option('--scoring-threshold', type=float, default=0.0,
              help='Minimum score threshold (default: 0.0)')
@click.option('--min-num-psms-model', type=int, default=50,
              help='Minimum number of PSMs for modeling (default: 50)')
@click.option('--threads', type=int, default=4,
              help='Number of threads to use (default: 4)')
@click.option('--rt-tolerance', type=float, default=0.01,
              help='Retention time tolerance (default: 0.01)')
@click.option('--debug', is_flag=True,
              help='Enable debug mode')
@click.option('--log-file', type=str,
              help='Log file path (only used in debug mode)')
def lucxor(input_spectrum, input_id, output, fragment_method, fragment_mass_tolerance,
           fragment_error_units, min_mz, target_modifications, neutral_losses,
           decoy_mass, decoy_neutral_losses, max_charge_state, max_peptide_length,
           max_num_perm, modeling_score_threshold, scoring_threshold,
           min_num_psms_model, threads, rt_tolerance, debug, log_file):
    """LucXor (LuciPHOr2) algorithm for PTM localization."""
    run_lucxor(input_spectrum, input_id, output, fragment_method, fragment_mass_tolerance,
               fragment_error_units, min_mz, target_modifications, neutral_losses,
               decoy_mass, decoy_neutral_losses, max_charge_state, max_peptide_length,
               max_num_perm, modeling_score_threshold, scoring_threshold,
               min_num_psms_model, threads, rt_tolerance, debug, log_file)

def run_ascore(in_file, id_file, out_file, fragment_mass_tolerance, fragment_mass_unit,
               threads, debug, add_decoys):
    """Run AScore algorithm."""
    # Import and run phospho_scoring.py
    from onsite.phospho_scoring import main as ascore_main
    
    # Set up sys.argv for phospho_scoring
    original_argv = sys.argv[:]
    sys.argv = ['phospho_scoring.py']
    
    # Add arguments
    sys.argv.extend(['-in', in_file])
    sys.argv.extend(['-id', id_file])
    sys.argv.extend(['-out', out_file])
    sys.argv.extend(['-fragment_mass_tolerance', str(fragment_mass_tolerance)])
    sys.argv.extend(['-fragment_mass_unit', fragment_mass_unit])
    sys.argv.extend(['-threads', str(threads)])
    if debug:
        sys.argv.append('-debug')
    if add_decoys:
        sys.argv.append('--add_decoys')
    
    try:
        ascore_main()
    finally:
        sys.argv = original_argv

def run_phosphors(in_file, id_file, out_file, fragment_mass_tolerance, fragment_mass_unit,
                  threads, debug, add_decoys):
    """Run PhosphoRS algorithm."""
    # Import and run phosphors_scoring.py
    from onsite.phosphors.phosphors_scoring import main as phosphors_main
    
    # Set up sys.argv for phosphors_scoring
    original_argv = sys.argv[:]
    sys.argv = ['phosphors_scoring.py']
    
    # Add arguments
    sys.argv.extend(['-in', in_file])
    sys.argv.extend(['-id', id_file])
    sys.argv.extend(['-out', out_file])
    sys.argv.extend(['-fragment_mass_tolerance', str(fragment_mass_tolerance)])
    sys.argv.extend(['-fragment_mass_unit', fragment_mass_unit])
    sys.argv.extend(['-threads', str(threads)])
    if debug:
        sys.argv.append('-debug')
    if add_decoys:
        sys.argv.append('--add_decoys')
    
    try:
        phosphors_main()
    finally:
        sys.argv = original_argv

def run_lucxor(input_spectrum, input_id, output, fragment_method, fragment_mass_tolerance,
               fragment_error_units, min_mz, target_modifications, neutral_losses,
               decoy_mass, decoy_neutral_losses, max_charge_state, max_peptide_length,
               max_num_perm, modeling_score_threshold, scoring_threshold,
               min_num_psms_model, threads, rt_tolerance, debug, log_file):
    """Run LucXor algorithm."""
    # Import and run lucxor CLI
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lucxor.cli import main as lucxor_main
    
    # Set up sys.argv for LucXor
    original_argv = sys.argv[:]
    sys.argv = ['lucxor/cli.py']
    
    # Add arguments
    sys.argv.extend(['-in', input_spectrum])
    sys.argv.extend(['-id', input_id])
    sys.argv.extend(['-out', output])
    sys.argv.extend(['--fragment_method', fragment_method])
    sys.argv.extend(['--fragment_mass_tolerance', str(fragment_mass_tolerance)])
    sys.argv.extend(['--fragment_error_units', fragment_error_units])
    sys.argv.extend(['--min_mz', str(min_mz)])
    sys.argv.extend(['--target_modifications'] + list(target_modifications))
    sys.argv.extend(['--neutral_losses'] + list(neutral_losses))
    sys.argv.extend(['--decoy_mass', str(decoy_mass)])
    sys.argv.extend(['--decoy_neutral_losses'] + list(decoy_neutral_losses))
    sys.argv.extend(['--max_charge_state', str(max_charge_state)])
    sys.argv.extend(['--max_peptide_length', str(max_peptide_length)])
    sys.argv.extend(['--max_num_perm', str(max_num_perm)])
    sys.argv.extend(['--modeling_score_threshold', str(modeling_score_threshold)])
    sys.argv.extend(['--scoring_threshold', str(scoring_threshold)])
    sys.argv.extend(['--min_num_psms_model', str(min_num_psms_model)])
    sys.argv.extend(['--threads', str(threads)])
    sys.argv.extend(['--rt_tolerance', str(rt_tolerance)])
    if debug:
        sys.argv.append('--debug')
    if log_file:
        sys.argv.extend(['--log_file', log_file])
    
    try:
        lucxor_main()
    finally:
        sys.argv = original_argv

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

if __name__ == '__main__':
    main()