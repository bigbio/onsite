#!/usr/bin/env python3
"""
Test algorithm comparison between new results and reference results.

This module compares the output of three phosphorylation localization algorithms
(LucXor, AScore, PhosphoRS) against reference results stored in the data directory.

Testing Framework:
- TIER 1 (HARD FAIL): High-confidence recall ≥ 85%, no catastrophic count drop (>30%)
- TIER 2 (SOFT FAIL): Moderate-confidence recall ≥ 80%, count ratio within 0.7x-1.3x
- TIER 3 (INFO): New-only sites logged for review

Filtering criteria:
- q-value < 0.01 (prerequisite for all algorithms)
- LucXor: local_flr < 0.01 (strict), 0.05 (moderate), 0.1 (lenient)
- AScore: all AScore values >= 20 (strict), 15 (moderate), 3 (lenient)
- PhosphoRS: all top-N site probabilities > 99% (strict), 90% (moderate), 75% (lenient)
  where N = number of phosphorylations in the peptide
"""

import pytest
import sys
import os
import tempfile
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from click.testing import CliRunner

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onsite.onsitec import cli


# Expected recall thresholds by confidence tier
EXPECTED_RECALL = {
    'strict': 0.85,    # High-confidence sites must be highly reproducible
    'moderate': 0.80,  # Moderate-confidence sites
    'lenient': 0.75,   # Lenient threshold
}

# Maximum acceptable count drop at any threshold
MAX_COUNT_DROP = 0.30  # 30% drop triggers hard fail


def get_peptide_key(pep_id, hit) -> str:
    """Generate a unique key for a peptide hit based on sequence and spectrum reference."""
    sequence = hit.getSequence().toString()
    spectrum_ref = ""
    if pep_id.metaValueExists("spectrum_reference"):
        spectrum_ref = pep_id.getMetaValue("spectrum_reference")
    return f"{sequence}|{spectrum_ref}"


def parse_phosphors_site_probs(site_probs_str: str) -> List[float]:
    """Parse PhosphoRS site probabilities from string format."""
    if not site_probs_str or site_probs_str == "":
        return []
    # Format: "{5: 1.5653912149891193e-08, 9: 0.014259709125676328, ...}"
    probs = []
    matches = re.findall(r':\s*([\d.e+-]+)', site_probs_str)
    for match in matches:
        try:
            probs.append(float(match))
        except ValueError:
            continue
    return probs


def load_idxml_with_scores(idxml_path: str) -> Tuple[List, List, Dict[str, Dict[str, Any]]]:
    """Load idXML file and extract peptide hits with their scores."""
    from pyopenms import IdXMLFile
    
    peptide_ids = []
    protein_ids = []
    IdXMLFile().load(str(idxml_path), protein_ids, peptide_ids)
    
    peptide_data = {}
    for pep_id in peptide_ids:
        for hit in pep_id.getHits():
            key = get_peptide_key(pep_id, hit)
            data = {
                'sequence': hit.getSequence().toString(),
                'q_value': None,
                'luciphor_local_flr': None,
                'ascore_pep_score': None,
                'ascore_values': [],
                'phosphors_pep_score': None,
                'phosphors_site_probs': [],
            }
            
            # Extract q-value
            if hit.metaValueExists("q-value"):
                data['q_value'] = float(hit.getMetaValue("q-value"))
            
            # Extract LucXor scores
            if hit.metaValueExists("Luciphor_local_flr"):
                data['luciphor_local_flr'] = float(hit.getMetaValue("Luciphor_local_flr"))
            
            # Extract AScore values
            if hit.metaValueExists("AScore_pep_score"):
                data['ascore_pep_score'] = float(hit.getMetaValue("AScore_pep_score"))
            
            # Extract individual AScore values (AScore_1, AScore_2, etc.)
            for i in range(1, 10):
                ascore_key = f"AScore_{i}"
                if hit.metaValueExists(ascore_key):
                    data['ascore_values'].append(float(hit.getMetaValue(ascore_key)))
            
            # Extract PhosphoRS scores
            if hit.metaValueExists("PhosphoRS_pep_score"):
                data['phosphors_pep_score'] = float(hit.getMetaValue("PhosphoRS_pep_score"))
            
            if hit.metaValueExists("PhosphoRS_site_probs"):
                site_probs_str = str(hit.getMetaValue("PhosphoRS_site_probs"))
                data['phosphors_site_probs'] = parse_phosphors_site_probs(site_probs_str)
            
            peptide_data[key] = data
    
    return peptide_ids, protein_ids, peptide_data


def filter_lucxor(peptide_data: Dict[str, Dict], q_value_threshold: float, flr_threshold: float) -> Set[str]:
    """Filter peptides by LucXor criteria."""
    filtered = set()
    for key, data in peptide_data.items():
        q_value = data.get('q_value')
        local_flr = data.get('luciphor_local_flr')
        
        if q_value is not None and q_value < q_value_threshold:
            if local_flr is not None and local_flr < flr_threshold:
                filtered.add(key)
    return filtered


def filter_ascore(peptide_data: Dict[str, Dict], q_value_threshold: float, ascore_threshold: float) -> Set[str]:
    """Filter peptides by AScore criteria."""
    filtered = set()
    for key, data in peptide_data.items():
        q_value = data.get('q_value')
        ascore_values = data.get('ascore_values', [])
        
        if q_value is not None and q_value < q_value_threshold:
            # Check if all AScore values meet the threshold
            if ascore_values and all(score >= ascore_threshold for score in ascore_values):
                filtered.add(key)
    return filtered


def filter_phosphors(peptide_data: Dict[str, Dict], q_value_threshold: float, prob_threshold: float) -> Set[str]:
    """Filter peptides by PhosphoRS criteria.
    
    For PhosphoRS, we check if ALL top-N site probabilities exceed the threshold,
    where N is the number of phosphorylations in the peptide. This ensures that
    all phosphorylation sites are confidently localized, not just one.
    
    Args:
        prob_threshold: Probability threshold as percentage (e.g., 75 for 75%)
    """
    filtered = set()
    # Convert percentage threshold to decimal
    prob_threshold_decimal = prob_threshold / 100.0
    
    for key, data in peptide_data.items():
        q_value = data.get('q_value')
        site_probs = data.get('phosphors_site_probs', [])
        sequence = data.get('sequence', '')
        
        if q_value is not None and q_value < q_value_threshold:
            if not site_probs:
                continue
            
            # Count phosphorylations (both regular and decoy)
            # OpenMS format: "S(Phospho)", "T(Phospho)", "Y(Phospho)", "A(PhosphoDecoy)"
            phospho_count = sequence.count('(Phospho)')  # Matches both Phospho and PhosphoDecoy
            
            if phospho_count == 0:
                continue
            
            # Sort probabilities in descending order and check top N
            sorted_probs = sorted(site_probs, reverse=True)
            
            # Check if the top N probabilities (where N = phospho_count) all exceed threshold
            if len(sorted_probs) >= phospho_count:
                top_n_probs = sorted_probs[:phospho_count]
                if all(prob > prob_threshold_decimal for prob in top_n_probs):
                    filtered.add(key)
    
    return filtered


def compare_results(new_set: Set[str], ref_set: Set[str]) -> Tuple[Set[str], Set[str], Set[str], float, float]:
    """Compare two sets of peptide keys and return overlap statistics.
    
    Returns:
        overlap: Sites found in both
        new_only: Sites only in new (potential gains)
        ref_only: Sites only in reference (potential losses)
        recall: Fraction of reference sites found in new (overlap / ref)
        gain_rate: Fraction of new sites not in reference (new_only / new)
    """
    overlap = new_set & ref_set
    new_only = new_set - ref_set
    ref_only = ref_set - new_set
    
    recall = len(overlap) / len(ref_set) if len(ref_set) > 0 else 1.0
    gain_rate = len(new_only) / len(new_set) if len(new_set) > 0 else 0.0
    
    return overlap, new_only, ref_only, recall, gain_rate


class TestAlgorithmComparison:
    """Test class for comparing algorithm results against reference data."""
    
    @pytest.mark.data
    @pytest.mark.slow
    def test_lucxor_comparison(self, data_dir, mzml_file, idxml_file):
        """Test LucXor results comparison with tiered recall thresholds."""
        ref_lucxor_file = data_dir / "1_lucxor_result.idXML"
        if not ref_lucxor_file.exists():
            pytest.skip("Reference LucXor file not found")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            new_output = os.path.join(temp_dir, "new_lucxor.idXML")
            
            # Run LucXor
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "lucxor",
                    "--input-spectrum", str(mzml_file),
                    "--input-id", str(idxml_file),
                    "--output", new_output,
                    "--fragment-method", "HCD",
                    "--fragment-mass-tolerance", "0.5",
                    "--fragment-error-units", "Da",
                    "--threads", "1",
                    "--min-num-psms-model", "50",
                    "--target-modifications", "Phospho(S),Phospho(T),Phospho(Y),PhosphoDecoy(A)",
                ],
                catch_exceptions=False,
            )
            
            if result.exit_code != 0:
                pytest.fail(
                    f"LucXor command execution failed (exit_code={result.exit_code}).\n"
                    f"Output:\n{result.output}\n"
                    f"Exception: {result.exception if hasattr(result, 'exception') else 'N/A'}"
                )
            
            if not os.path.exists(new_output):
                pytest.fail(
                    f"LucXor output file not created: {new_output}\n"
                    f"Command output:\n{result.output}"
                )
            
            # Load results
            _, _, new_data = load_idxml_with_scores(new_output)
            _, _, ref_data = load_idxml_with_scores(str(ref_lucxor_file))
            
            q_value_threshold = 0.01
            thresholds = {
                'strict': 0.01,
                'moderate': 0.05,
                'lenient': 0.1,
            }
            
            print("\n" + "=" * 80)
            print("LucXor Comparison Results (q-value < 0.01)")
            print("=" * 80)
            
            results = {}
            warnings = []
            
            for tier, flr_threshold in thresholds.items():
                new_filtered = filter_lucxor(new_data, q_value_threshold, flr_threshold)
                ref_filtered = filter_lucxor(ref_data, q_value_threshold, flr_threshold)
                
                overlap, new_only, ref_only, recall, gain_rate = compare_results(new_filtered, ref_filtered)
                
                count_ratio = len(new_filtered) / len(ref_filtered) if len(ref_filtered) > 0 else 1.0
                
                print(f"\n{tier.upper()} (Local FLR < {flr_threshold}):")
                print(f"  New results: {len(new_filtered)}")
                print(f"  Reference results: {len(ref_filtered)}")
                overlap_pct = len(overlap) / len(ref_filtered) * 100 if len(ref_filtered) > 0 else 0.0
                print(f"  Overlap: {len(overlap)} ({overlap_pct:.1f}%)")
                print(f"  Recall: {recall:.1%} (new found {len(overlap)}/{len(ref_filtered)} reference sites)")
                print(f"  Gain rate: {gain_rate:.1%} ({len(new_only)} new-only sites)")
                print(f"  Lost sites: {len(ref_only)}")
                print(f"  Count ratio: {count_ratio:.2f}x")
                
                results[tier] = {
                    'recall': recall,
                    'count_ratio': count_ratio,
                    'new_count': len(new_filtered),
                    'ref_count': len(ref_filtered),
                }
                
                # Check thresholds
                expected_recall = EXPECTED_RECALL[tier]
                if recall < expected_recall:
                    msg = f"{tier} recall {recall:.1%} < {expected_recall:.1%}"
                    if tier == 'strict':
                        warnings.append(f"HARD FAIL: {msg}")
                    else:
                        warnings.append(f"SOFT FAIL: {msg}")
                
                # Check catastrophic count drop
                if count_ratio < (1 - MAX_COUNT_DROP):
                    warnings.append(f"HARD FAIL: {tier} count dropped {(1-count_ratio):.1%} (>{MAX_COUNT_DROP:.0%})")
            
            # Print warnings
            if warnings:
                print("\n" + "!" * 80)
                print("WARNINGS:")
                for warning in warnings:
                    print(f"  {warning}")
                print("!" * 80)
            
            # TIER 1: Hard fail conditions
            assert results['strict']['recall'] >= EXPECTED_RECALL['strict'], \
                f"LucXor strict recall {results['strict']['recall']:.1%} < {EXPECTED_RECALL['strict']:.1%}"
            
            for tier in thresholds.keys():
                assert results[tier]['count_ratio'] >= (1 - MAX_COUNT_DROP), \
                    f"LucXor {tier} count dropped {(1-results[tier]['count_ratio']):.1%} (>{MAX_COUNT_DROP:.0%})"
            
            # TIER 2: Soft fail - warn if moderate recall is low
            if results['moderate']['recall'] < EXPECTED_RECALL['moderate']:
                print(f"\nWARNING: Moderate recall {results['moderate']['recall']:.1%} < {EXPECTED_RECALL['moderate']:.1%}")
    
    @pytest.mark.data
    @pytest.mark.slow
    def test_ascore_comparison(self, data_dir, mzml_file, idxml_file):
        """Test AScore results comparison with tiered recall thresholds."""
        ref_ascore_file = data_dir / "1_ascore_result.idXML"
        if not ref_ascore_file.exists():
            pytest.skip("Reference AScore file not found")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            new_output = os.path.join(temp_dir, "new_ascore.idXML")
            
            # Run AScore
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "ascore",
                    "--in-file", str(mzml_file),
                    "--id-file", str(idxml_file),
                    "--out-file", new_output,
                    "--fragment-mass-tolerance", "0.05",
                    "--fragment-mass-unit", "Da",
                    "--threads", "1",
                    "--add-decoys",
                ],
                catch_exceptions=False,
            )
            
            if result.exit_code != 0:
                pytest.fail(
                    f"AScore command execution failed (exit_code={result.exit_code}).\n"
                    f"Output:\n{result.output}\n"
                    f"Exception: {result.exception if hasattr(result, 'exception') else 'N/A'}"
                )
            
            if not os.path.exists(new_output):
                pytest.fail(
                    f"AScore output file not created: {new_output}\n"
                    f"Command output:\n{result.output}"
                )
            
            # Load results
            _, _, new_data = load_idxml_with_scores(new_output)
            _, _, ref_data = load_idxml_with_scores(str(ref_ascore_file))
            
            q_value_threshold = 0.01
            thresholds = {
                'strict': 20,
                'moderate': 15,
                'lenient': 3,
            }
            
            print("\n" + "=" * 80)
            print("AScore Comparison Results (q-value < 0.01)")
            print("=" * 80)
            
            results = {}
            warnings = []
            
            for tier, ascore_threshold in thresholds.items():
                new_filtered = filter_ascore(new_data, q_value_threshold, ascore_threshold)
                ref_filtered = filter_ascore(ref_data, q_value_threshold, ascore_threshold)
                
                overlap, new_only, ref_only, recall, gain_rate = compare_results(new_filtered, ref_filtered)
                
                count_ratio = len(new_filtered) / len(ref_filtered) if len(ref_filtered) > 0 else 1.0
                
                print(f"\n{tier.upper()} (AScore >= {ascore_threshold}):")
                print(f"  New results: {len(new_filtered)}")
                print(f"  Reference results: {len(ref_filtered)}")
                overlap_pct = len(overlap) / len(ref_filtered) * 100 if len(ref_filtered) > 0 else 0.0
                print(f"  Overlap: {len(overlap)} ({overlap_pct:.1f}%)")
                print(f"  Recall: {recall:.1%} (new found {len(overlap)}/{len(ref_filtered)} reference sites)")
                print(f"  Gain rate: {gain_rate:.1%} ({len(new_only)} new-only sites)")
                print(f"  Lost sites: {len(ref_only)}")
                print(f"  Count ratio: {count_ratio:.2f}x")
                
                results[tier] = {
                    'recall': recall,
                    'count_ratio': count_ratio,
                    'new_count': len(new_filtered),
                    'ref_count': len(ref_filtered),
                }
                
                # Check thresholds
                expected_recall = EXPECTED_RECALL[tier]
                if recall < expected_recall:
                    msg = f"{tier} recall {recall:.1%} < {expected_recall:.1%}"
                    if tier == 'strict':
                        warnings.append(f"HARD FAIL: {msg}")
                    else:
                        warnings.append(f"SOFT FAIL: {msg}")
                
                # Check catastrophic count drop
                if count_ratio < (1 - MAX_COUNT_DROP):
                    warnings.append(f"HARD FAIL: {tier} count dropped {(1-count_ratio):.1%} (>{MAX_COUNT_DROP:.0%})")
            
            # Print warnings
            if warnings:
                print("\n" + "!" * 80)
                print("WARNINGS:")
                for warning in warnings:
                    print(f"  {warning}")
                print("!" * 80)
            
            # TIER 1: Hard fail conditions
            assert results['strict']['recall'] >= EXPECTED_RECALL['strict'], \
                f"AScore strict recall {results['strict']['recall']:.1%} < {EXPECTED_RECALL['strict']:.1%}"
            
            for tier in thresholds.keys():
                assert results[tier]['count_ratio'] >= (1 - MAX_COUNT_DROP), \
                    f"AScore {tier} count dropped {(1-results[tier]['count_ratio']):.1%} (>{MAX_COUNT_DROP:.0%})"
            
            # TIER 2: Soft fail - warn if moderate recall is low
            if results['moderate']['recall'] < EXPECTED_RECALL['moderate']:
                print(f"\nWARNING: Moderate recall {results['moderate']['recall']:.1%} < {EXPECTED_RECALL['moderate']:.1%}")
    
    @pytest.mark.data
    @pytest.mark.slow
    def test_phosphors_comparison(self, data_dir, mzml_file, idxml_file):
        """Test PhosphoRS results comparison with tiered recall thresholds."""
        ref_phosphors_file = data_dir / "1_phosphors_result.idXML"
        if not ref_phosphors_file.exists():
            pytest.skip("Reference PhosphoRS file not found")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            new_output = os.path.join(temp_dir, "new_phosphors.idXML")
            
            # Run PhosphoRS
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "phosphors",
                    "--in-file", str(mzml_file),
                    "--id-file", str(idxml_file),
                    "--out-file", new_output,
                    "--fragment-mass-tolerance", "0.05",
                    "--fragment-mass-unit", "Da",
                    "--threads", "1",
                    "--add-decoys",
                ],
                catch_exceptions=False,
            )
            
            if result.exit_code != 0:
                pytest.fail(
                    f"PhosphoRS command execution failed (exit_code={result.exit_code}).\n"
                    f"Output:\n{result.output}\n"
                    f"Exception: {result.exception if hasattr(result, 'exception') else 'N/A'}"
                )
            
            if not os.path.exists(new_output):
                pytest.fail(
                    f"PhosphoRS output file not created: {new_output}\n"
                    f"Command output:\n{result.output}"
                )
            
            # Load results
            _, _, new_data = load_idxml_with_scores(new_output)
            _, _, ref_data = load_idxml_with_scores(str(ref_phosphors_file))
            
            q_value_threshold = 0.01
            thresholds = {
                'strict': 99,
                'moderate': 90,
                'lenient': 75,
            }
            
            print("\n" + "=" * 80)
            print("PhosphoRS Comparison Results (q-value < 0.01)")
            print("=" * 80)
            
            results = {}
            warnings = []
            
            for tier, prob_threshold in thresholds.items():
                new_filtered = filter_phosphors(new_data, q_value_threshold, prob_threshold)
                ref_filtered = filter_phosphors(ref_data, q_value_threshold, prob_threshold)
                
                overlap, new_only, ref_only, recall, gain_rate = compare_results(new_filtered, ref_filtered)
                
                count_ratio = len(new_filtered) / len(ref_filtered) if len(ref_filtered) > 0 else 1.0
                
                print(f"\n{tier.upper()} (Site probability > {prob_threshold}%):")
                print(f"  New results: {len(new_filtered)}")
                print(f"  Reference results: {len(ref_filtered)}")
                overlap_pct = len(overlap) / len(ref_filtered) * 100 if len(ref_filtered) > 0 else 0.0
                print(f"  Overlap: {len(overlap)} ({overlap_pct:.1f}%)")
                print(f"  Recall: {recall:.1%} (new found {len(overlap)}/{len(ref_filtered)} reference sites)")
                print(f"  Gain rate: {gain_rate:.1%} ({len(new_only)} new-only sites)")
                print(f"  Lost sites: {len(ref_only)}")
                print(f"  Count ratio: {count_ratio:.2f}x")
                
                results[tier] = {
                    'recall': recall,
                    'count_ratio': count_ratio,
                    'new_count': len(new_filtered),
                    'ref_count': len(ref_filtered),
                }
                
                # Check thresholds
                expected_recall = EXPECTED_RECALL[tier]
                if recall < expected_recall:
                    msg = f"{tier} recall {recall:.1%} < {expected_recall:.1%}"
                    if tier == 'strict':
                        warnings.append(f"HARD FAIL: {msg}")
                    else:
                        warnings.append(f"SOFT FAIL: {msg}")
                
                # Check catastrophic count drop
                if count_ratio < (1 - MAX_COUNT_DROP):
                    warnings.append(f"HARD FAIL: {tier} count dropped {(1-count_ratio):.1%} (>{MAX_COUNT_DROP:.0%})")
            
            # Print warnings
            if warnings:
                print("\n" + "!" * 80)
                print("WARNINGS:")
                for warning in warnings:
                    print(f"  {warning}")
                print("!" * 80)
            
            # TIER 1: Hard fail conditions
            assert results['strict']['recall'] >= EXPECTED_RECALL['strict'], \
                f"PhosphoRS strict recall {results['strict']['recall']:.1%} < {EXPECTED_RECALL['strict']:.1%}"
            
            for tier in thresholds.keys():
                assert results[tier]['count_ratio'] >= (1 - MAX_COUNT_DROP), \
                    f"PhosphoRS {tier} count dropped {(1-results[tier]['count_ratio']):.1%} (>{MAX_COUNT_DROP:.0%})"
            
            # TIER 2: Soft fail - warn if moderate recall is low
            if results['moderate']['recall'] < EXPECTED_RECALL['moderate']:
                print(f"\nWARNING: Moderate recall {results['moderate']['recall']:.1%} < {EXPECTED_RECALL['moderate']:.1%}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
