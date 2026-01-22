"""
Global FLR (False Localization Rate) calculator for cross-file analysis.

This module aggregates FLR data from multiple processed idXML files and
calculates global FLR across all files.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PSMRecord:
    """Record of a PSM for global FLR calculation."""
    file_path: str
    delta_score: float
    is_decoy: bool
    original_global_flr: float = 1.0
    original_local_flr: float = 1.0
    cross_file_global_flr: float = 1.0
    cross_file_local_flr: float = 1.0


@dataclass
class FileSummary:
    """Summary statistics for a single file."""
    file_path: str
    total_psms: int = 0
    decoy_psms: int = 0
    target_psms: int = 0
    passing_counts: Dict[float, int] = field(default_factory=dict)


class GlobalFLRCalculator:
    """
    Calculator for cross-file global FLR.

    Aggregates PSM data from multiple processed idXML files and calculates
    FLR using kernel density estimation across all files.
    """

    def __init__(self, min_delta_score: float = 0.1):
        """
        Initialize the GlobalFLRCalculator.

        Args:
            min_delta_score: Minimum delta score threshold for FLR calculation
        """
        self.min_delta_score = min_delta_score
        self.psm_records: List[PSMRecord] = []
        self.file_summaries: Dict[str, FileSummary] = {}

        # FLR calculation results
        self.sorted_delta_scores: Optional[np.ndarray] = None
        self.sorted_global_flr: Optional[np.ndarray] = None
        self.sorted_local_flr: Optional[np.ndarray] = None

        logger.debug(f"GlobalFLRCalculator initialized with min_delta_score={min_delta_score}")

    def extract_from_idxml(self, idxml_path: str) -> int:
        """
        Extract Luciphor_delta_score and decoy status from a processed idXML file.

        Args:
            idxml_path: Path to the processed idXML file

        Returns:
            Number of PSM records extracted
        """
        from pyopenms import IdXMLFile, PeptideIdentificationList

        prot_ids = []
        pep_ids = PeptideIdentificationList()
        IdXMLFile().load(idxml_path, prot_ids, pep_ids)

        extracted_count = 0
        file_summary = FileSummary(file_path=idxml_path)

        for pep_id in pep_ids:
            hits = pep_id.getHits()
            if not hits:
                continue

            hit = hits[0]
            file_summary.total_psms += 1

            # Get delta score
            delta_score = float('nan')
            if hit.metaValueExists("Luciphor_delta_score"):
                delta_score = float(hit.getMetaValue("Luciphor_delta_score"))
            elif pep_id.getScoreType() == "Luciphor_delta_score":
                delta_score = float(hit.getScore())

            if np.isnan(delta_score):
                continue

            # Determine decoy status
            is_decoy = False
            if hit.metaValueExists("target_decoy"):
                target_decoy = hit.getMetaValue("target_decoy")
                is_decoy = str(target_decoy).lower() in ["decoy", "1", "true"]

            # Check sequence for decoy modification (PhosphoDecoy)
            seq_str = hit.getSequence().toString()
            if "(PhosphoDecoy)" in seq_str:
                is_decoy = True

            if is_decoy:
                file_summary.decoy_psms += 1
            else:
                file_summary.target_psms += 1

            # Get original FLR values if available
            original_global_flr = 1.0
            original_local_flr = 1.0
            if hit.metaValueExists("Luciphor_global_flr"):
                original_global_flr = float(hit.getMetaValue("Luciphor_global_flr"))
            if hit.metaValueExists("Luciphor_local_flr"):
                original_local_flr = float(hit.getMetaValue("Luciphor_local_flr"))

            # Create PSM record
            record = PSMRecord(
                file_path=idxml_path,
                delta_score=delta_score,
                is_decoy=is_decoy,
                original_global_flr=original_global_flr,
                original_local_flr=original_local_flr
            )
            self.psm_records.append(record)
            extracted_count += 1

        self.file_summaries[idxml_path] = file_summary
        logger.info(
            f"Extracted {extracted_count} PSMs from {idxml_path} "
            f"(targets: {file_summary.target_psms}, decoys: {file_summary.decoy_psms})"
        )
        return extracted_count

    def calculate_global_flr(self) -> None:
        """
        Calculate FLR using KDE on aggregated data from all files.

        This reuses the FLR calculation logic from onsite.lucxor.flr.FLRCalculator
        but applies it to aggregated data from multiple files.
        """
        from onsite.lucxor.flr import FLRCalculator

        # Filter records by minimum delta score
        valid_records = [
            r for r in self.psm_records
            if r.delta_score > self.min_delta_score
        ]

        if not valid_records:
            logger.warning("No valid PSM records for global FLR calculation")
            return

        # Separate target and decoy records
        target_scores = [r.delta_score for r in valid_records if not r.is_decoy]
        decoy_scores = [r.delta_score for r in valid_records if r.is_decoy]

        n_target = len(target_scores)
        n_decoy = len(decoy_scores)

        logger.info(f"Global FLR calculation: {n_target} target PSMs, {n_decoy} decoy PSMs")

        if n_target < 2 or n_decoy < 2:
            logger.warning(
                f"Insufficient PSMs for global FLR calculation "
                f"(need at least 2 targets and 2 decoys)"
            )
            return

        # Create FLR calculator and add PSMs
        flr_calc = FLRCalculator(min_delta_score=self.min_delta_score)

        for score in target_scores:
            flr_calc.add_psm(score, is_decoy=False)
        for score in decoy_scores:
            flr_calc.add_psm(score, is_decoy=True)

        # Perform FLR calculation
        flr_calc.prep_arrays()
        flr_calc.initialize_tick_marks()

        from onsite.lucxor.constants import DECOY, REAL
        flr_calc.eval_tick_marks(DECOY)
        flr_calc.eval_tick_marks(REAL)

        flr_calc.calc_both_fdrs()
        flr_calc.set_minor_maps()
        flr_calc.perform_minorization()

        # Store sorted FLR mapping
        flr_data = []
        for i in range(len(flr_calc.pos)):
            if i < len(flr_calc.global_fdr) and i < len(flr_calc.local_fdr):
                flr_data.append({
                    'delta_score': float(flr_calc.pos[i]),
                    'global_flr': min(1.0, float(flr_calc.global_fdr[i])),
                    'local_flr': min(1.0, float(flr_calc.local_fdr[i]))
                })

        flr_data.sort(key=lambda x: x['delta_score'])

        self.sorted_delta_scores = np.array([d['delta_score'] for d in flr_data])
        self.sorted_global_flr = np.array([d['global_flr'] for d in flr_data])
        self.sorted_local_flr = np.array([d['local_flr'] for d in flr_data])

        # Assign cross-file FLR to all records
        for record in self.psm_records:
            if record.is_decoy:
                record.cross_file_global_flr = float('nan')
                record.cross_file_local_flr = float('nan')
            elif record.delta_score <= self.min_delta_score:
                record.cross_file_global_flr = 1.0
                record.cross_file_local_flr = 1.0
            else:
                global_flr, local_flr = self._lookup_flr(record.delta_score)
                record.cross_file_global_flr = global_flr
                record.cross_file_local_flr = local_flr

        logger.info("Global FLR calculation completed")

    def _lookup_flr(self, delta_score: float) -> Tuple[float, float]:
        """
        Look up FLR values for a given delta score using binary search.

        Args:
            delta_score: The delta score to look up

        Returns:
            Tuple of (global_flr, local_flr)
        """
        if self.sorted_delta_scores is None or len(self.sorted_delta_scores) == 0:
            return (1.0, 1.0)

        # Binary search for closest delta score
        idx = np.searchsorted(self.sorted_delta_scores, delta_score)

        if idx == 0:
            closest_idx = 0
        elif idx >= len(self.sorted_delta_scores):
            closest_idx = len(self.sorted_delta_scores) - 1
        else:
            dist_left = abs(delta_score - self.sorted_delta_scores[idx - 1])
            dist_right = abs(delta_score - self.sorted_delta_scores[idx])
            closest_idx = (idx - 1) if dist_left <= dist_right else idx

        return (
            float(self.sorted_global_flr[closest_idx]),
            float(self.sorted_local_flr[closest_idx])
        )

    def get_counts_at_threshold(self, threshold: float) -> Dict[str, Dict[str, int]]:
        """
        Get counts of PSMs passing a global FLR threshold per file.

        Args:
            threshold: Global FLR threshold

        Returns:
            Dictionary mapping file paths to count statistics
        """
        counts = {}

        for file_path in self.file_summaries:
            file_records = [r for r in self.psm_records if r.file_path == file_path]
            passing = sum(
                1 for r in file_records
                if not r.is_decoy and r.cross_file_global_flr <= threshold
            )
            total_target = sum(1 for r in file_records if not r.is_decoy)

            counts[file_path] = {
                'total': len(file_records),
                'total_target': total_target,
                'passing': passing
            }

            # Store in file summary
            self.file_summaries[file_path].passing_counts[threshold] = passing

        # Calculate totals
        total_psms = sum(c['total'] for c in counts.values())
        total_target = sum(c['total_target'] for c in counts.values())
        total_passing = sum(c['passing'] for c in counts.values())

        counts['TOTAL'] = {
            'total': total_psms,
            'total_target': total_target,
            'passing': total_passing
        }

        return counts

    def plot_cumulative_flr_curve(self, output_path: str) -> None:
        """
        Generate a cumulative FLR curve plot.

        Args:
            output_path: Path to save the plot (PNG or PDF)

        Raises:
            ImportError: If matplotlib is not installed
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with: pip install matplotlib"
            )

        if self.sorted_delta_scores is None or len(self.sorted_delta_scores) == 0:
            logger.warning("No FLR data available for plotting")
            return

        # Get target records sorted by delta score
        target_records = sorted(
            [r for r in self.psm_records if not r.is_decoy],
            key=lambda r: r.delta_score,
            reverse=True  # Higher delta score = better
        )

        if not target_records:
            logger.warning("No target PSMs for plotting")
            return

        # Calculate cumulative counts at different FLR thresholds
        thresholds = np.linspace(0, 0.2, 100)
        cumulative_counts = []

        for threshold in thresholds:
            count = sum(1 for r in target_records if r.cross_file_global_flr <= threshold)
            cumulative_counts.append(count)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(thresholds * 100, cumulative_counts, 'b-', linewidth=2)
        ax.set_xlabel('Global FLR Threshold (%)', fontsize=12)
        ax.set_ylabel('Number of Localized PSMs', fontsize=12)
        ax.set_title('Cumulative FLR Curve (Cross-File)', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add vertical lines at common thresholds
        for t, color in [(0.01, 'green'), (0.05, 'orange'), (0.1, 'red')]:
            count = sum(1 for r in target_records if r.cross_file_global_flr <= t)
            ax.axvline(x=t * 100, color=color, linestyle='--', alpha=0.7)
            ax.annotate(
                f'{t*100:.0f}% FLR: {count}',
                xy=(t * 100, count),
                xytext=(t * 100 + 1, count),
                fontsize=9
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Cumulative FLR curve saved to: {output_path}")

    def generate_summary_report(
        self,
        output_path: str,
        threshold: float = 0.01
    ) -> None:
        """
        Generate a TSV summary report with per-file and total counts.

        Args:
            output_path: Path to save the report (TSV)
            threshold: Primary FLR threshold to report (default: 0.01)
        """
        # Calculate counts at multiple thresholds
        thresholds = [0.001, 0.005, 0.01, 0.05, 0.1]

        # Ensure we have counts for all thresholds
        for t in thresholds:
            self.get_counts_at_threshold(t)

        total_psms = sum(s.total_psms for s in self.file_summaries.values())
        total_target = sum(s.target_psms for s in self.file_summaries.values())

        with open(output_path, 'w') as f:
            # Header comments
            f.write(f"# Global FLR Summary Report\n")
            f.write(f"# Files processed: {len(self.file_summaries)}\n")
            f.write(f"# Total PSMs: {total_psms}\n")
            f.write(f"# Total target PSMs: {total_target}\n")
            f.write(f"#\n")

            # Per-file summary
            f.write("# Per-file Summary\n")
            headers = ["file_path", "total_psms", "target_psms", "decoy_psms"]
            for t in thresholds:
                headers.append(f"passing_{t:.3f}")
            f.write("\t".join(headers) + "\n")

            for file_path, summary in self.file_summaries.items():
                row = [
                    os.path.basename(file_path),
                    str(summary.total_psms),
                    str(summary.target_psms),
                    str(summary.decoy_psms)
                ]
                for t in thresholds:
                    count = summary.passing_counts.get(t, 0)
                    row.append(str(count))
                f.write("\t".join(row) + "\n")

            # Total row
            total_row = ["TOTAL", str(total_psms), str(total_target)]
            total_decoy = sum(s.decoy_psms for s in self.file_summaries.values())
            total_row.append(str(total_decoy))
            for t in thresholds:
                total_passing = sum(
                    s.passing_counts.get(t, 0) for s in self.file_summaries.values()
                )
                total_row.append(str(total_passing))
            f.write("\t".join(total_row) + "\n")

            # Threshold analysis
            f.write("\n# Threshold Analysis\n")
            f.write("threshold\ttotal_passing\tpercentage\n")
            for t in thresholds:
                total_passing = sum(
                    s.passing_counts.get(t, 0) for s in self.file_summaries.values()
                )
                percentage = (total_passing / total_target * 100) if total_target > 0 else 0
                f.write(f"{t:.3f}\t{total_passing}\t{percentage:.1f}%\n")

        logger.info(f"Global FLR summary report saved to: {output_path}")

    def update_idxml_files(self, idxml_paths: List[str]) -> None:
        """
        Update idXML files with cross-file global FLR metadata.

        Adds Luciphor_cross_file_global_flr and Luciphor_cross_file_local_flr
        metadata to each peptide hit.

        Args:
            idxml_paths: List of idXML file paths to update
        """
        from pyopenms import IdXMLFile, PeptideIdentificationList

        for idxml_path in idxml_paths:
            if idxml_path not in self.file_summaries:
                logger.warning(f"Skipping {idxml_path}: not in extracted files")
                continue

            # Load the file
            prot_ids = []
            pep_ids = PeptideIdentificationList()
            IdXMLFile().load(idxml_path, prot_ids, pep_ids)

            # Get records for this file
            file_records = {
                r.delta_score: r
                for r in self.psm_records
                if r.file_path == idxml_path
            }

            # Update peptide hits
            updated_count = 0
            for pep_id in pep_ids:
                hits = pep_id.getHits()
                if not hits:
                    continue

                hit = hits[0]

                # Get delta score
                delta_score = float('nan')
                if hit.metaValueExists("Luciphor_delta_score"):
                    delta_score = float(hit.getMetaValue("Luciphor_delta_score"))
                elif pep_id.getScoreType() == "Luciphor_delta_score":
                    delta_score = float(hit.getScore())

                if np.isnan(delta_score):
                    continue

                # Find matching record
                record = file_records.get(delta_score)
                if record:
                    hit.setMetaValue(
                        "Luciphor_cross_file_global_flr",
                        record.cross_file_global_flr
                    )
                    hit.setMetaValue(
                        "Luciphor_cross_file_local_flr",
                        record.cross_file_local_flr
                    )
                    updated_count += 1

                # Update the hits
                pep_id.setHits([hit])

            # Save the updated file
            IdXMLFile().store(idxml_path, prot_ids, pep_ids)
            logger.info(f"Updated {updated_count} PSMs in {idxml_path}")

    def get_total_counts(self) -> Dict[str, int]:
        """
        Get total counts across all files.

        Returns:
            Dictionary with total_psms, target_psms, decoy_psms
        """
        return {
            'total_psms': sum(s.total_psms for s in self.file_summaries.values()),
            'target_psms': sum(s.target_psms for s in self.file_summaries.values()),
            'decoy_psms': sum(s.decoy_psms for s in self.file_summaries.values())
        }
