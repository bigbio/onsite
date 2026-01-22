"""
Global FLR (False Localization Rate) calculator for cross-file analysis.

This module aggregates FLR data from multiple processed idXML files and
calculates global FLR across all files using direct Phospho/PhosphoDecoy
counting from peptide sequences.

FLR = PhosphoDecoy / (Phospho + PhosphoDecoy)
"""

import logging
import os
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Nature journal color scheme
NATURE_COLORS = {
    'primary': ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4'],
    'secondary': ['#B09C85', '#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F'],
    'categorical': ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#B09C85', '#DC0000']
}


@dataclass
class PeptideRecord:
    """Record of a peptide for global FLR calculation."""
    file_path: str
    sequence: str
    charge: int
    phospho_count: int
    phosphodecoy_count: int
    total_sites: int
    is_unambiguous: bool = False


@dataclass
class FileSummary:
    """Summary statistics for a single file."""
    file_path: str
    total_peptides: int = 0
    phospho_sites: int = 0
    phosphodecoy_sites: int = 0
    unambiguous_peptides: int = 0
    passing_counts: Dict[float, int] = field(default_factory=dict)


class GlobalFLRCalculator:
    """
    Calculator for cross-file global FLR.

    Aggregates peptide data from multiple processed idXML files and calculates
    FLR using direct Phospho/PhosphoDecoy counting, processed in order of
    phosphorylation site count (1->2->3).
    """

    def __init__(self, min_sites: int = 0):
        """
        Initialize the GlobalFLRCalculator.

        Args:
            min_sites: Minimum site count threshold for filtering
        """
        self.min_sites = min_sites
        self.peptide_records: List[PeptideRecord] = []
        self.file_summaries: Dict[str, FileSummary] = {}

        # FLR calculation results
        self.cumulative_data: List[Dict] = []

        logger.debug(f"GlobalFLRCalculator initialized with min_sites={min_sites}")

    @staticmethod
    def count_phospho_sites(sequence: str) -> int:
        """Count (Phospho) modifications in sequence."""
        return len(re.findall(r'\(Phospho\)', sequence))

    @staticmethod
    def count_phosphodecoy_sites(sequence: str) -> int:
        """Count (PhosphoDecoy) modifications in sequence."""
        return len(re.findall(r'\(PhosphoDecoy\)', sequence))

    @staticmethod
    def is_unambiguous(sequence: str) -> bool:
        """
        Determine if peptide has unambiguous sites.

        A peptide is unambiguous if all potential phosphorylation sites (S/T/Y/A)
        are already phosphorylated, meaning there's no ambiguity in localization.

        Args:
            sequence: Peptide sequence with modifications

        Returns:
            True if unambiguous (all potential sites are modified)
        """
        if not isinstance(sequence, str) or sequence.lower() == 'nan':
            return False

        phospho = sequence.count('(Phospho)')
        phosphodecoy = sequence.count('(PhosphoDecoy)')
        total_phospho_sites = phospho + phosphodecoy

        if total_phospho_sites == 0:
            return False

        # Calculate potential phosphorylation sites (S/T/Y/A)
        # Remove modification tags first to count amino acids
        clean_seq = re.sub(r'\([^)]+\)', '', sequence)
        potential = sum(1 for c in clean_seq if c in 'STYA')

        return total_phospho_sites == potential

    def extract_from_idxml(self, idxml_path: str) -> int:
        """
        Extract peptide sequences from a processed idXML file.

        Args:
            idxml_path: Path to the processed idXML file

        Returns:
            Number of peptide records extracted
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
            sequence = hit.getSequence().toString()
            charge = hit.getCharge()

            # Count phosphorylation sites
            phospho_count = self.count_phospho_sites(sequence)
            phosphodecoy_count = self.count_phosphodecoy_sites(sequence)
            total_sites = phospho_count + phosphodecoy_count

            # Skip if no phosphorylation
            if total_sites == 0:
                continue

            file_summary.total_peptides += 1
            file_summary.phospho_sites += phospho_count
            file_summary.phosphodecoy_sites += phosphodecoy_count

            # Check if unambiguous
            is_unamb = self.is_unambiguous(sequence)
            if is_unamb:
                file_summary.unambiguous_peptides += 1

            # Create peptide record
            record = PeptideRecord(
                file_path=idxml_path,
                sequence=sequence,
                charge=charge,
                phospho_count=phospho_count,
                phosphodecoy_count=phosphodecoy_count,
                total_sites=total_sites,
                is_unambiguous=is_unamb
            )
            self.peptide_records.append(record)
            extracted_count += 1

        self.file_summaries[idxml_path] = file_summary
        logger.info(
            f"Extracted {extracted_count} peptides from {idxml_path} "
            f"(Phospho: {file_summary.phospho_sites}, PhosphoDecoy: {file_summary.phosphodecoy_sites}, "
            f"Unambiguous: {file_summary.unambiguous_peptides})"
        )
        return extracted_count

    def calculate_cumulative_flr_ordered(self, exclude_unambiguous: bool = True) -> Tuple[List, List, List, List]:
        """
        Calculate cumulative FLR ordered by phosphorylation site count.

        Processing order:
        1. All 1-site Phospho peptides
        2. All 2-site Phospho peptides
        3. All 3-site Phospho peptides
        4. Mixed peptides (both Phospho and PhosphoDecoy)
        5. All 1-site PhosphoDecoy peptides
        6. All 2-site PhosphoDecoy peptides
        7. All 3-site PhosphoDecoy peptides

        Args:
            exclude_unambiguous: Whether to exclude unambiguous peptides

        Returns:
            Tuple of (site_counts, cumulative_flr, cumulative_phospho, cumulative_phosphodecoy)
        """
        logger.info("Calculating cumulative FLR ordered by phosphorylation site count...")

        # Filter records
        records = self.peptide_records
        if exclude_unambiguous:
            records = [r for r in records if not r.is_unambiguous]
            logger.info(f"Excluded {len(self.peptide_records) - len(records)} unambiguous peptides")

        # Group peptides by type and site count
        peptides_phospho = {1: [], 2: [], 3: []}  # Pure Phospho peptides
        peptides_phosphodecoy = {1: [], 2: [], 3: []}  # Pure PhosphoDecoy peptides
        peptides_mixed = []  # Peptides with both Phospho and PhosphoDecoy

        for record in records:
            if record.total_sites not in [1, 2, 3]:
                continue

            if record.phospho_count > 0 and record.phosphodecoy_count > 0:
                # Mixed peptide
                peptides_mixed.append(record)
            elif record.phospho_count > 0:
                # Pure Phospho
                peptides_phospho[record.total_sites].append(record)
            elif record.phosphodecoy_count > 0:
                # Pure PhosphoDecoy
                peptides_phosphodecoy[record.total_sites].append(record)

        # Calculate cumulative FLR
        site_counts = []
        cumulative_flr = []
        cumulative_phospho = []
        cumulative_phosphodecoy = []

        total_phospho = 0
        total_phosphodecoy = 0

        # Stage 1: Process all Phospho peptides (1 -> 2 -> 3 sites)
        logger.info("Stage 1: Processing Phospho peptides")
        for num_sites in [1, 2, 3]:
            logger.info(f"  Processing {num_sites}-site Phospho peptides: {len(peptides_phospho[num_sites])}")
            for record in peptides_phospho[num_sites]:
                total_phospho += record.phospho_count
                total_phosphodecoy += record.phosphodecoy_count

                total_sites = total_phospho + total_phosphodecoy
                flr = total_phosphodecoy / total_sites if total_sites > 0 else 0

                site_counts.append(total_sites)
                cumulative_flr.append(flr)
                cumulative_phospho.append(total_phospho)
                cumulative_phosphodecoy.append(total_phosphodecoy)

        # Stage 2: Process mixed peptides
        logger.info(f"Stage 2: Processing mixed peptides: {len(peptides_mixed)}")
        for record in peptides_mixed:
            total_phospho += record.phospho_count
            total_phosphodecoy += record.phosphodecoy_count

            total_sites = total_phospho + total_phosphodecoy
            flr = total_phosphodecoy / total_sites if total_sites > 0 else 0

            site_counts.append(total_sites)
            cumulative_flr.append(flr)
            cumulative_phospho.append(total_phospho)
            cumulative_phosphodecoy.append(total_phosphodecoy)

        # Stage 3: Process all PhosphoDecoy peptides (1 -> 2 -> 3 sites)
        logger.info("Stage 3: Processing PhosphoDecoy peptides")
        for num_sites in [1, 2, 3]:
            logger.info(f"  Processing {num_sites}-site PhosphoDecoy peptides: {len(peptides_phosphodecoy[num_sites])}")
            for record in peptides_phosphodecoy[num_sites]:
                total_phospho += record.phospho_count
                total_phosphodecoy += record.phosphodecoy_count

                total_sites = total_phospho + total_phosphodecoy
                flr = total_phosphodecoy / total_sites if total_sites > 0 else 0

                site_counts.append(total_sites)
                cumulative_flr.append(flr)
                cumulative_phospho.append(total_phospho)
                cumulative_phosphodecoy.append(total_phosphodecoy)

        if site_counts:
            logger.info(f"Final statistics:")
            logger.info(f"  Total Phospho sites: {total_phospho}")
            logger.info(f"  Total PhosphoDecoy sites: {total_phosphodecoy}")
            logger.info(f"  Total sites: {site_counts[-1]}")
            logger.info(f"  Final FLR: {cumulative_flr[-1]:.4f} ({cumulative_flr[-1]*100:.2f}%)")

        # Store for later use
        self.cumulative_data = [{
            'site_count': sc,
            'flr': flr,
            'phospho': p,
            'phosphodecoy': pd
        } for sc, flr, p, pd in zip(site_counts, cumulative_flr, cumulative_phospho, cumulative_phosphodecoy)]

        return site_counts, cumulative_flr, cumulative_phospho, cumulative_phosphodecoy

    def calculate_global_flr(self) -> None:
        """
        Calculate global FLR using ordered cumulative calculation.
        """
        self.calculate_cumulative_flr_ordered(exclude_unambiguous=True)
        logger.info("Global FLR calculation completed")

    @staticmethod
    def enforce_monotonic(y: np.ndarray, tool_name: str = '') -> np.ndarray:
        """
        Enforce monotonic increasing curve.

        Args:
            y: Y-axis data (FLR values)
            tool_name: Tool name for logging

        Returns:
            Monotonically increasing FLR values
        """
        y = np.array(y)
        y_monotonic = np.copy(y)

        for i in range(1, len(y_monotonic)):
            if y_monotonic[i] < y_monotonic[i - 1]:
                y_monotonic[i] = y_monotonic[i - 1]

        logger.debug(f"Applied monotonic increasing constraint to {tool_name}")
        return y_monotonic

    @staticmethod
    def smooth_early_surge(x: np.ndarray, y: np.ndarray, surge_window: int = 20000,
                          tool_name: str = '') -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle early surge phenomenon by smoothing the initial data points.

        Args:
            x: X-axis data (site count)
            y: Y-axis data (FLR values)
            surge_window: Number of data points to smooth
            tool_name: Tool name for logging

        Returns:
            Tuple of (smoothed_x, smoothed_y)
        """
        if len(x) < surge_window:
            return x, y

        x = np.array(x)
        y = np.array(y)

        logger.debug(f"Applying surge smoothing to first {surge_window} data points for {tool_name}")

        y_early = y[:surge_window]
        y_start = y_early[0]
        y_surge_end = y_early[-1]

        y_smooth = np.copy(y)

        # Apply progressive smoothing to first surge_window points
        for i in range(surge_window):
            progress = i / surge_window

            # Smooth S-curve (sigmoid variant)
            smooth_factor = 1 / (1 + np.exp(-10 * (progress - 0.5)))

            # Linear interpolation as baseline
            linear_value = y_start + (y_surge_end - y_start) * progress

            # Local average of original data
            local_window = min(500, surge_window // 20)
            start_idx = max(0, i - local_window)
            end_idx = min(surge_window, i + local_window + 1)
            local_avg = np.mean(y[start_idx:end_idx])

            # Mix: early relies on linear, later relies on local average
            y_smooth[i] = (1 - smooth_factor) * linear_value + smooth_factor * local_avg

        # Apply monotonic constraint within smoothing window
        for i in range(1, surge_window):
            if y_smooth[i] < y_smooth[i - 1]:
                y_smooth[i] = y_smooth[i - 1]

        # Smooth transition near surge_window
        transition_window = min(2000, surge_window // 5)
        for i in range(surge_window, min(surge_window + transition_window, len(y))):
            transition_progress = (i - surge_window) / transition_window
            y_smooth[i] = y_smooth[surge_window - 1] * (1 - transition_progress) + y[i] * transition_progress

        return x, y_smooth

    def get_counts_at_threshold(self, threshold: float) -> Dict[str, Dict[str, int]]:
        """
        Get counts of sites passing a global FLR threshold.

        Args:
            threshold: Global FLR threshold

        Returns:
            Dictionary with count statistics
        """
        if not self.cumulative_data:
            self.calculate_cumulative_flr_ordered()

        # Find last index where FLR <= threshold
        threshold_idx = None
        for idx, data in enumerate(self.cumulative_data):
            if data['flr'] <= threshold:
                threshold_idx = idx

        if threshold_idx is None:
            return {
                'threshold': threshold,
                'achieved': False,
                'total_sites': 0,
                'phospho_sites': 0,
                'phosphodecoy_sites': 0,
                'observed_flr': self.cumulative_data[-1]['flr'] if self.cumulative_data else 0
            }

        data = self.cumulative_data[threshold_idx]
        return {
            'threshold': threshold,
            'achieved': True,
            'total_sites': data['site_count'],
            'phospho_sites': data['phospho'],
            'phosphodecoy_sites': data['phosphodecoy'],
            'observed_flr': data['flr']
        }

    def plot_cumulative_flr_curve(self, output_path: str, max_flr: float = 0.1) -> None:
        """
        Generate a publication-quality cumulative FLR curve plot.

        Uses Nature journal styling with smoothing algorithms.

        Args:
            output_path: Path to save the plot (PNG or PDF)
            max_flr: Maximum FLR value for x-axis (default: 0.1 = 10%)

        Raises:
            ImportError: If matplotlib is not installed
        """
        try:
            import matplotlib.pyplot as plt
            from scipy.ndimage import gaussian_filter1d
        except ImportError:
            raise ImportError(
                "matplotlib and scipy are required for plotting. "
                "Install them with: pip install matplotlib scipy"
            )

        if not self.cumulative_data:
            self.calculate_cumulative_flr_ordered()

        if not self.cumulative_data:
            logger.warning("No FLR data available for plotting")
            return

        # Set matplotlib style - Nature journal style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 10,
            'axes.linewidth': 1.0,
            'axes.edgecolor': '#000000',
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'axes.titleweight': 'bold',
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,
            'xtick.major.size': 4,
            'ytick.major.size': 4,
            'legend.fontsize': 9,
            'legend.frameon': True,
            'legend.edgecolor': '#000000',
            'legend.fancybox': False,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        })

        # Extract data
        site_counts = np.array([d['site_count'] for d in self.cumulative_data])
        cumulative_flr = np.array([d['flr'] for d in self.cumulative_data])

        # Apply smoothing
        if len(site_counts) > 20000:
            site_counts_smooth, cumulative_flr_smooth = self.smooth_early_surge(
                site_counts, cumulative_flr, surge_window=20000, tool_name='Global'
            )
        else:
            site_counts_smooth = site_counts
            cumulative_flr_smooth = cumulative_flr

        # Enforce monotonic
        cumulative_flr_smooth = self.enforce_monotonic(cumulative_flr_smooth, tool_name='Global')

        # Apply Gaussian smoothing for extra smoothness
        if len(cumulative_flr_smooth) > 200:
            sigma = min(100, len(cumulative_flr_smooth) // 20)
            site_counts_smooth = gaussian_filter1d(site_counts_smooth.astype(float), sigma=sigma)
            cumulative_flr_smooth = gaussian_filter1d(cumulative_flr_smooth, sigma=sigma)

            # Ensure still monotonically increasing
            for j in range(1, len(site_counts_smooth)):
                if site_counts_smooth[j] < site_counts_smooth[j - 1]:
                    site_counts_smooth[j] = site_counts_smooth[j - 1]
            cumulative_flr_smooth = self.enforce_monotonic(cumulative_flr_smooth)

        # Extend curve to max_flr if needed
        final_flr = cumulative_flr_smooth[-1]
        final_site_count = site_counts_smooth[-1]

        if final_flr < max_flr:
            cumulative_flr_smooth = np.append(cumulative_flr_smooth, max_flr)
            site_counts_smooth = np.append(site_counts_smooth, final_site_count)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot curve (FLR on x-axis, site count on y-axis)
        ax.plot(cumulative_flr_smooth, site_counts_smooth,
                label='Cross-file Global FLR',
                color=NATURE_COLORS['primary'][0],
                linewidth=2.5,
                alpha=0.85)

        # Set labels
        ax.set_xlabel('FLR', fontweight='bold', fontsize=12)
        ax.set_ylabel('Cumulative Number of Phospho Sites', fontweight='bold', fontsize=12)
        ax.set_title('Cumulative FLR Curve (Cross-File)', fontweight='bold', fontsize=14, pad=20)

        # Set x-axis range
        ax.set_xlim(-0.002, max_flr)
        ax.set_ylim(bottom=0)

        # Add vertical line at FLR=0.01
        ax.axvline(x=0.01, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        # Add annotation at 1% FLR
        counts_at_1pct = self.get_counts_at_threshold(0.01)
        if counts_at_1pct['achieved']:
            ax.annotate(
                f"1% FLR: {counts_at_1pct['phospho_sites']} sites",
                xy=(0.01, counts_at_1pct['total_sites']),
                xytext=(0.02, counts_at_1pct['total_sites']),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7)
            )

        # Add legend
        ax.legend(frameon=True, loc='best', fontsize=10, edgecolor='black')

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')

        # Beautify
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

        # Also save PDF if PNG
        if output_path.endswith('.png'):
            pdf_path = output_path.replace('.png', '.pdf')
            plt.savefig(pdf_path, bbox_inches='tight')
            logger.info(f"Cumulative FLR curve saved to: {pdf_path}")

        plt.close()
        logger.info(f"Cumulative FLR curve saved to: {output_path}")

    def generate_summary_report(self, output_path: str, threshold: float = 0.01) -> None:
        """
        Generate a TSV summary report with per-file and total counts.

        Args:
            output_path: Path to save the report (TSV)
            threshold: Primary FLR threshold to report (default: 0.01)
        """
        if not self.cumulative_data:
            self.calculate_cumulative_flr_ordered()

        # Calculate counts at multiple thresholds
        thresholds = [0.001, 0.005, 0.01, 0.05, 0.1]
        threshold_stats = {t: self.get_counts_at_threshold(t) for t in thresholds}

        total_phospho = sum(s.phospho_sites for s in self.file_summaries.values())
        total_phosphodecoy = sum(s.phosphodecoy_sites for s in self.file_summaries.values())
        total_peptides = sum(s.total_peptides for s in self.file_summaries.values())
        total_unambiguous = sum(s.unambiguous_peptides for s in self.file_summaries.values())

        with open(output_path, 'w') as f:
            # Header
            f.write(f"# Global FLR Summary Report\n")
            f.write(f"# Files processed: {len(self.file_summaries)}\n")
            f.write(f"# Total peptides: {total_peptides}\n")
            f.write(f"# Unambiguous peptides excluded: {total_unambiguous}\n")
            f.write(f"# Total Phospho sites: {total_phospho}\n")
            f.write(f"# Total PhosphoDecoy sites: {total_phosphodecoy}\n")
            f.write(f"#\n")

            # Per-file summary
            f.write("# Per-file Summary\n")
            headers = ["file_path", "total_peptides", "phospho_sites", "phosphodecoy_sites", "unambiguous"]
            f.write("\t".join(headers) + "\n")

            for file_path, summary in self.file_summaries.items():
                row = [
                    os.path.basename(file_path),
                    str(summary.total_peptides),
                    str(summary.phospho_sites),
                    str(summary.phosphodecoy_sites),
                    str(summary.unambiguous_peptides)
                ]
                f.write("\t".join(row) + "\n")

            # Total row
            f.write(f"TOTAL\t{total_peptides}\t{total_phospho}\t{total_phosphodecoy}\t{total_unambiguous}\n")

            # Threshold analysis
            f.write("\n# Threshold Analysis\n")
            f.write("# FLR = PhosphoDecoy / (Phospho + PhosphoDecoy)\n")
            f.write("threshold\tachieved\tphospho_sites\tphosphodecoy_sites\ttotal_sites\tobserved_flr\n")

            for t in thresholds:
                stats = threshold_stats[t]
                f.write(f"{t:.3f}\t{stats['achieved']}\t{stats['phospho_sites']}\t"
                       f"{stats['phosphodecoy_sites']}\t{stats['total_sites']}\t{stats['observed_flr']:.6f}\n")

            # Final FLR
            if self.cumulative_data:
                final = self.cumulative_data[-1]
                f.write(f"\n# Final cumulative FLR: {final['flr']:.4f} ({final['flr']*100:.2f}%)\n")

        logger.info(f"Global FLR summary report saved to: {output_path}")

    def export_flr_data(self, output_path: str) -> None:
        """
        Export cumulative FLR data to CSV.

        Args:
            output_path: Path to save the data (CSV)
        """
        if not self.cumulative_data:
            self.calculate_cumulative_flr_ordered()

        with open(output_path, 'w') as f:
            f.write("cumulative_site_count,cumulative_flr,cumulative_phospho,cumulative_phosphodecoy\n")
            for data in self.cumulative_data:
                f.write(f"{data['site_count']},{data['flr']:.6f},{data['phospho']},{data['phosphodecoy']}\n")

        logger.info(f"FLR data exported to: {output_path}")

        # Also export Excel if openpyxl is available
        try:
            import pandas as pd
            df = pd.DataFrame(self.cumulative_data)
            df = df.rename(columns={
                'site_count': 'Cumulative_Site_Count',
                'flr': 'Cumulative_FLR',
                'phospho': 'Cumulative_Phospho',
                'phosphodecoy': 'Cumulative_PhosphoDecoy'
            })
            excel_path = output_path.replace('.csv', '.xlsx')
            df.to_excel(excel_path, index=False)
            logger.info(f"FLR data exported to: {excel_path}")
        except ImportError:
            pass  # openpyxl not available

    def update_idxml_files(self, idxml_paths: List[str]) -> None:
        """
        Update idXML files with cross-file global FLR metadata.

        Note: This calculates FLR based on the cumulative position of each peptide
        in the ordered processing.

        Args:
            idxml_paths: List of idXML file paths to update
        """
        from pyopenms import IdXMLFile, PeptideIdentificationList

        if not self.cumulative_data:
            self.calculate_cumulative_flr_ordered()

        # Build a lookup from sequence to FLR
        # Use the final FLR value for each unique sequence
        seq_to_flr = {}
        for data in self.cumulative_data:
            # We don't have sequence here, so we'll use a different approach
            pass

        logger.warning("update_idxml_files: Cross-file FLR injection requires sequence tracking. "
                      "Consider using the summary report instead.")

    def get_total_counts(self) -> Dict[str, int]:
        """
        Get total counts across all files.

        Returns:
            Dictionary with total statistics
        """
        return {
            'total_peptides': sum(s.total_peptides for s in self.file_summaries.values()),
            'phospho_sites': sum(s.phospho_sites for s in self.file_summaries.values()),
            'phosphodecoy_sites': sum(s.phosphodecoy_sites for s in self.file_summaries.values()),
            'unambiguous_peptides': sum(s.unambiguous_peptides for s in self.file_summaries.values())
        }
