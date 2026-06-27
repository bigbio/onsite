# onsite

[![Python application](https://github.com/bigbio/onsite/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/bigbio/onsite/actions/workflows/python-app.yml)
![PyPI - Version](https://img.shields.io/pypi/v/pyonsite?style=flat)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pyonsite)
![Pepy Total Downloads](https://img.shields.io/pepy/dt/pyonsite)
![GitHub Repo stars](https://img.shields.io/github/stars/bigbio/onsite)

## What is onsite?

**onsite** is a comprehensive Python package for mass spectrometry post-translational modification (PTM) localization. It provides algorithms for confident phosphorylation site localization and scoring, including implementations of AScore, PhosphoRS, and LucXor (LuciPHOr2).

### Key Features

- **Multiple Algorithms**: AScore, PhosphoRS, and LucXor implementations
- **Statistical Validation**: Probability-based scoring with FLR estimation
- **Unified CLI**: Single command-line interface for all algorithms
- **Multi-threading**: Parallel processing for improved performance
- **PyOpenMS Integration**: Seamless integration with the OpenMS ecosystem
- **High Accuracy**: Confident site localization with statistical validation
- **Flexible API**: Both command-line and Python API support

## Supported Algorithms

onsite provides three complementary algorithms for PTM localization:

### 1. **AScore Algorithm**
- **Method**: Probability-based approach using binomial statistics
- **Features**: Site-determining ion analysis, fast processing
- **Output**: AScore values indicating localization confidence
- **Citation**: Beausoleil et al. (2006) *Nature Biotechnology*

### 2. **PhosphoRS Algorithm**
- **Method**: Compomics-style scoring with isomer analysis
- **Features**: Site-specific probabilities, detailed isomer analysis
- **Output**: Site probability scores and isomer details
- **Citation**: Taus et al. (2011) *Journal of Proteome Research*

### 3. **LucXor (LuciPHOr2) Algorithm**
- **Method**: Two-stage processing with FLR estimation
- **Features**: False localization rate calculation, decoy-based validation
- **Output**: Delta scores, peptide scores, global and local FLR
- **Citation**: Fermin et al. (2013, 2015) *MCP* and *Bioinformatics*

## Installation

### Prerequisites

- Python 3.11+
- PyOpenMS 3.5.0+
- NumPy 2.3.2+
- SciPy 1.16.1+

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/bigbio/onsite.git
cd onsite

# Install with Poetry
poetry install

# Activate the virtual environment
poetry shell
```

### Using pip

```bash
# Install from PyPI (note: PyPI package name is 'pyonsite')
pip install pyonsite

# Or install from source
git clone https://github.com/bigbio/onsite.git
cd onsite
pip install -e .
```

**Note:** The package is published on PyPI as `pyonsite` due to a naming conflict, but the module is still imported as `onsite`.

### Development Installation

```bash
# Clone the repository
git clone https://github.com/bigbio/onsite.git
cd onsite

# Install with development dependencies
poetry install --with dev

# Or with pip
pip install -e ".[dev]"
```

## Usage

### Command Line Interface

onsite provides a unified command-line interface for all algorithms:

#### Unified onsite CLI

```bash
# AScore algorithm
onsite ascore -in spectra.mzML -id identifications.idparquet -out results.idparquet

# PhosphoRS algorithm  
onsite phosphors -in spectra.mzML -id identifications.idparquet -out results.idparquet

# LucXor algorithm
onsite lucxor -in spectra.mzML -id identifications.idparquet -out results.idparquet
```

#### Individual Pipeline Tools

##### AScore Pipeline

```bash
# Basic usage
python -m onsite.ascore.cli -in spectra.mzML -id identifications.idparquet -out results.idparquet

# With custom parameters
python -m onsite.ascore.cli -in spectra.mzML -id identifications.idparquet -out results.idparquet \
    --fragment-mass-tolerance 0.05 \
    --fragment-mass-unit Da \
    --threads 4 \
    --add-decoys
```

##### PhosphoRS Pipeline

```bash
# Basic usage
python -m onsite.phosphors.cli -in spectra.mzML -id identifications.idparquet -out results.idparquet

# With custom parameters
python -m onsite.phosphors.cli -in spectra.mzML -id identifications.idparquet -out results.idparquet \
    --fragment-mass-tolerance 0.05 \
    --fragment-mass-unit Da \
    --threads 1 \
    --add-decoys
```

##### LucXor Pipeline

```bash
# Basic usage
python -m onsite.lucxor.cli --input-spectrum spectra.mzML --input-id identifications.idparquet --output results.idparquet

# With custom parameters
python -m onsite.lucxor.cli --input-spectrum spectra.mzML --input-id identifications.idparquet --output results.idparquet \
    --fragment-method HCD \
    --fragment-mass-tolerance 0.5 \
    --fragment-mass-unit Da \
    --threads 8 \
    --debug
```

### Command-line Options

#### AScore Options

| Option | Default | Description |
|---|---|---|
| `-in` | - | Input mzML file with spectra |
| `-id` | - | Input idparquet file with identifications |
| `-out` | - | Output idparquet file with scores |
| `--fragment-mass-tolerance` | 0.05 | Fragment mass tolerance |
| `--fragment-mass-unit` | Da | Tolerance unit (Da or ppm) |
| `--threads` | 1 | Number of threads for parallel processing |
| `--add-decoys` | False | Include decoy sites for validation |
| `--compute-all-scores` | False | Run all three algorithms and merge results |
| `--debug` | False | Enable debug logging |

#### PhosphoRS Options

| Option | Default | Description |
|---|---|---|
| `-in` | - | Input mzML file with spectra |
| `-id` | - | Input idparquet file with identifications |
| `-out` | - | Output idparquet file with scores |
| `--fragment-mass-tolerance` | 0.05 | Fragment mass tolerance |
| `--fragment-mass-unit` | Da | Tolerance unit (Da or ppm) |
| `--threads` | 1 | Number of threads for parallel processing |
| `--add-decoys` | False | Include decoy sites for validation |
| `--compute-all-scores` | False | Run all three algorithms and merge results |
| `--debug` | False | Enable debug logging |

#### LucXor Options

| Option | Default | Description |
|---|---|---|
| `-in` | - | Input mzML file with spectra |
| `-id` | - | Input idparquet file with identifications |
| `-out` | - | Output idparquet file with scores |
| `--fragment-method` | CID | Fragmentation method (CID or HCD) |
| `--fragment-mass-tolerance` | 0.5 | Fragment mass tolerance |
| `--fragment-mass-unit` | Da | Tolerance units (Da or ppm) |
| `--min-mz` | 150.0 | Minimum m/z value to consider |
| `--target-modifications` | Phospho (S/T/Y) | List of target PTM definitions |
| `--neutral-losses` | sty -H3PO4 -97.97690 | Neutral loss definitions applied during scoring |
| `--decoy-mass` | 79.966331 | Mass offset used when generating decoy permutations |
| `--decoy-neutral-losses` | X -H3PO4 -97.97690 | Neutral loss patterns for decoy permutations |
| `--max-charge-state` | 5 | Maximum charge state |
| `--max-peptide-length` | 40 | Maximum peptide length |
| `--max-num-perm` | 16384 | Maximum permutations |
| `--modeling-score-threshold` | 0.95 | Minimum score for selecting PSMs during model building |
| `--scoring-threshold` | 0.0 | Minimum LucXor score to report |
| `--min-num-psms-model` | 50 | Minimum number of high-scoring PSMs required for modeling |
| `--threads` | 1 | Number of threads for parallel processing |
| `--seed` | 42 | RNG seed for reproducible decoy permutations / model subsampling (deterministic for the default single-threaded run) |
| `--rt-tolerance` | 0.01 | RT tolerance used when matching spectra by retention time |
| `--disable-split-by-charge` | False | Disable splitting PSMs by charge state for model training |
| `--compute-all-scores` | False | Run all three algorithms and merge results |
| `--debug` | False | Enable debug logging |

## Algorithm Details

### AScore Algorithm

The AScore algorithm provides phosphorylation site localization by analyzing MS/MS fragment ions to identify site-determining ions and computing localization probabilities based on fragment evidence.

**Output Metrics:**

- Hit score: the best per-site AScore (**higher = more confident**).
- `AScore_pep_score`: overall peptide-level AScore.
- `AScore_site_scores`: `{position: AScore}` dict, one entry per candidate site (0-based positions).
- `AScore_1, AScore_2, ...`: per-rank individual site scores.
- `ProForma`: standardized sequence notation with confidence scores.
- *Typical threshold:* **AScore ≥ 13** (~99% site-level confidence; Beausoleil et al. 2006).

### PhosphoRS Algorithm

The PhosphoRS algorithm implements a comprehensive approach using isomer generation, theoretical spectrum matching, and probability scoring for confident phosphorylation site assignment.

**Output Metrics:**
- `PhosphoRS_site_probs`: `{position: probability}` on a **0–100% scale** (**higher = more confident**) — the classic phosphoRS site probability.
- `PhosphoRS_site_delta`: `{position: Δ}` — the `−10·log10 P` gap between the best and best-alternative isoform (rank1 − rank2). Used to rank a global FLR because, unlike the probability, it does not saturate at 100%.
- `PhosphoRS_pep_score`: peptide-level binomial probability *P* (**lower = more confident**).
- `regular_phospho_count` / `phospho_decoy_count`: number of phospho / decoy sites placed.
- *Typical threshold:* **site probability ≥ 75%** (or 90 / 99% for stricter sets).

### LucXor (LuciPHOr2) Algorithm

LucXor implements the complete LuciPHOr2 algorithm with two-stage processing for accurate PTM localization with false localization rate (FLR) estimation.

**Output Metrics:**
- `Luciphor_delta_score`: main localization score (the hit score type; **higher = more confident**).
- `Luciphor_pep_score`: per-PSM delta score.
- `Luciphor_global_flr` / `Luciphor_local_flr`: LucXor's **native false-localization-rate** estimates per PSM (**lower = more confident**) — the only tool that emits an FLR directly.
- `Luciphor_site_scores`: `{position: Δ}` per-site confidence derived from the permutation scores.
- *Typical threshold:* **local FLR ≤ 0.05** (or global FLR ≤ 0.01).

## Interpreting the output: PSM-FDR vs localization FLR

These tools assume your input idparquet is **already filtered at the PSM level** (e.g. 1% PSM-FDR). That FDR answers *"is the peptide identification correct?"* and is left untouched. Localization adds a **second, orthogonal** error axis: *"is the PTM on the right residue?"* — the **false localization rate (FLR)**. A confident identification can still carry an ambiguous site, so the two rates are independent and you typically control both.

Running any tool on a 1%-PSM-FDR idparquet re-localizes each hit to its best-scoring site and writes these scores:

| | primary score | per-site confidence | typical cutoff | native FLR? |
|---|---|---|---|---|
| **AScore** | hit score = best AScore (higher = better) | `AScore_site_scores` | AScore ≥ 13 | no |
| **PhosphoRS** | `PhosphoRS_site_probs`, 0–100% (higher = better) | `PhosphoRS_site_probs` / `PhosphoRS_site_delta` | prob ≥ 75% | no |
| **LucXor** | `Luciphor_delta_score` (higher = better) | `Luciphor_site_scores` | `Luciphor_local_flr` ≤ 0.05 | **yes** |

Positions in the per-site dicts are **0-based** indices into the unmodified peptide.

**Quickest single-tool answer:** run LucXor and keep `Luciphor_local_flr ≤ 0.05` — it is the only tool that reports an FLR out of the box.

### Unified decoy-amino-acid global FLR (compare tools, or get an FLR for AScore/PhosphoRS)

AScore and PhosphoRS report a per-site *confidence* but no global FLR. To put all three on one comparable FLR scale, run each tool with **`--add-decoys`** (adds Alanine as a `PhosphoDecoy`; A cannot be phosphorylated, so a localization onto A is a known false one), then:

```bash
python -m onsite.decoy_flr \
    --ascore a.idparquet --phosphors p.idparquet --lucxor l.idparquet \
    --q-value-threshold 0.01 --flr-threshold 0.05
```

This reads the `target_decoy` and `q-value` UserParams from your FDR-filtered idparquet, re-applies the q-value cutoff, intersects the PSM set across the supplied tools, and reports the sites recovered at your global FLR threshold (decoy-amino-acid method of Ramsbottom et al. 2022; **5%** is the recommended cutoff). `--add-decoys` is only needed for this FLR estimation — for plain localization you can omit it. Any subset of `--ascore` / `--phosphors` / `--lucxor` may be passed.

## Example Results

You can find example result files in the `data` directory. Here are the direct links to different algorithm result files:

| Algorithm | Description | Result File |
|---|---|---|
| AScore | AScore phosphorylation site localization results | [AScore Example](data/1_ascore_result.idparquet) |
| PhosphoRS | PhosphoRS phosphorylation site localization results | [PhosphoRS Example](data/1_phosphors_result.idparquet) |
| LucXor | LucXor (LuciPHOr2) PTM localization results with FLR | [LucXor Example](data/1_lucxor_result.idparquet) |

## Documentation

For more detailed information:

- [AScore Algorithm Documentation](docs/algorithms/ascore.md)
- [PhosphoRS Algorithm Documentation](docs/algorithms/phosphors.md)
- [LucXor Algorithm Documentation](docs/algorithms/lucxor.md)
- [Citations and References](docs/citations.md)

## Contributing

To contribute to onsite:

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/onsite`
3. Create a feature branch: `git checkout -b new-feature`
4. Make your changes
5. Install in development mode: `pip install -e .`
6. Test your changes: `poetry run pytest`
7. Commit your changes: `git commit -am 'Add new feature'`
8. Push to the branch: `git push origin new-feature`
9. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use onsite in your research, please cite:

```text
onsite: Mass spectrometry post-translational modification localization tool. 
https://github.com/bigbio/onsite
```

## Related Tools

- [PyOpenMS](https://pyopenms.readthedocs.io/) - Python bindings for OpenMS
- [OpenMS](https://www.openms.de/) - Open-source tools for mass spectrometry
- [nf-core/quantms](https://nf-co.re/quantms) - Quantitative mass spectrometry workflow

## Need Help?

If you have questions or need assistance:
- [Open an issue](https://github.com/bigbio/onsite/issues) on GitHub
- Check [existing issues](https://github.com/bigbio/onsite/issues?q=is%3Aissue) for solutions

## Acknowledgments

onsite builds upon the excellent work of the original algorithm developers and the OpenMS community. We thank all contributors and users for their feedback and support.

---






