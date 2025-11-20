# onsite 🔬🎯

[![Python application](https://github.com/bigbio/onsite/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/bigbio/onsite/actions/workflows/python-app.yml)
![PyPI - Version](https://img.shields.io/pypi/v/onsite?style=flat)
![PyPI - Downloads](https://img.shields.io/pypi/dm/onsite)
![Pepy Total Downloads](https://img.shields.io/pepy/dt/onsite)
![GitHub Repo stars](https://img.shields.io/github/stars/bigbio/onsite)

## 🚀 What is onsite?

**onsite** is a comprehensive Python package for mass spectrometry post-translational modification (PTM) localization. It provides algorithms for confident phosphorylation site localization and scoring, including implementations of AScore, PhosphoRS, and LucXor (LuciPHOr2).

### ✨ Key Features

- 🎯 **Multiple Algorithms**: AScore, PhosphoRS, and LucXor implementations
- 📊 **Statistical Validation**: Probability-based scoring with FLR estimation
- 💻 **Unified CLI**: Single command-line interface for all algorithms
- ⚡ **Multi-threading**: Parallel processing for improved performance
- 🔬 **PyOpenMS Integration**: Seamless integration with the OpenMS ecosystem
- 📈 **High Accuracy**: Confident site localization with statistical validation
- 🧩 **Flexible API**: Both command-line and Python API support

## 📊 Benchmark

We benchmarked onsite algorithms on the PXD000138 dataset using unified mzML/idXML inputs and consistent filtering (FDR < 0.01, tool-specific localization thresholds). Key takeaways:

| Tool | Total PSMs | Total phospho sites | Well-resolved sites | Uncertain sites | Global FLR |
|---|---|---|---|---|---|
| LuciPHOr | 111,588 | 118,625 | 48,186 | 37,337 | 0.0237 |
| AScore | 111,747 | 101,382 | 52,906 | 23,628 | 0.0283 |
| pyLucXor | 111,588 | 117,341 | 51,468 | 38,970 | 0.0304 |
| PhosphoRS | 111,747 | 107,552 | 50,000 | 26,354 | 0.0142 |

See [benchmark.md](benchmark.md) for methodology, full tables, and analysis details.

## 📋 Supported Algorithms

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

## 💾 Installation

### Prerequisites

- Python 3.11+
- PyOpenMS 3.4.0+
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
# Install from PyPI (when available)
pip install onsite

# Or install from source
git clone https://github.com/bigbio/onsite.git
cd onsite
pip install -e .
```

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

## 🛠️ Usage

### Command Line Interface

onsite provides a unified command-line interface for all algorithms:

#### Unified onsite CLI

```bash
# AScore algorithm
onsite ascore -in spectra.mzML -id identifications.idXML -out results.idXML

# PhosphoRS algorithm  
onsite phosphors -in spectra.mzML -id identifications.idXML -out results.idXML

# LucXor algorithm
onsite lucxor -in spectra.mzML -id identifications.idXML -out results.idXML
```

#### Individual Pipeline Tools

##### AScore Pipeline

```bash
# Basic usage
python -m onsite.ascore.cli -in spectra.mzML -id identifications.idXML -out results.idXML

# With custom parameters
python -m onsite.ascore.cli -in spectra.mzML -id identifications.idXML -out results.idXML \
    --fragment-mass-tolerance 0.05 \
    --fragment-mass-unit Da \
    --threads 4 \
    --add-decoys
```

##### PhosphoRS Pipeline

```bash
# Basic usage
python -m onsite.phosphors.cli -in spectra.mzML -id identifications.idXML -out results.idXML

# With custom parameters
python -m onsite.phosphors.cli -in spectra.mzML -id identifications.idXML -out results.idXML \
    --fragment-mass-tolerance 0.05 \
    --fragment-mass-unit Da \
    --threads 1 \
    --add-decoys
```

##### LucXor Pipeline

```bash
# Basic usage
python -m onsite.lucxor.cli -in spectra.mzML -id identifications.idXML -out results.idXML

# With custom parameters
python -m onsite.lucxor.cli -in spectra.mzML -id identifications.idXML -out results.idXML \
    --fragment-method HCD \
    --fragment-mass-tolerance 0.5 \
    --fragment-error-units Da \
    --threads 8 \
    --debug
```

### Command-line Options

#### AScore Options

| Option | Default | Description |
|---|---|---|
| `-in` | - | Input mzML file with spectra |
| `-id` | - | Input idXML file with identifications |
| `-out` | - | Output idXML file with scores |
| `--fragment-mass-tolerance` | 0.05 | Fragment mass tolerance |
| `--fragment-mass-unit` | Da | Tolerance unit (Da or ppm) |
| `--threads` | 1 | Number of threads for parallel processing |
| `--add-decoys` | False | Include decoy sites for validation |
| `--debug` | False | Enable debug logging |

#### PhosphoRS Options

| Option | Default | Description |
|---|---|---|
| `-in` | - | Input mzML file with spectra |
| `-id` | - | Input idXML file with identifications |
| `-out` | - | Output idXML file with scores |
| `--fragment-mass-tolerance` | 0.05 | Fragment mass tolerance |
| `--fragment-mass-unit` | Da | Tolerance unit (Da or ppm) |
| `--threads` | 1 | Number of threads for parallel processing |
| `--add-decoys` | False | Include decoy sites for validation |
| `--debug` | False | Enable debug logging |

#### LucXor Options

| Option | Default | Description |
|---|---|---|
| `-in` | - | Input mzML file with spectra |
| `-id` | - | Input idXML file with identifications |
| `-out` | - | Output idXML file with scores |
| `--fragment-method` | CID | Fragmentation method (CID or HCD) |
| `--fragment-mass-tolerance` | 0.5 | Fragment mass tolerance |
| `--fragment-error-units` | Da | Tolerance units (Da or ppm) |
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
| `--rt-tolerance` | 0.01 | RT tolerance used when matching spectra by retention time |
| `--debug` | False | Enable debug logging |

## 📊 Algorithm Details

### AScore Algorithm

The AScore algorithm provides phosphorylation site localization by analyzing MS/MS fragment ions to identify site-determining ions and computing localization probabilities based on fragment evidence.

**Output Metrics:**

- `AScore_pep_score`: Overall peptide score
- `AScore_1, AScore_2, ...`: Individual site scores
- `ProForma`: Standardized sequence notation with confidence scores

### PhosphoRS Algorithm

The PhosphoRS algorithm implements a comprehensive approach using isomer generation, theoretical spectrum matching, and probability scoring for confident phosphorylation site assignment.

**Output Metrics:**
- Site-specific probability scores (0-100%)
- Isomer details with sequence and score
- Detailed confidence metrics

### LucXor (LuciPHOr2) Algorithm

LucXor implements the complete LuciPHOr2 algorithm with two-stage processing for accurate PTM localization with false localization rate (FLR) estimation.

**Output Metrics:**
- `Luciphor_delta_score`: Main localization score
- `Luciphor_pep_score`: Peptide identification score
- `Luciphor_global_flr`: Global false localization rate
- `Luciphor_local_flr`: Local false localization rate

## 🔍 Example Results

You can find example result files in the `data` directory. Here are the direct links to different algorithm result files:

| Algorithm | Description | Result File |
|---|---|---|
| AScore | AScore phosphorylation site localization results | [AScore Example](../data/1_ascore_result.idXML) |
| PhosphoRS | PhosphoRS phosphorylation site localization results | [PhosphoRS Example](../data/1_phosphors_result.idXML) |
| LucXor | LucXor (LuciPHOr2) PTM localization results with FLR | [LucXor Example](../data/1_lucxor_result.idXML) |

## 📖 Documentation

For more detailed information:

- [AScore Algorithm Documentation](algorithms/ascore.md)
- [PhosphoRS Algorithm Documentation](algorithms/phosphors.md)
- [LucXor Algorithm Documentation](algorithms/lucxor.md)
- [Citations and References](citations.md)

## 👥 Contributing

To contribute to onsite:

1. 🍴 Fork the repository
2. 📥 Clone your fork: `git clone https://github.com/YOUR-USERNAME/onsite`
3. 🌿 Create a feature branch: `git checkout -b new-feature`
4. ✏️ Make your changes
5. 🔧 Install in development mode: `pip install -e .`
6. 🧪 Test your changes: `poetry run pytest`
7. 💾 Commit your changes: `git commit -am 'Add new feature'`
8. 📤 Push to the branch: `git push origin new-feature`
9. 📩 Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 📝 Citation

If you use onsite in your research, please cite:

```text
onsite: Mass spectrometry post-translational modification localization tool. 
https://github.com/bigbio/onsite
```

## 🔗 Related Tools

- [PyOpenMS](https://pyopenms.readthedocs.io/) - Python bindings for OpenMS
- [OpenMS](https://www.openms.de/) - Open-source tools for mass spectrometry
- [nf-core/quantms](https://nf-co.re/quantms) - Quantitative mass spectrometry workflow

## ❓ Need Help?

If you have questions or need assistance:
- [Open an issue](https://github.com/bigbio/onsite/issues) on GitHub
- Check [existing issues](https://github.com/bigbio/onsite/issues?q=is%3Aissue) for solutions

## 🙏 Acknowledgments

onsite builds upon the excellent work of the original algorithm developers and the OpenMS community. We thank all contributors and users for their feedback and support.

---

