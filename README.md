# OnSite

Mass spectrometry post-translational modification localization tool.

This package provides comprehensive tools for phosphorylation site localization and scoring using algorithms including AScore, PhosphoRS, and LucXor (LuciPHOr2).

## Features

- **AScore Algorithm**: Implementation of the AScore algorithm for phosphorylation site localization
- **PhosphoRS Algorithm**: Implementation of the PhosphoRS algorithm for phosphorylation site localization  
- **PhosphoScoring Pipeline**: Complete workflow tool for processing MS/MS data files with AScore
- **PhosphoRS Scoring Pipeline**: Dedicated pipeline for PhosphoRS-based localization
- **LucXor**: Python port of LuciPHOr2 for accurate PTM localization with probabilistic scoring and FLR estimation
- **Unified CLI**: Single command-line interface for all algorithms
- **Multi-threading Support**: Parallel processing for improved performance
- **PyOpenMS Integration**: Seamless integration with the OpenMS ecosystem

## Installation

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

## Usage

### Command Line Interface

OnSite provides multiple command-line tools for different algorithms:

#### Unified OnSite CLI

Use the main `onsite` command for all algorithms:

```bash
# AScore algorithm
onsite ascore -in spectra.mzML -id identifications.idXML -out results.idXML

# PhosphoRS algorithm  
onsite phosphors -in spectra.mzML -id identifications.idXML -out results.idXML

# LucXor algorithm
onsite lucxor -in spectra.mzML -id identifications.idXML -out results.idXML
```

#### Individual Pipeline Tools

##### PhosphoScoring Pipeline (AScore)

Process MS/MS data files using the AScore-based PhosphoScoring pipeline:

```bash
# Using Poetry
poetry run phospho-scoring -in spectra.mzML -id identifications.idXML -out results.idXML

# Or after installation
phospho-scoring -in spectra.mzML -id identifications.idXML -out results.idXML

# With custom parameters
phospho-scoring -in spectra.mzML -id identifications.idXML -out results.idXML \
    -fragment_mass_tolerance 0.05 \
    -fragment_mass_unit Da \
    -threads 4 \
    --add_decoys
```

##### PhosphoRS Scoring Pipeline

Use the dedicated PhosphoRS scoring pipeline:

```bash
# Using Poetry
poetry run phosphors-scoring -in spectra.mzML -id identifications.idXML -out results.idXML

# Or after installation
phosphors-scoring -in spectra.mzML -id identifications.idXML -out results.idXML

# With custom parameters
phosphors-scoring -in spectra.mzML -id identifications.idXML -out results.idXML \
    -fragment_mass_tolerance 0.05 \
    -fragment_mass_unit Da \
    -threads 1 \
    --add_decoys
```

##### LucXor Tool

Use the LucXor tool for advanced PTM localization with FLR estimation:

```bash
# Using Poetry
poetry run lucxor -in spectra.mzML -id identifications.idXML -out results.idXML

# Or after installation
lucxor -in spectra.mzML -id identifications.idXML -out results.idXML

# With custom parameters
lucxor -in spectra.mzML -id identifications.idXML -out results.idXML \
    --fragment_method HCD \
    --fragment_mass_tolerance 0.5 \
    --fragment_error_units Da \
    --threads 8 \
    --debug
```

### Python API

#### AScore Algorithm

```python
from pyopenms import *
from onsite import AScore

# Initialize AScore
ascore = AScore()

# Set parameters
ascore.setParameter("fragment_mass_tolerance", 0.05)
ascore.setParameter("fragment_mass_unit", "Da")

# Process a peptide hit
result = ascore.compute(peptide_hit, spectrum)
```

#### PhosphoRS Algorithm

```python
from onsite import calculate_phospho_localization_compomics_style

# Calculate phosphorylation site probabilities
site_probs, isomer_details = calculate_phospho_localization_compomics_style(
    peptide_hit, 
    spectrum,
    fragment_tolerance=0.05
)
```

#### LucXor API

```python
from onsite.lucxor.cli import PyLuciPHOr2
from onsite.lucxor.psm import PSM
from onsite.lucxor.peptide import Peptide
from onsite.lucxor.models import CIDModel, HCDModel
from onsite.lucxor.spectrum import Spectrum
from onsite.lucxor.flr import FLRCalculator

# Initialize the main processor
processor = PyLuciPHOr2()

# The processor will parse command line arguments automatically
# For programmatic use, you can also use the core components directly
```

#### Advanced LucXor Usage

```python
from onsite.lucxor.core import CoreProcessor
from onsite.lucxor.config import LucXorConfig
from onsite.lucxor.psm import PSM
from onsite.lucxor.spectrum import Spectrum

# Custom configuration
config = LucXorConfig()
config.fragment_method = "HCD"
config.fragment_mass_tolerance = 0.5
config.fragment_error_units = "Da"
config.threads = 4

# Initialize processor
processor = CoreProcessor(config)

# Process PSMs
psm_list = [PSM(...)]  # Your PSM objects
spectrum_map = {...}   # Your spectrum mapping
results = processor.process_all_psms(psm_list, spectrum_map)
```

#### Pipeline Integration

```python
from onsite.phospho_scoring import main as phospho_scoring_main
from onsite.phosphors.phosphors_scoring import main as phosphors_scoring_main

# Run PhosphoScoring pipeline programmatically
# Note: These functions expect command line arguments
# For programmatic use, consider using the individual algorithm classes directly
```

## LucXor Algorithm

LucXor implements the complete LuciPHOr2 algorithm with two-stage processing:

### Stage 1: FLR Estimation (RN=0)
- Process all PSMs with both real and decoy permutations
- Calculate delta scores for all permutations
- Estimate false localization rates based on decoy distributions

### Stage 2: FLR Assignment (RN=1)
- Re-process PSMs using only real permutations
- Assign FLR values based on the estimated distributions
- Generate final localization confidence scores

### Key Features
1. **Two-stage processing**: Implements the complete LuciPHOr2 workflow with FLR estimation
2. **Multiple fragmentation methods**: Support for CID and HCD fragmentation
3. **FLR calculation**: Accurate false localization rate estimation for PTM site confidence
4. **Multi-threading**: Parallel processing for improved performance
5. **PyOpenMS integration**: Seamless integration with OpenMS ecosystem

## LucXor Configuration

Key parameters:

- `--fragment_method`: Fragmentation method (CID or HCD, default: CID)
- `--fragment_mass_tolerance`: Fragment mass tolerance (default: 0.5)
- `--fragment_error_units`: Tolerance units (Da or ppm, default: Da)
- `--min_mz`: Minimum m/z value to consider (default: 150.0)
- `--target_modifications`: Target modifications (default: ["Phospho (S)", "Phospho (T)", "Phospho (Y)"])
- `--neutral_losses`: Neutral loss definitions (default: ["sty -H3PO4 -97.97690"])
- `--decoy_mass`: Mass for decoy generation (default: 79.966331)
- `--max_charge_state`: Maximum charge state (default: 5)
- `--max_peptide_length`: Maximum peptide length (default: 40)
- `--max_num_perm`: Maximum permutations (default: 16384)
- `--modeling_score_threshold`: Minimum score for modeling (default: 0.95)
- `--min_num_psms_model`: Minimum PSMs for modeling (default: 50)
- `--threads`: Number of threads (default: 4)
- `--rt_tolerance`: Retention time tolerance (default: 0.01)

## LucXor Output

The tool generates an idXML file containing:
- **Luciphor_delta_score**: Main localization score
- **Luciphor_pep_score**: Peptide identification score
- **Luciphor_global_flr**: Global false localization rate
- **Luciphor_local_flr**: Local false localization rate

## LucXor Performance

- **Multi-threading**: Automatic thread optimization based on PSM count
- **Memory efficient**: Streaming processing for large datasets
- **Scalable**: Handles datasets with thousands of PSMs
- **Processing Speed**: ~100-1000 PSMs/second depending on data complexity
- **Memory Usage**: ~1-2 GB for typical datasets

## LucXor Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `max_peptide_length` or use fewer threads
2. **Slow Processing**: Increase number of threads or reduce fragment tolerance
3. **Poor Results**: Check fragment tolerance settings and data quality
4. **FLR Issues**: Ensure sufficient PSMs for modeling (minimum 50 recommended)

## Algorithm Details

### AScore Algorithm

The AScore algorithm provides phosphorylation site localization by:

1. **Fragment Analysis**: Analyzes MS/MS fragment ions to identify site-determining ions
2. **Probability Calculation**: Computes localization probabilities based on fragment evidence
3. **Score Assignment**: Assigns AScore values indicating confidence in site localization
4. **Multi-threading Support**: Parallel processing for improved performance

**Key Features:**
- Fragment mass tolerance: 0.05 Da (default)
- Support for multiple phosphorylation sites per peptide
- Decoy site analysis for validation
- Integration with PyOpenMS ecosystem

### PhosphoRS Algorithm

The PhosphoRS algorithm implements a comprehensive approach to phosphorylation site localization:

1. **Isomer Generation**: Generates all possible phosphorylation site combinations
2. **Theoretical Spectrum Matching**: Compares experimental spectra with theoretical predictions
3. **Probability Scoring**: Calculates site-specific probabilities using statistical models
4. **Confidence Assessment**: Provides detailed confidence metrics for each potential site

**Key Features:**
- Fragment tolerance: 0.05 Da (default)
- Support for S, T, Y phosphorylation sites
- Detailed isomer analysis and reporting
- Compatible with various mass spectrometry platforms

### Pipeline Tools

#### PhosphoScoring Pipeline (AScore-based)

The `phospho-scoring` tool provides a complete workflow for AScore-based phosphorylation site localization:

**Features:**
- Multi-threaded processing (default: 4 threads)
- Automatic spectrum and identification loading
- Progress tracking and logging
- Debug mode for detailed analysis
- Decoy site analysis option

**Output:**
- Enhanced idXML file with AScore annotations
- Site-specific localization scores
- Confidence metrics and statistics

#### PhosphoRS Scoring Pipeline

The `phosphors-scoring` tool provides a dedicated pipeline for PhosphoRS-based localization:

**Features:**
- Sequential processing (default: 1 thread for stability)
- Enhanced validation and error handling
- Detailed progress feedback
- Comprehensive logging capabilities
- Decoy site analysis for validation

**Output:**
- Enhanced idXML file with PhosphoRS annotations
- Site probability scores
- Isomer-specific details and confidence metrics

## Dependencies

- Python 3.11+
- pyOpenMS 3.4.0+
- NumPy 2.3.2+
- SciPy 1.16.1+

## Development

### Setup Development Environment

```bash
# Install development dependencies
poetry install --with dev

# Run tests
poetry run pytest

# Format code
poetry run black .

# Lint code
poetry run flake8
```

### Project Structure

```
onsite/
├── onsite/                    # Main package directory
│   ├── __init__.py           # Package initialization and exports
│   ├── ascore/               # AScore algorithm package
│   │   ├── __init__.py       # AScore package initialization
│   │   └── ascore.py         # AScore algorithm implementation
│   ├── phosphors/             # PhosphoRS algorithm package
│   │   ├── __init__.py       # PhosphoRS package initialization
│   │   ├── phosphors.py      # PhosphoRS algorithm implementation
│   │   └── phosphors_scoring.py  # PhosphoRS-based scoring pipeline
│   ├── phospho_scoring.py    # AScore-based scoring pipeline
│   ├── onsitec.py            # Unified command-line interface
│   └── lucxor/               # LucXor (LuciPHOr2) implementation
│       ├── __init__.py       # LucXor package initialization
│       ├── cli.py            # LucXor command-line interface
│       ├── core.py           # Core processing logic
│       ├── config.py         # Configuration management
│       ├── constants.py      # Constants and definitions
│       ├── flr.py            # False localization rate calculation
│       ├── globals.py        # Global variables and settings
│       ├── models.py         # Fragmentation models (CID/HCD)
│       ├── parallel.py       # Parallel processing utilities
│       ├── peak.py           # Peak processing and filtering
│       ├── peptide.py        # Peptide handling and modifications
│       ├── psm.py            # PSM (Peptide Spectrum Match) handling
│       ├── spectrum.py       # Spectrum processing and analysis
│       └── README.md         # LucXor-specific documentation
├── tests/                    # Test suite
│   ├── test_ascore.py       # AScore algorithm tests
│   ├── test_cli.py          # CLI interface tests
│   ├── test_imports.py      # Import validation tests
│   ├── test_lucxor.py       # LucXor functionality tests
│   └── test_phosphors.py    # PhosphoRS algorithm tests
├── pyproject.toml           # Project configuration and dependencies
├── requirements.txt         # Python dependencies
├── README.md                # This documentation file
└── LICENSE                  # MIT License
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this software in your research, please cite:

```
BigBio Stack. (2025). OnSite: Mass spectrometry post-translational 
modification localization tool. https://github.com/bigbio/onsite
```

### Algorithm-Specific Citations

#### AScore Algorithm
If you use the AScore implementation, please cite:

```
Beausoleil, S. A., et al. (2006). A probability-based approach for high-throughput 
protein phosphorylation analysis and site localization. Nature Biotechnology, 
24(10), 1285-1292.
```

#### PhosphoRS Algorithm
If you use the PhosphoRS implementation, please cite:

```
Taus, T., et al. (2011). Universal and confident phosphorylation site 
localization using phosphoRS. Journal of Proteome Research, 10(12), 5354-5362.
```

#### LucXor (LuciPHOr2) Algorithm
If you use the LucXor implementation, please cite:

```
Fermin, D., et al. (2011). LuciPHOr: algorithm for phosphorylation site 
localization with false localization rate estimation using target-decoy approach. 
Molecular & Cellular Proteomics, 10(3), M110.003830.
```

And the updated version:

```
Fermin, D., et al. (2013). LuciPHOr2: site identification of generic 
post-translational modifications from tandem mass spectrometry data. 
Nature Methods, 10(8), 744-746.
```

## Summary

OnSite provides a comprehensive suite of tools for mass spectrometry-based post-translational modification localization:

- **Multiple Algorithms**: AScore, PhosphoRS, and LucXor (LuciPHOr2) implementations
- **Flexible Usage**: Command-line tools and Python API for different use cases
- **Production Ready**: Multi-threading support, comprehensive error handling, and detailed logging
- **OpenMS Integration**: Seamless integration with the PyOpenMS ecosystem
- **Well Tested**: Comprehensive test suite and continuous integration
- **Active Development**: Regular updates and community contributions

Whether you need quick phosphorylation site localization with AScore, detailed probability analysis with PhosphoRS, or advanced FLR-based confidence estimation with LucXor, OnSite provides the tools you need for reliable PTM analysis.

## Support

For questions, bug reports, or feature requests, please:

1. Check the [Issues](https://github.com/bigbio/onsite/issues) page
2. Create a new issue with detailed information
3. Join our community discussions

## Acknowledgments

OnSite builds upon the excellent work of the original algorithm developers and the OpenMS community. We thank all contributors and users for their feedback and support.

