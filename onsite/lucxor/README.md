# lucxor

Python implementation of LuciPHOr2 for post-translational modification (PTM) site localization using PyOpenMS.

## Overview

lucxor is a Python port of the Java-based LuciPHOr2 tool for analyzing post-translational modifications in mass spectrometry data. It provides accurate localization of PTM sites (particularly phosphorylation) using a two-stage approach with false localization rate (FLR) calculation.

## Features

- **Two-stage processing**: Implements the complete LuciPHOr2 workflow with FLR estimation and assignment
- **Multiple fragmentation methods**: Support for CID and HCD fragmentation
- **FLR calculation**: Accurate false localization rate estimation for PTM site confidence
- **Multi-threading**: Parallel processing for improved performance
- **PyOpenMS integration**: Seamless integration with OpenMS ecosystem
- **Flexible input/output**: Support for mzML and idXML formats
- **Configurable parameters**: Extensive customization options for different experimental setups

## Installation

### Prerequisites

- Python 3.7+
- PyOpenMS
- NumPy
- SciPy

### Install from source

```bash
git clone <repository-url>
cd lucxor
pip install -e .
```

## Usage

### Command Line Interface

The main command-line tool is `PyLuciPHOr2`:

```bash
python -m lucxor.cli -in spectra.mzML -id identifications.idXML -out results.idXML
```

#### Required Arguments

- `-in, --input_spectrum`: Input spectrum file (mzML format)
- `-id, --input_id`: Input identification file (idXML format)
- `-out, --output`: Output file (idXML format)

#### Optional Arguments

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
- `--debug`: Enable debug mode
- `--log_file`: Custom log file path

#### Example

```bash
python -m lucxor.cli \
    -in data/spectra.mzML \
    -id data/identifications.idXML \
    -out results/results.idXML \
    --fragment_method HCD \
    --fragment_mass_tolerance 0.5 \
    --threads 8 \
    --debug
```

### Python API

```python
from lucxor import PyLuciPHOr2, LucXor, Peptide, PSM
from lucxor.models import CIDModel, HCDModel

# Using the main CLI class
tool = PyLuciPHOr2()
result = tool.run()

# Using the core processor
from lucxor.core import CoreProcessor
from lucxor.config import LucXorConfig

config = LucXorConfig()
config.fragment_method = "HCD"
config.fragment_mass_tolerance = 0.5

processor = CoreProcessor(config)
# Add PSMs and process
processor.process_all_psms()
```

## Algorithm

lucxor implements the complete LuciPHOr2 algorithm:

### Stage 1: FLR Estimation (RN=0)
- Process all PSMs with both real and decoy permutations
- Calculate delta scores for all permutations
- Estimate false localization rates based on decoy distributions

### Stage 2: FLR Assignment (RN=1)
- Re-process PSMs using only real permutations
- Assign FLR values based on the estimated distributions
- Generate final localization confidence scores

## Output

The tool generates an idXML file containing:
- **Luciphor_delta_score**: Main localization score
- **Luciphor_pep_score**: Peptide identification score
- **Luciphor_global_flr**: Global false localization rate
- **Luciphor_local_flr**: Local false localization rate

## Configuration

The tool uses a comprehensive configuration system with sensible defaults for most parameters. Key configuration options include:

- **Fragmentation method**: CID or HCD specific parameters
- **Mass tolerances**: Fragment and precursor mass tolerances
- **Modification definitions**: Target PTMs and neutral losses
- **Modeling parameters**: Score thresholds and minimum PSM counts
- **Performance settings**: Threading and memory management

## Logging

The tool provides comprehensive logging with configurable levels:
- **INFO**: General progress information
- **DEBUG**: Detailed processing information
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors

Log files are automatically generated with the pattern `{output_base}_debug.log`.

## Performance

- **Multi-threading**: Automatic thread optimization based on PSM count
- **Memory efficient**: Streaming processing for large datasets
- **Scalable**: Handles datasets with thousands of PSMs

## Dependencies

- **PyOpenMS**: Mass spectrometry data handling
- **NumPy**: Numerical computations
- **SciPy**: Statistical functions
- **Standard library**: argparse, logging, json, etc.

## License

Apache License 2.0

## Citation

If you use lucxor in your research, please cite the original LuciPHOr2 paper:

```
Fermin, D., et al. (2011). LuciPHOr: algorithm for phosphorylation site localization with false localization rate estimation using target-decoy approach. 
Molecular & Cellular Proteomics, 10(3), M110.003830.
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Support

For questions and support, please open an issue on the project repository.