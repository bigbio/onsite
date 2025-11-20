# LucXor (LuciPHOr2) Algorithm Documentation

## Overview

LucXor is a Python implementation of the LuciPHOr2 algorithm for accurate post-translational modification (PTM) localization with probabilistic scoring and false localization rate (FLR) estimation. It provides a comprehensive two-stage workflow for confident PTM site assignment with statistical validation.

## Algorithm Description

### Core Principles

1. **Two-Stage Processing**: Implements the complete LuciPHOr2 workflow with FLR estimation
2. **Probabilistic Scoring**: Uses statistical models for site localization confidence
3. **FLR Calculation**: Provides accurate false localization rate estimation
4. **Multi-threading**: Parallel processing for improved performance

### Mathematical Foundation

LucXor implements a sophisticated two-stage approach:

#### Stage 1: FLR Estimation (RN=0)
- Process all PSMs with both real and decoy permutations
- Calculate delta scores for all permutations
- Estimate false localization rates based on decoy distributions

#### Stage 2: FLR Assignment (RN=1)
- Re-process PSMs using only real permutations
- Assign FLR values based on the estimated distributions
- Generate final localization confidence scores

### Key Features

- **Two-stage processing**: Complete LuciPHOr2 workflow implementation
- **Multiple fragmentation methods**: Support for CID and HCD fragmentation
- **FLR calculation**: Accurate false localization rate estimation
- **Multi-threading**: Parallel processing for improved performance
- **PyOpenMS integration**: Seamless integration with OpenMS ecosystem

## Implementation Details

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fragment_method` | "CID" | Fragmentation method (CID or HCD) |
| `fragment_mass_tolerance` | 0.5 | Fragment mass tolerance |
| `fragment_error_units` | "Da" | Tolerance units (Da or ppm) |
| `min_mz` | 150.0 | Minimum m/z value to consider |
| `target_modifications` | ["Phospho (S)", "Phospho (T)", "Phospho (Y)"] | Target modifications |
| `neutral_losses` | ["sty -H3PO4 -97.97690"] | Neutral loss definitions |
| `decoy_mass` | 79.966331 | Mass for decoy generation |
| `max_charge_state` | 5 | Maximum charge state |
| `max_peptide_length` | 40 | Maximum peptide length |
| `max_num_perm` | 16384 | Maximum permutations |
| `modeling_score_threshold` | 0.95 | Minimum score for modeling |
| `min_num_psms_model` | 50 | Minimum PSMs for modeling |
| `threads` | 4 | Number of threads |
| `rt_tolerance` | 0.01 | Retention time tolerance |

### Workflow

#### Stage 1: FLR Estimation

1. **PSM Processing**: Process all PSMs with real and decoy permutations
2. **Delta Score Calculation**: Calculate delta scores for all permutations
3. **FLR Estimation**: Estimate false localization rates from decoy distributions
4. **Model Building**: Build statistical models for FLR calculation

#### Stage 2: FLR Assignment

1. **PSM Re-processing**: Re-process PSMs with only real permutations
2. **FLR Assignment**: Assign FLR values based on estimated distributions
3. **Score Calculation**: Calculate final localization scores
4. **Result Generation**: Generate final results with confidence metrics

### Advanced Features

#### FLR Calculator

```python
class FLRCalculator:
    """
    Calculates false localization rates using decoy distributions.
    """
    def __init__(self, min_delta_score=0.1, min_psms_per_charge=50):
        self.min_delta_score = min_delta_score
        self.min_psms_per_charge = min_psms_per_charge
    
    def calculate_flr_estimates(self, psms):
        """Calculate FLR estimates from decoy distributions."""
        # Implementation details...
    
    def assign_flr_to_psms(self, psms):
        """Assign FLR values to PSMs based on estimates."""
        # Implementation details...
```

#### Core Processor

```python
class CoreProcessor:
    """
    Main processor for scoring PSMs and calculating FLR.
    """
    def process_all_psms(self):
        """Process all PSMs using two-stage workflow."""
        # Stage 1: Calculate FLR estimates
        self._process_psms_stage1()
        
        # Stage 2: Assign FLR values
        self._process_psms_stage2()
```

#### PSM Handling

```python
class PSM:
    """
    Peptide Spectrum Match with localization scoring.
    """
    def process(self, config_dict):
        """Process PSM with real and decoy permutations."""
        # Implementation details...
    
    def process_stage2(self, config_dict):
        """Process PSM with only real permutations."""
        # Implementation details...
```

## Usage Examples

### Command Line Interface

```bash
# Basic usage
onsite lucxor -in spectra.mzML -id identifications.idXML -out results.idXML

# With custom parameters
onsite lucxor -in spectra.mzML -id identifications.idXML -out results.idXML \
    --fragment-method HCD \
    --fragment-mass-tolerance 0.5 \
    --fragment-error-units Da \
    --threads 8 \
    --debug
```

### Python API

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

### Advanced Usage

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

## Performance Characteristics

### Computational Complexity

- **Time Complexity**: O(n²) where n is the number of permutations
- **Space Complexity**: O(n) for storing permutations and spectra
- **Parallelization**: Supports multi-threading for improved performance

### Performance Metrics

- **Processing Speed**: ~100-1000 PSMs/second (depending on data complexity)
- **Memory Usage**: ~1-2 GB for typical datasets
- **Accuracy**: High accuracy with FLR-based confidence estimation

### Optimization Features

1. **Multi-threading**: Automatic thread optimization based on PSM count
2. **Memory Efficiency**: Streaming processing for large datasets
3. **Scalability**: Handles datasets with thousands of PSMs
4. **Caching**: Efficient caching of probability calculations

## Output Format

### LucXor Output

The tool generates an idXML file containing:

- **Luciphor_delta_score**: Main localization score
- **Luciphor_pep_score**: Peptide identification score
- **Luciphor_global_flr**: Global false localization rate
- **Luciphor_local_flr**: Local false localization rate

### CSV Output

```csv
SpectrumID,Peptide,ModifiedPeptide,Score,DeltaScore,GlobalFLR,LocalFLR,IsDecoy
1,PEPTIDE,PEPTIDE(Phospho),0.95,0.85,0.05,0.02,0
2,PEPTIDE,PEPTIDE(Phospho),0.90,0.80,0.10,0.05,0
```

## Limitations

1. **Memory Requirements**: Can be memory-intensive for large datasets
2. **Computational Cost**: Two-stage processing can be time-consuming
3. **FLR Dependencies**: Requires sufficient PSMs for accurate FLR estimation
4. **Fragment Quality**: Requires high-quality MS/MS spectra

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `max_peptide_length` or use fewer threads
2. **Slow Processing**: Increase number of threads or reduce fragment tolerance
3. **Poor Results**: Check fragment tolerance settings and data quality
4. **FLR Issues**: Ensure sufficient PSMs for modeling (minimum 50 recommended)

### Optimization Tips

1. **Thread Count**: Adjust based on available CPU cores
2. **Fragment Tolerance**: Use appropriate tolerance for your instrument
3. **Peptide Length**: Filter very long peptides if not needed
4. **Modeling Threshold**: Adjust based on data quality

## References

### Original Publications

#### LuciPHOr (2013)
Fermin, D., Walmsley, S. J., Gingras, A. C., Choi, H., & Nesvizhskii, A. I. (2013). LuciPHOr: algorithm for phosphorylation site localization with false localization rate estimation using modified target-decoy approach. *Molecular & Cellular Proteomics*, 12(11), 3409-3419.

**DOI**: 10.1074/mcp.M113.028928

#### LuciPHOr2 (2015)
Fermin, D., Avtonomov, D., Choi, H., & Nesvizhskii, A. I. (2015). LuciPHOr2: site localization of generic post-translational modifications from tandem mass spectrometry data. *Bioinformatics*, 31(7), 1141-1143.

**DOI**: 10.1093/bioinformatics/btu788

### Key Features of Original Implementation

1. **FLR Estimation**: Accurate false localization rate calculation
2. **Two-Stage Processing**: Complete workflow with validation
3. **Generic PTM Support**: Works with various post-translational modifications
4. **Statistical Validation**: Robust statistical framework

### Related Work

- **AScore**: Alternative probability-based approach
- **PhosphoRS**: Compomics-style scoring method
- **Mascot Delta Score**: Similar concept in database search engines

## Implementation Notes

### Differences from Original

1. **Python Implementation**: Native Python port of LuciPHOr2
2. **PyOpenMS Integration**: Seamless integration with OpenMS ecosystem
3. **Enhanced Multi-threading**: Improved parallel processing
4. **Memory Optimization**: Efficient memory usage for large datasets

### Future Improvements

1. **Machine Learning**: Integration of ML-based scoring
2. **Real-time Processing**: Streaming analysis capabilities
3. **Cross-linking**: Extension to cross-linked peptides
4. **Quantification**: Integration with quantification methods

## Algorithm Comparison

| Feature | AScore | PhosphoRS | LucXor |
|---------|--------|-----------|--------|
| **Statistical Framework** | Binomial | Binomial | FLR-based |
| **Site-Determining Ions** | Yes | Yes | Yes |
| **FLR Estimation** | No | No | Yes |
| **Two-Stage Processing** | No | No | Yes |
| **Decoy Support** | Optional | Optional | Required |
| **Multi-threading** | Yes | Limited | Yes |
| **Memory Usage** | Low | Medium | High |
| **Accuracy** | High | High | Very High |

## Best Practices

1. **Data Quality**: Ensure high-quality MS/MS spectra
2. **Fragment Tolerance**: Use appropriate tolerance for your instrument
3. **Thread Count**: Optimize based on available resources
4. **FLR Thresholds**: Set appropriate thresholds for your application
5. **Validation**: Use decoy sites for validation in research settings
