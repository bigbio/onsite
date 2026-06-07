# PhosphoRS Algorithm Documentation

## Overview

PhosphoRS (Phosphorylation site Ranking and Scoring) is a comprehensive algorithm for phosphorylation site localization that implements a Compomics-inspired scoring method. It provides detailed site-specific probability calculations and isomer analysis for confident phosphorylation site assignment.

## Algorithm Description

### Core Principles

1. **Isomer Generation**: Generates all possible phosphorylation site combinations
2. **Theoretical Spectrum Matching**: Compares experimental spectra with theoretical predictions
3. **Probability Scoring**: Calculates site-specific probabilities using statistical models
4. **Confidence Assessment**: Provides detailed confidence metrics for each potential site

### Mathematical Foundation

PhosphoRS uses a sophisticated statistical approach:

1. **Site-Determining Ions (SDI)**: Identifies ions that distinguish between different site assignments
2. **Binomial Probability**: Uses cumulative binomial probability for scoring:
   ```
   P(X ≥ k) = Σ(i=k to n) C(n,i) * p^i * (1-p)^(n-i)
   ```
   Where:
   - `n` = number of theoretical ions
   - `k` = number of matched ions
   - `p` = probability of random match

3. **Delta Selection**: Optimizes spectrum reduction using site-determining ion differences
4. **Normalization**: Converts raw scores to site-specific probabilities

### Key Features

- **Fragment Tolerance**: 0.05 Da (default)
- **Site-Specific Probabilities**: Individual confidence scores for each site
- **Isomer Analysis**: Detailed analysis of all possible site combinations
- **Neutral Loss Support**: Handles phospho-specific neutral losses
- **Decoy Validation**: Optional decoy site analysis

## Implementation Details

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fragment_tolerance` | 0.05 | Fragment mass tolerance in Da |
| `fragment_method_ppm` | False | Use ppm tolerance instead of Da |
| `add_precursor_peak` | False | Include precursor peaks in analysis |
| `add_ion_types` | ("b", "y") | Ion types to consider |
| `max_ion_charge` | 2 | Maximum fragment charge state |
| `add_neutral_losses` | True | Include neutral losses |
| `add_decoys` | False | Include decoy sites for validation |
| `window_size` | 100.0 | Window size for spectrum reduction |
| `max_depth` | 8 | Maximum depth for intensity thresholds |
| `min_depth` | 2 | Minimum depth for intensity thresholds |

### Workflow

1. **Site Identification**: Identify potential phosphorylation sites (S, T, Y)
2. **Isomer Generation**: Create all possible site combinations
3. **Spectrum Filtering**: Reduce spectrum using window-based peak selection
4. **SDI Analysis**: Identify site-determining ions across isomers
5. **Delta Selection**: Optimize spectrum reduction using SDI differences
6. **Theoretical Matching**: Generate and match theoretical spectra
7. **Probability Calculation**: Calculate site-specific probabilities
8. **Result Assignment**: Assign final localization scores

### Advanced Features

#### Site-Determining Ion Analysis

The algorithm identifies ions that are unique to specific site assignments:

```python
def _site_determining_ions(profiles, precursor_charge, add_neutral_losses):
    """
    Identify site-determining ions that distinguish between different
    phosphorylation site assignments.
    """
    # Implementation details...
```

#### Delta Selection

Optimizes spectrum reduction by selecting the best depth based on site-determining ion differences:

```python
def _reduce_by_delta_selection(filtered_spec, profiles, fragment_tolerance, fragment_method_ppm):
    """
    Reduce spectrum using delta-based depth selection with site-determining ions.
    """
    # Implementation details...
```

#### Binomial Probability Calculation

Uses sophisticated statistical methods for probability calculation:

```python
def binomial_tail_probability(k: int, n: int, p: float) -> float:
    """
    Calculate cumulative binomial probability P(X >= k) for X~Bin(n,p).
    """
    # Implementation details...
```

## Usage Examples

### Command Line Interface

```bash
# Basic usage
onsite phosphors -in spectra.mzML -id identifications.idparquet -out results.idparquet

# With custom parameters
onsite phosphors -in spectra.mzML -id identifications.idparquet -out results.idparquet \
    --fragment-mass-tolerance 0.05 \
    --fragment-mass-unit Da \
    --threads 1 \
    --add-decoys
```

### Python API

```python
from onsite import calculate_phospho_localization_compomics_style

# Calculate phosphorylation site probabilities
site_probs, isomer_details = calculate_phospho_localization_compomics_style(
    peptide_hit, 
    spectrum,
    fragment_tolerance=0.05,
    fragment_method_ppm=False,
    add_neutral_losses=True
)
```

### Advanced Usage

```python
# Custom configuration
site_probs, isomer_details = calculate_phospho_localization_compomics_style(
    peptide_hit,
    spectrum,
    modification_name="Phospho",
    potential_sites={"S", "T", "Y"},
    fragment_tolerance=0.05,
    fragment_method_ppm=False,
    add_precursor_peak=False,
    add_ion_types=("b", "y"),
    max_ion_charge=2,
    add_neutral_losses=True,
    add_decoys=False
)

# Process results
if site_probs is not None:
    print("Site Probabilities:")
    for site_index, probability in site_probs.items():
        print(f"  Site {site_index}: {probability:.4f}")
    
    print("\nIsomer Details:")
    for seq_str, score in isomer_details:
        print(f"  {seq_str}: {score:.4f}")
```

## Performance Characteristics

### Computational Complexity

- **Time Complexity**: O(n²) where n is the number of potential sites
- **Space Complexity**: O(n) for storing isomers and spectra
- **Optimization**: Uses caching for probability calculations

### Performance Metrics

- **Processing Speed**: ~50-200 PSMs/second (depending on complexity)
- **Memory Usage**: ~2-4 GB for typical datasets
- **Accuracy**: High accuracy for peptides with clear site-determining ions

### Optimization Features

1. **Distribution Caching**: Caches binomial probability calculations
2. **Window Reduction**: Efficient spectrum filtering
3. **SDI Optimization**: Focuses on site-determining ions
4. **Memory Management**: Efficient memory usage for large datasets

## Output Format

### Site Probabilities

Returns a dictionary mapping site indices to probability scores:

```python
{
    0: 85.2,  # Site 0 has 85.2% probability
    3: 14.8,  # Site 3 has 14.8% probability
    7: 0.0    # Site 7 has 0% probability
}
```

### Isomer Details

Returns a list of tuples with sequence and score:

```python
[
    ("PEPTIDE(Phospho)SEQUENCE", 0.95),
    ("PEPTIDESEQUENCE(Phospho)", 0.05)
]
```

## Limitations

1. **Computational Cost**: Can be expensive for peptides with many potential sites
2. **Fragment Quality**: Requires high-quality MS/MS spectra
3. **Site Ambiguity**: May struggle with highly ambiguous localizations
4. **Memory Usage**: Can be memory-intensive for large datasets

## Troubleshooting

### Common Issues

1. **Low Probabilities**: Check fragment tolerance and spectrum quality
2. **Memory Errors**: Reduce window size or max depth
3. **Poor Localization**: Ensure sufficient site-determining ions
4. **Slow Processing**: Consider reducing max depth or window size

### Optimization Tips

1. **Fragment Tolerance**: Use appropriate tolerance for your instrument
2. **Window Size**: Adjust based on spectrum complexity
3. **Depth Settings**: Balance between accuracy and performance
4. **Neutral Losses**: Enable for phospho-specific analysis

## References

### Original Publication

Taus, T., Köcher, T., Pichler, P., Paschke, C., Schmidt, A., Henrich, C., & Mechtler, K. (2011). Universal and confident phosphorylation site localization using phosphoRS. *Journal of Proteome Research*, 10(12), 5354-5362.

**DOI**: 10.1021/pr200611n

**Abstract**: We present a new approach for confident phosphorylation site localization using phosphoRS, a method that combines the intensity information of site-determining fragment ions with a probability-based scoring scheme. The method is universally applicable to any type of mass spectrometric data and provides confident phosphorylation site localization with high accuracy.

### Key Features of Original Implementation

1. **Universal Applicability**: Works with any type of MS data
2. **Confident Localization**: High accuracy for site assignment
3. **Probability-Based Scoring**: Statistical framework for confidence assessment
4. **Site-Determining Ions**: Focus on ions that distinguish between sites

### Related Work

- **AScore**: Alternative probability-based approach
- **LuciPHOr**: FLR-based approach for site localization
- **Mascot Delta Score**: Similar concept in database search engines

## Implementation Notes

### Differences from Original

1. **Compomics Integration**: Enhanced with Compomics-style scoring
2. **PyOpenMS Integration**: Uses PyOpenMS for spectrum handling
3. **Advanced SDI Analysis**: Improved site-determining ion identification
4. **Delta Selection**: Optimized spectrum reduction

### Future Improvements

1. **Machine Learning**: Integration of ML-based scoring
2. **Real-time Processing**: Streaming analysis capabilities
3. **Cross-linking**: Extension to cross-linked peptides
4. **Quantification**: Integration with quantification methods
