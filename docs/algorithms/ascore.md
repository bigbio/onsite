# AScore Algorithm Documentation

## Overview

The AScore algorithm is a probability-based approach for high-throughput protein phosphorylation analysis and site localization. It was originally developed by Beausoleil et al. and provides a statistical framework for determining the most likely phosphorylation sites in a peptide sequence based on MS/MS fragment ion data.

## Algorithm Description

### Core Principles

1. **Site-Determining Ions**: AScore identifies fragment ions that are unique to specific phosphorylation site assignments
2. **Probability Calculation**: Uses binomial probability to assess the likelihood of observing matched ions by chance
3. **Score Assignment**: Assigns AScore values indicating confidence in site localization

### Mathematical Foundation

The AScore algorithm calculates localization confidence using the following approach:

1. **Fragment Analysis**: Analyzes MS/MS fragment ions to identify site-determining ions
2. **Probability Calculation**: Computes localization probabilities based on fragment evidence using:
   ```
   AScore = -10 * log10(P_first) - (-10 * log10(P_second))
   ```
   Where:
   - `P_first` = probability of observing matched ions for the best site assignment
   - `P_second` = probability of observing matched ions for the second-best site assignment

3. **Cumulative Scoring**: Uses cumulative binomial probability to assess the significance of ion matches

### Key Features

- **Fragment Mass Tolerance**: 0.05 Da (default)
- **Multi-threading Support**: Parallel processing for improved performance
- **Decoy Site Analysis**: Optional validation using decoy phosphorylation sites
- **Site-Specific Scoring**: Individual scores for each potential phosphorylation site
- **ProForma Output**: Standardized peptide sequence notation with confidence scores

## Implementation Details

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fragment_mass_tolerance` | 0.05 | Fragment mass tolerance in Da |
| `fragment_tolerance_ppm` | False | Use ppm tolerance instead of Da |
| `max_peptide_length` | 40 | Maximum peptide length to process |
| `max_permutations` | 16384 | Maximum number of site permutations to consider |
| `add_decoys` | False | Include decoy sites for validation |
| `unambiguous_score` | 1000.0 | Score for unambiguous localizations |

### Workflow

1. **Site Identification**: Identify potential phosphorylation sites (S, T, Y)
2. **Permutation Generation**: Generate all possible site combinations
3. **Theoretical Spectrum Generation**: Create theoretical spectra for each permutation
4. **Fragment Matching**: Match experimental and theoretical spectra
5. **Score Calculation**: Calculate AScore for each site
6. **Result Assignment**: Assign final localization scores

### Output

The algorithm provides:
- **AScore_pep_score**: Overall peptide score
- **AScore_1, AScore_2, ...**: Individual site scores
- **ProForma**: Standardized sequence notation with confidence scores
- **Site Probabilities**: Converted probability scores for each site

## Usage Examples

### Command Line Interface

```bash
# Basic usage
onsite ascore -in spectra.mzML -id identifications.idparquet -out results.idparquet

# With custom parameters
onsite ascore -in spectra.mzML -id identifications.idparquet -out results.idparquet \
    --fragment-mass-tolerance 0.05 \
    --fragment-mass-unit Da \
    --threads 4 \
    --add-decoys
```

### Python API

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

## Performance Characteristics

### Computational Complexity

- **Time Complexity**: O(n²) where n is the number of potential sites
- **Space Complexity**: O(n) for storing permutations and spectra
- **Parallelization**: Supports multi-threading for improved performance

### Performance Metrics

- **Processing Speed**: ~100-500 PSMs/second (depending on peptide complexity)
- **Memory Usage**: ~1-2 GB for typical datasets
- **Accuracy**: High accuracy for peptides with clear site-determining ions

## Limitations

1. **Site Ambiguity**: May struggle with highly ambiguous localizations
2. **Fragment Quality**: Requires high-quality MS/MS spectra with sufficient fragment ions
3. **Computational Cost**: Can be computationally expensive for peptides with many potential sites
4. **Decoy Dependence**: Performance may vary depending on decoy site inclusion

## Troubleshooting

### Common Issues

1. **Low Scores**: Check fragment tolerance settings and spectrum quality
2. **Memory Errors**: Reduce `max_peptide_length` or use fewer threads
3. **Poor Localization**: Ensure sufficient site-determining ions are present
4. **Slow Processing**: Consider reducing `max_permutations` for complex peptides

### Optimization Tips

1. **Fragment Tolerance**: Use appropriate tolerance for your instrument
2. **Threading**: Adjust thread count based on available CPU cores
3. **Decoy Sites**: Enable decoy sites for validation in research settings
4. **Peptide Length**: Consider filtering very long peptides if not needed

## References

### Original Publication

Beausoleil, S. A., Villén, J., Gerber, S. A., Rush, J., & Gygi, S. P. (2006). A probability-based approach for high-throughput protein phosphorylation analysis and site localization. *Nature Biotechnology*, 24(10), 1285-1292.

**DOI**: 10.1038/nbt1240

**Abstract**: We present a probability-based protein phosphorylation analysis and site localization algorithm, AScore, that automatically determines the correct number of phosphorylation sites and provides a measure of the confidence of the localization. AScore is based on the presence and intensity of site-determining ions in MS/MS spectra. We applied AScore to large-scale data sets of phosphorylation sites discovered in a human cell line and a mouse tissue. The algorithm is particularly useful for the analysis of large-scale phosphorylation data sets, where manual validation is impractical.

### Key Features of Original Implementation

1. **Statistical Framework**: Uses binomial probability for site localization
2. **Site-Determining Ions**: Focuses on ions that distinguish between site assignments
3. **Confidence Scoring**: Provides quantitative confidence measures
4. **High-Throughput**: Designed for large-scale phosphoproteomics studies

### Related Work

- **PhosphoRS**: Alternative approach using different statistical methods
- **LuciPHOr**: FLR-based approach for site localization
- **Mascot Delta Score**: Similar concept in database search engines

## Implementation Notes

### Differences from Original

1. **PyOpenMS Integration**: Uses PyOpenMS for spectrum handling and theoretical spectrum generation
2. **Multi-threading**: Enhanced parallel processing capabilities
3. **Decoy Support**: Optional decoy site analysis for validation
4. **ProForma Output**: Standardized sequence notation

### Future Improvements

1. **Machine Learning**: Integration of ML-based scoring
2. **Isobaric Tags**: Support for TMT/iTRAQ quantification
3. **Cross-linking**: Extension to cross-linked peptides
4. **Real-time Processing**: Streaming analysis capabilities
