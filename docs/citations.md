# Citations and References

This document provides comprehensive citations for all algorithms implemented in the onsite package, including original manuscripts, related work, and implementation references.

## Algorithm Citations

### AScore Algorithm

#### Primary Citation
Beausoleil, S. A., Villén, J., Gerber, S. A., Rush, J., & Gygi, S. P. (2006). A probability-based approach for high-throughput protein phosphorylation analysis and site localization. *Nature Biotechnology*, 24(10), 1285-1292.

**DOI**: 10.1038/nbt1240

**BibTeX**:
```bibtex
@article{beausoleil2006probability,
  title={A probability-based approach for high-throughput protein phosphorylation analysis and site localization},
  author={Beausoleil, Sean A and Vill{\'e}n, Judit and Gerber, Scott A and Rush, John and Gygi, Steven P},
  journal={Nature Biotechnology},
  volume={24},
  number={10},
  pages={1285--1292},
  year={2006},
  publisher={Nature Publishing Group}
}
```

#### Related Work
- **PhosphoRS**: Taus, T., et al. (2011). Universal and confident phosphorylation site localization using phosphoRS. *Journal of Proteome Research*, 10(12), 5354-5362.
- **LuciPHOr**: Fermin, D., et al. (2013). LuciPHOr: algorithm for phosphorylation site localization with false localization rate estimation using modified target-decoy approach. *Molecular & Cellular Proteomics*, 12(11), 3409-3419.

### PhosphoRS Algorithm

#### Primary Citation
Taus, T., Köcher, T., Pichler, P., Paschke, C., Schmidt, A., Henrich, C., & Mechtler, K. (2011). Universal and confident phosphorylation site localization using phosphoRS. *Journal of Proteome Research*, 10(12), 5354-5362.

**DOI**: 10.1021/pr200611n

**BibTeX**:
```bibtex
@article{taus2011universal,
  title={Universal and confident phosphorylation site localization using phosphoRS},
  author={Taus, Thomas and K{\"o}cher, Thomas and Pichler, Peter and Paschke, Christian and Schmidt, Andreas and Henrich, Christian and Mechtler, Karl},
  journal={Journal of Proteome Research},
  volume={10},
  number={12},
  pages={5354--5362},
  year={2011},
  publisher={ACS Publications}
}
```

#### Related Work
- **compomics-utilities**: Barsnes, H., Vaudel, M., Colaert, N., Helsens, K., Sickmann, A., Berven, F. S., & Martens, L. (2011). compomics-utilities: an open-source Java library for computational proteomics. *BMC Bioinformatics*, 12, 70.
- **AScore**: Beausoleil, S. A., et al. (2006). A probability-based approach for high-throughput protein phosphorylation analysis and site localization. *Nature Biotechnology*, 24(10), 1285-1292.

### LucXor (LuciPHOr2) Algorithm

#### Primary Citations

**LuciPHOr (2013)**:
Fermin, D., Walmsley, S. J., Gingras, A. C., Choi, H., & Nesvizhskii, A. I. (2013). LuciPHOr: algorithm for phosphorylation site localization with false localization rate estimation using modified target-decoy approach. *Molecular & Cellular Proteomics*, 12(11), 3409-3419.

**DOI**: 10.1074/mcp.M113.028928

**BibTeX**:
```bibtex
@article{fermin2013luciphor,
  title={LuciPHOr: algorithm for phosphorylation site localization with false localization rate estimation using modified target-decoy approach},
  author={Fermin, Damian and Walmsley, Scott J and Gingras, Anne-Claude and Choi, Hyungwon and Nesvizhskii, Alexey I},
  journal={Molecular \& Cellular Proteomics},
  volume={12},
  number={11},
  pages={3409--3419},
  year={2013},
  doi={10.1074/mcp.M113.028928}
}
```

**LuciPHOr2 (2015)**:
Fermin, D., Avtonomov, D., Choi, H., & Nesvizhskii, A. I. (2015). LuciPHOr2: site localization of generic post-translational modifications from tandem mass spectrometry data. *Bioinformatics*, 31(7), 1141-1143.

**DOI**: 10.1093/bioinformatics/btu788

**BibTeX**:
```bibtex
@article{fermin2015luciphor2,
  title={LuciPHOr2: site localization of generic post-translational modifications from tandem mass spectrometry data},
  author={Fermin, Damian and Avtonomov, Dmitry and Choi, Hyungwon and Nesvizhskii, Alexey I},
  journal={Bioinformatics},
  volume={31},
  number={7},
  pages={1141--1143},
  year={2015},
  doi={10.1093/bioinformatics/btu788}
}
```

**pAla (Phospho-Alanine Decoy Method) (2022)**:
Ramsbottom, K. A., Prakash, A., Riverol, Y. P., Camacho, O. M., Martin, M. J., Vizcaíno, J. A., Deutsch, E. W., & Jones, A. R. (2022). Method for Independent Estimation of the False Localization Rate for Phosphoproteomics. *Journal of Proteome Research*, 21(7), 1603-1615.

**DOI**: 10.1021/acs.jproteome.1c00827

**PMID**: 35640880; **PMCID**: PMC9251759

**BibTeX**:
```bibtex
@article{ramsbottom2022pala,
  title={Method for Independent Estimation of the False Localization Rate for Phosphoproteomics},
  author={Ramsbottom, Kirsty A and Prakash, Ananth and Riverol, Yasset Perez and Camacho, Oscar M and Martin, Maria J and Vizca{\'i}no, Juan Antonio and Deutsch, Eric W and Jones, Andrew R},
  journal={Journal of Proteome Research},
  volume={21},
  number={7},
  pages={1603--1615},
  year={2022},
  doi={10.1021/acs.jproteome.1c00827},
  pmid={35640880},
  pmcid={PMC9251759}
}
```

#### Related Work
- **Target-Decoy Approach**: Elias, J. E., & Gygi, S. P. (2007). Target-decoy search strategy for increased confidence in large-scale protein identifications by mass spectrometry. *Nature Methods*, 4(3), 207-214.
- **FLR Estimation**: Käll, L., et al. (2008). Posterior error probabilities and false discovery rates: two sides of the same coin. *Journal of Proteome Research*, 7(1), 40-44.

## Software and Framework Citations

### PyOpenMS
Röst, H. L., Sachsenberg, T., Aiche, S., Bielow, C., Weisser, H., Aicheler, F., Andreotti, S., Ehrlich, H. C., Gutenbrunner, P., Kenar, E., Liang, X., Nahnsen, S., Nilse, L., Pfeuffer, J., Rosenberger, G., Rurik, M., Schmitt, U., Veit, J., Walzer, M., Wojnar, D., Wolski, W. E., Schilling, O., Choudhary, J. S., Malmström, L., Aebersold, R., Reinert, K., & Kohlbacher, O. (2016). OpenMS: a flexible open-source software platform for mass spectrometry data analysis. *Nature Methods*, 13(9), 741-748.

**DOI**: 10.1038/nmeth.3959

### OpenMS
Kohlbacher, O., Reinert, K., Gröpl, C., Lange, E., Pfeifer, N., Schulz-Trieglaff, O., & Sturm, M. (2007). TOPP--the OpenMS proteomics pipeline. *Bioinformatics*, 23(2), e191-e197.

**DOI**: 10.1093/bioinformatics/btl299

### compomics-utilities
Barsnes, H., Vaudel, M., Colaert, N., Helsens, K., Sickmann, A., Berven, F. S., & Martens, L. (2011). compomics-utilities: an open-source Java library for computational proteomics. *BMC Bioinformatics*, 12, 70.

**DOI**: 10.1186/1471-2105-12-70

### NumPy
Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., Del Río, J. F., Wiebe, M., Peterson, P., Gérard-Marchant, P., Sheppard, K., Reddy, T., Weckesser, W., Abbasi, H., Gohlke, C., & Oliphant, T. E. (2020). Array programming with NumPy. *Nature*, 585(7825), 357-362.

**DOI**: 10.1038/s41586-020-2649-2

### SciPy
Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., Carey, C. J., Polat, İ., Feng, Y., Moore, E. W., VanderPlas, J., Laxalde, D., Perktold, J., Cimrman, R., Henriksen, I., Quintero, E. A., Harris, C. R., Archibald, A. M., Ribeiro, A. H., Pedregosa, F., van Mulbregt, P., & SciPy 1.0 Contributors. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, 17(3), 261-272.

**DOI**: 10.1038/s41592-019-0686-2

## Methodological References

### Phosphorylation Site Localization
- **Statistical Methods**: Savitski, M. M., et al. (2011). Confident phosphorylation site localization using the Mascot Delta Score. *Molecular & Cellular Proteomics*, 10(2), M110.003830.
- **Site-Determining Ions**: Steen, H., et al. (2006). The ABC's (and XYZ's) of peptide sequencing. *Nature Reviews Molecular Cell Biology*, 7(9), 633-643.

### False Discovery Rate
- **FDR Control**: Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B (Methodological)*, 57(1), 289-300.
- **Target-Decoy**: Elias, J. E., & Gygi, S. P. (2007). Target-decoy search strategy for increased confidence in large-scale protein identifications by mass spectrometry. *Nature Methods*, 4(3), 207-214.

### Mass Spectrometry
- **Tandem MS**: Aebersold, R., & Mann, M. (2003). Mass spectrometry-based proteomics. *Nature*, 422(6928), 198-207.
- **Fragmentation**: Steen, H., & Mann, M. (2004). The ABC's (and XYZ's) of peptide sequencing. *Nature Reviews Molecular Cell Biology*, 5(9), 699-711.

## Implementation References

### Python Implementation
- **Python**: Van Rossum, G., & Drake, F. L. (2009). Python 3 Reference Manual. CreateSpace.
- **NumPy**: Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585(7825), 357-362.
- **SciPy**: Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, 17(3), 261-272.

### Multi-threading
- **Threading**: Python Software Foundation. (2021). Python threading documentation. https://docs.python.org/3/library/threading.html
- **Parallel Processing**: Amdahl, G. M. (1967). Validity of the single processor approach to achieving large scale computing capabilities. *AFIPS Conference Proceedings*, 30, 483-485.

## How to Cite onsite

If you use onsite in your research, please cite:

### onsite Package
```
onsite: Mass spectrometry post-translational 
modification localization tool. https://github.com/bigbio/onsite
```

### Algorithm-Specific Citations

#### For AScore
```
Beausoleil, S. A., et al. (2006). A probability-based approach for high-throughput 
protein phosphorylation analysis and site localization. Nature Biotechnology, 
24(10), 1285-1292.
```

#### For PhosphoRS
```
Taus, T., et al. (2011). Universal and confident phosphorylation site 
localization using phosphoRS. Journal of Proteome Research, 10(12), 5354-5362.
```

#### For LucXor (LuciPHOr2)
```
Fermin, D., Walmsley, S. J., Gingras, A. C., Choi, H., & Nesvizhskii, A. I. (2013). 
LuciPHOr: algorithm for phosphorylation site localization with false localization rate 
estimation using modified target-decoy approach. Molecular & Cellular Proteomics, 
12(11), 3409-3419.

Fermin, D., Avtonomov, D., Choi, H., & Nesvizhskii, A. I. (2015). LuciPHOr2: site 
localization of generic post-translational modifications from tandem mass spectrometry 
data. Bioinformatics, 31(7), 1141-1143.
```

## Additional Resources

### Online Documentation
- **onsite Documentation**: https://github.com/bigbio/onsite/docs
- **PyOpenMS Documentation**: https://pyopenms.readthedocs.io/
- **OpenMS Documentation**: https://openms.readthedocs.io/

### Tutorials and Examples
- **onsite Tutorials**: https://github.com/bigbio/onsite/docs/tutorials
- **Algorithm Comparisons**: https://github.com/bigbio/onsite/docs/benchmarks
- **API Reference**: https://github.com/bigbio/onsite/docs/api

### Community and Support
- **GitHub Issues**: https://github.com/bigbio/onsite/issues
- **Discussions**: https://github.com/bigbio/onsite/discussions
- **BigBio Community**: https://github.com/bigbio

## License and Acknowledgments

This software is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Acknowledgments
onsite builds upon the excellent work of the original algorithm developers and the OpenMS community. We thank all contributors and users for their feedback and support.

### Contributing
We welcome contributions to onsite. Please see our [Contributing Guidelines](CONTRIBUTING.md) for more information.
