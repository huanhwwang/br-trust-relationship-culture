# Cultural Neuroscience of Trust: Ingroup–Outgroup Boundaries Across Cultures

A coordinate-based neuroimaging meta-analysis examining how cultural background and social relationship type jointly shape the neural architecture of trust and social cognition.

## Overview

This project reports a coordinate-based meta-analysis of 21 published neuroimaging studies (N_EuA = 13, N_EA = 8) spanning Euro-American (EuA) and East Asian (EA) participant samples. Two complementary analytical levels were conducted:

1. A **2×2 factorial decomposition** of Social Relationship (Stranger vs. Close Other) × Culture
2. A **targeted analysis of economic game paradigms** (trust game, ultimatum game, public goods game)

The central hypothesis is that cultural variation in trust reflects two orthogonal boundary dimensions:
- **Self–other boundary** (gradient in EuA; fused in EA)
- **Ingroup–outgroup boundary** (permeable in EuA; sharp and categorical in EA)

## Key Findings

- **EA close-other cognition** recruits a **dmPFC–precuneus** self-referential circuit (no caudate, no TPJ), consistent with representational self-other fusion.
- **EuA close-other cognition** recruits **vmPFC–caudate–TPJ**, consistent with a separately modelled, reward-tracked partner.
- For strangers, EuA engages OFC/vmPFC (value computation); EA engages aInsula–caudate (norm vigilance).
- The mPFC activation centroid shifts **19.5 mm** across cultures for strangers but only **3.8 mm** for close others.
- Economic game paradigms replicate the stranger-trust contrast (15.1 mm EuA–EA mPFC shift).

## Repository Structure

```
.
├── coordinates.tsv                        # Peak coordinates from included studies
├── refs.bib                               # BibTeX references
│
├── analysis1_ale.py                       # ALE meta-analysis (NiMARE)
├── analysis2_roi_distance.py              # mPFC centroid & ROI distance analysis
├── analysis4_decoding.py                  # Cognitive decoding via Neurosynth
├── analysis5_macm_inference.py            # Meta-analytic connectivity (MACM)
├── analysis_2x2_relationship_culture.py  # 2×2 factorial ALE decomposition
│
├── ale_maps/                              # ALE stat maps (EuA, EA, contrast, conjunction)
├── ale_maps_2x2/                          # ALE maps per 2×2 cell
├── ale_maps_econ/                         # ALE maps for economic game paradigms
│
├── ale_peaks_*.csv                        # ALE peak cluster tables
├── 2x2_summary.json                       # mPFC centroid summary per 2×2 cell
├── 2x2_macm.csv                           # MACM co-activation frequencies per cell
├── analysis23_summary.json                # ROI distance & centroid summary
├── econ_analysis_summary.json             # Economic game analysis summary
│
├── fig_conceptual_framework.py            # Figure 1: conceptual framework (matplotlib)
├── fig_conceptual_framework.pdf/.png      # Rendered Figure 1
├── fig_2x2_interaction.pdf                # Figure 2: 2×2 interaction results
├── fig_analysis1_ale_glassbrain.pdf       # Figure 3: ALE glass-brain maps
├── fig_analysis4_differential_decoding.pdf # Figure 4: cognitive decoding
├── fig_analysis5_macm_networks.pdf        # Figure 5: MACM connectivity profiles
│
├── synthesis_report_v2.tex                # Full synthesis report (LaTeX)
└── synthesis_report_v2.pdf               # Compiled report (22 pp)
```

## Methods

- **Meta-analysis tool**: [NiMARE](https://nimare.readthedocs.io/) — Activation Likelihood Estimation (ALE)
- **Thresholding**: FWE-corrected *p* < 0.05, cluster-forming threshold log*P* > 1.3, 1000 permutations
- **Coordinate space**: MNI152
- **ROI analysis**: mPFC subregion centroids (vmPFC/dmPFC); Euclidean distances across conditions
- **Cognitive decoding**: Neurosynth term-frequency profiles for EuA vs. EA peak clusters
- **MACM**: Co-activation frequency of seed regions (vmPFC, dmPFC) across all included studies

## Theoretical Framework

Three converging frameworks motivate the two-boundary prediction:

| Framework | Self–Other Boundary | Ingroup–Outgroup Boundary |
|---|---|---|
| Self-construal theory (Markus & Kitayama, 1991) | Independent (EuA) vs. interdependent (EA) self | — |
| Ingroup boundary tightness (Yamagishi & Yamagishi, 1994) | — | Permeable (EuA) vs. categorical (EA) |
| Emancipation theory (Triandis, 1995) | — | Collectivism → sharp ingroup boundary |

## Dependencies

```bash
pip install nimare numpy pandas matplotlib scipy
```

## Citation

Wang, H. (2026). *Ingroup–outgroup boundaries and neural trust across cultures: A coordinate-based meta-analysis*. Unpublished manuscript.

## Author

**Huan Wang** · huan.hw.wang@gmail.com
