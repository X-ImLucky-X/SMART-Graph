# Research Hypotheses - SMART-Graph

This document formalizes the scientific hypotheses we aim to test during this study:

## Null Hypothesis ($H_0$)
- **$H_0$**: Attention-Calibrated Boundary Stratification (ACBS) provides no statistically significant improvement in Semantic Divergence Rate (SDR) or fact retention over baseline sequential serialization strategies.

## Alternative Hypotheses ($H_1$, $H_2$, $H_3$)
- **$H_1$ (SOS Validity)**:
  Semantic Outlier Score (SOS) exhibits a positive correlation with factual omission frequency in flat graph serialization.
  - *Metric*: Pearson ($r$) and Spearman ($\rho$) correlation coefficients between triple SOS and empirical omission probability.
- **$H_2$ (ACBS Boundary Gain)**:
  U-shaped serialization preserves high-SOS (vulnerable) triples better than sequential or random orderings.
  - *Metric*: Omission rates for top-20% SOS triples positioned at boundary vs. middle.
- **$H_3$ (Scale-Insulated Convergence)**:
  The proposed SMART-Graph framework maintains lower Semantic Divergence Rates (SDR) and higher fact coverage compared to baseline methods as the input graph scale increases ($40+$ triples).
  - *Metric*: SDR slope and paired t-test differences across graph scales.
