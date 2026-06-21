# Research Contributions - SMART-Graph

This document lists the core research contributions of the SMART-Graph project:

## Contribution 1: Attention-Calibrated Boundary Stratification (ACBS)
A novel, non-parametric graph serialization framework that segment graphs using local semantic budgets and maps high-risk outlier facts directly to transformer attention primacy and recency zones.

## Contribution 2: Semantic Outlier Score (SOS) Profiling
A geometric method in continuous vector space to identify which triple facts in a global narrative are most vulnerable to context omission, incorporating literal value and syntax multipliers.

## Contribution 3: Soft-Match Graph Alignment (SMGA)
A bipartite graph matching algorithm (via Hungarian matching) that aligns input triples with LLM-extracted triples using dense sentence embedding similarities to decouple evaluation from extraction grammar/phrasing variances.

## Contribution 4: Attention Protection Score (APS)
A novel evaluation metric that specifically calculates the ratio of high-risk (outlier) triples preserved in the generated text relative to the total number of high-risk facts presented.
