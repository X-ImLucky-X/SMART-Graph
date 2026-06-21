# Research Risk Register - SMART-Graph

This document lists potential threats to research validity, along with planned mitigations and early kill criteria:

## Core Risks & Mitigations

### Risk 1: Semantic Outlier Score (SOS) does not correlate with observed omission probability
- *Description*: The foundation of ACBS relies on the assumption that semantic outlier triples are dropped more frequently than in-cluster triples.
- *Mitigation*: Run `experiments/sos_correlation.py` immediately to compute correlation metrics.
- *Early Kill Criterion 1*: If the correlation coefficients (Pearson or Spearman) between SOS and observed omission rates are less than $0.2$, the hypothesis is considered unsupported and we stop the project to rethink the outlier definition.

### Risk 2: Weak LLM extraction quality dominates the Semantic Divergence Rate (SDR)
- *Description*: The local extraction model might generate syntax-corrupted or incomplete triples, causing false omissions or false hallucinations.
- *Mitigation*: Introduce Soft-Match Graph Alignment (SMGA) to handle semantic shifts, and implement an automated audit logger (`analysis/extraction_audit.json`) for manual verification of high-mismatch outputs.

### Risk 3: U-shaped prompt serialization (ACBS) performs identically to random serialization
- *Description*: The primary contribution (V-shaped attention placement) shows no empirical improvement over shuffling.
- *Mitigation*: Implement multiple baseline comparisons (Baseline B vs. proposed ACBS) to measure effect sizes.
- *Early Kill Criterion 2*: If ACBS improves SDR by less than $5\%$ compared to a random serialization baseline across medium/large graph scales, no further optimization is pursued.
