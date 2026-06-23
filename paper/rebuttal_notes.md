# SMART-Graph Academic Submission: Proactive Rebuttal Notes

This document drafts strategic responses to common reviewer concerns that may arise during peer review.

---

## 1. Concern: Use of Programmatically Chained Synthetic Large Graphs
> **Reviewer Comment:** *"The scaling study relies on programmatically chained synthetic graphs. It is unclear if the results generalize to real-world large-scale graphs or if this represents an artifact of the generation process."*

### Proposed Response Strategy
- **Acknowledge and Frame**: We acknowledge that synthetic chaining is a simplification of large-scale graph structures. We chose this approach to systemically test the boundary limits of context length ($40, 50, 60, 70$ triples) while maintaining a controlled, connected topology.
- **Duality of Evaluation**: We evaluate SMART-Graph on a dual dataset layout:
  1. **Native WebNLG Data**: Human-annotated, diverse small-scale graphs (under 10 triples) are used to baseline baseline capabilities.
  2. **Synthetic connected graphs**: Specifically generated to evaluate scalability up to 70 triples, which exceeds the scale of typical benchmark datasets.
- **Generality of Graph Topology**: The ACBS traversal algorithm depends only on sharing subject/object entities (co-reference links) and makes no assumptions about the generation source of the graph, making it directly applicable to any connected RDF/triple graph.

---

## 2. Concern: Outlier Isolation Correlation Effect Size is Small
> **Reviewer Comment:** *"The Spearman rank correlation coefficient ($r = -0.2392$) for the Outlier Isolation study is statistically significant but small, suggesting the effect might not be strong enough to justify the SOS-based ACBS sorting."*

### Proposed Response Strategy
- **Acknowledge and Contextualize**: We report the Spearman rank correlation of $-0.2392$ (and Pearson of $-0.4190$, $p = 0.0033$) as *supporting evidence* for the Outlier Isolation hypothesis rather than a definitive proof of a singular causative factor.
- **Factual Retention vs. Correlation**: While the correlation is moderate, the physical impact of exploiting this asymmetry via ACBS is large: the U-shaped boundary stratification places these outliers in high-attention boundaries, leading to a **+71.4% relative coverage improvement** on large graphs.
- **Rigorous Reporting**: We report both correlation metrics, p-values, and 95% Confidence Intervals to ensure transparent and rigorous representation of effect sizes, avoiding over-claiming.

---

## 3. Concern: Generalizability to Other LLMs
> **Reviewer Comment:** *"The results might be highly dependent on the choice of Qwen 2.5. How does the proposed framework generalize to other architectures?"*

### Proposed Response Strategy
- **Cross-Model Benchmarks**: To demonstrate that SMART-Graph is model-agnostic, we benchmarked the pipeline across three distinct open-weights model families:
  1. **Qwen 2.5 (7B)**
  2. **Llama 3 (8B)**
  3. **Gemma (7B)**
- **Effect Size Validation**: We computed paired t-tests and Cohen's $d$ paired effect size across models. 
  - On **Llama 3**, SMART-Graph achieved a statistically significant improvement ($p = 0.0392$, Cohen's $d = 2.8289$).
  - On **Gemma**, a large effect size was demonstrated (Cohen's $d = 0.9952$).
  - This confirms that attention decay is a shared architectural constraint of transformers, and input calibration via SMART-Graph benefits them collectively.

---

## 4. Concern: Fragility of Fact Extraction Metrics
> **Reviewer Comment:** *"Fact extraction using LLMs can introduce hallucinations or parsing failures, corrupting the coverage metrics."*

### Proposed Response Strategy
- **Soft-Match Mitigation (SMGA)**: To prevent evaluation artifacts from phrasing shifts (e.g. "Adolfo Suarez Airport" vs "Madrid Airport"), we developed Soft-Match Graph Alignment (SMGA).
- **Hungarian Optimization**: Rather than exact string matching (which results in 0.00% coverage on large graphs), SMGA models alignment as a maximum-weight bipartite matching problem solved using the Hungarian algorithm, using a high cosine similarity threshold ($\tau=0.75$) to filter weak matches.
- **Double-Extraction Safeguards**: We report both Strict (exact) and SMGA (soft) metrics, and run an automated extraction audit logging cases where strict and soft coverage deviate by more than 30% for manual verification.
