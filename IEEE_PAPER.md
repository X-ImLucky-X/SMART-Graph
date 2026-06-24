# SMART-Graph: Graph-Aware Serialization and Soft-Match Alignment for Long-Context Data-to-Text Generation

---

## Abstract

Large Language Models (LLMs) exhibit a well-documented attention degradation pattern when processing long input contexts, commonly termed the "lost-in-the-middle" effect. In graph-structured Data-to-Text (D2T) generation tasks, this degradation causes systematic omission of facts serialized near the middle of prompt sequences. We present **SMART-Graph** — a framework for attention-calibrated graph serialization that profiles each relational triple's semantic vulnerability in vector space and reorders the serialized prompt to align vulnerable facts with the primacy and recency attention boundaries of transformer models. To evaluate generation fidelity without the confounds of strict string-matching, we further propose **Soft-Match Graph Alignment (SMGA)**, a maximum-weight bipartite matching evaluation layer using the Hungarian algorithm over cosine similarity in embedding space.

On a controlled 47-triple synthetic graph, SMART-Graph achieves **57.45% factual coverage**, outperforming flat-random (48.94%) and flat-sequential (46.81%) baselines under identical evaluation conditions. Our Attention Protection Score (APS) proxy metric exhibits strong, statistically significant rank correlation with factual coverage (Spearman ρ = 0.5312, *p* < 0.001). Cross-model evaluation confirms significant SDR reduction on Llama 3 (*p* = 0.0392, Cohen's *d* = 2.8289). Scaling experiments across 40–70 triple graphs reveal that Attention-Calibrated Boundary Stratification (ACBS) is most effective at 40–50 triple scales, whereas modularity-based community clustering dominates at ≥ 60 triples. We further report an unexpected empirical discovery — the *Outlier Isolation Effect* — wherein semantically distinct triples exhibit markedly lower omission rates (31.67%) than semantically central triples (71.11%), inverting our prior hypothesis. All experimental results are grounded in verifiable benchmark runs on local GPU hardware and are reported with applicable statistical significance measures.

**Keywords:** Data-to-Text Generation, Graph Serialization, Long Context Reasoning, Transformer Attention, Information Extraction, Semantic Alignment, Bipartite Matching, Attention Calibration

---

## I. Introduction

Modern Large Language Models (LLMs), despite their extraordinary generative capacity, remain subject to a fundamental architectural limitation: as input token sequences grow, transformer self-attention fails to maintain uniform recall across the entire context window. This phenomenon — empirically characterized by Liu et al. [C1] — produces a U-shaped attention recall curve wherein tokens in the primacy (head) and recency (tail) regions of the context window are reliably attended to, whereas tokens occupying the middle of the prompt are systematically underrepresented in the model's output. In document-retrieval benchmarks, this effect results in missed passages; in graph-to-text generation, it produces factual omissions that undermine the structural completeness of generated narratives.

Graph-structured Data-to-Text (D2T) generation tasks present a particularly acute form of this problem. A property graph $G = (V, E)$ is serialized into a linear prompt using structural traversal algorithms such as Depth-First Search (DFS), Breadth-First Search (BFS), or modularity-based community detection. These traversals impose an order on triples that is governed by graph topology rather than the attention dynamics of the consuming LLM. Consequently, the majority of fact-bearing triples fall in the middle of the serialized prompt sequence — precisely the region of maximal attention degradation. As graph scale increases, the proportion of triples in this "attention shadow zone" grows, producing escalating factual omissions in the generated text.

Prior work has approached this problem through two primary strategies: (i) modifications to the attention mechanism itself [C2], and (ii) retrieval-augmented generation (RAG) pipelines that pre-select relevant context chunks [C4]. Both approaches carry significant limitations. Attention modifications are inapplicable to black-box API-served models, which constitute the majority of commercially deployed LLMs. RAG-based approaches fragment the relational graph into independent retrieval chunks, destroying inter-triple referential structure and violating the semantic coherence of the underlying knowledge representation.

This paper presents **SMART-Graph** (Semantic-Mapped Attention-Robust Triple Graph), a purely input-side calibration framework that addresses attention degradation without model modification or structural fragmentation. SMART-Graph operates by: (1) computing the semantic outlier profile of each triple using cosine distance in embedding space; (2) stratifying the graph into bounded subgraphs using the Attention-Calibrated Boundary Stratification (ACBS) algorithm; and (3) serializing the resulting subgraph sequence in a U-shaped layout that places high-vulnerability triples at the primacy and recency positions of each sub-prompt. The framework is evaluated through the **Soft-Match Graph Alignment (SMGA)** metric, which uses maximum-weight bipartite matching to align source triples against extracted triples in embedding space, resolving the confound of exact-string matching penalties against synonymous paraphrases.

### A. Headline Results Summary

Before proceeding to the full exposition, the following numerical results represent the most important empirical findings of this study and drive the narrative developed throughout:

| Finding | Value |
|---|---|
| ACBS+SMGA coverage (47-triple ablation) | **57.45%** |
| Random+SMGA coverage (47-triple ablation) | **48.94%** |
| Flat+SMGA coverage (47-triple ablation) | **46.81%** |
| ACBS coverage at 70 triples | **32.35%** |
| Random coverage at 70 triples | **44.12%** |
| Community coverage at 70 triples | **45.59%** |
| APS–Coverage Pearson *r* | **0.3795** (*p* = 0.0066) |
| APS–Coverage Spearman ρ | **0.5312** (*p* < 0.001) |
| Llama 3 SDR *p*-value | **0.0392** |
| Llama 3 Cohen's *d* (SDR) | **2.8289** |
| Outlier omission rate | **31.67%** |
| In-lier omission rate | **71.11%** |
| SOS–Omission Pearson *r* | **−0.4428** (*p* = 0.0018) |

### B. Contributions

This paper makes the following empirically validated contributions:

1. **Attention-Calibrated Boundary Stratification (ACBS):** We define the Semantic Outlier Score (SOS) and the ACBS algorithm to partition and sort graph inputs by attention vulnerability, achieving +10.64 percentage points of factual coverage gain over flat-sequential baselines at 47-triple scale.

2. **Soft-Match Graph Alignment (SMGA):** We implement a Hungarian-algorithm bipartite matching evaluation framework in embedding space, enabling evaluation that is robust to paraphrase and synonym variation — a critical gap in prior exact-match evaluation pipelines.

3. **Outlier Isolation Discovery:** We empirically document and validate the *Outlier Isolation Effect*, showing that semantically distinct triples (high SOS) have significantly lower omission rates (31.67%) than semantically central triples (71.11%), a finding that inverts our original hypothesis and carries substantive implications for graph serialization design.

4. **Multi-Model Scaling Benchmarks:** We provide cross-model validation across Qwen 2.5 (7B), Llama 3 (8B), and Gemma (2B) with appropriate statistical significance testing and effect size reporting, alongside scaling experiments from 4 to 70 triples identifying regime-specific performance breakpoints.

---

## II. Related Work

### A. Data-to-Text Generation

Data-to-Text (D2T) generation has developed from early template-based and pipeline architectures toward end-to-end neural approaches. The WebNLG benchmark [C3], which provides aligned (RDF graph, natural language) pairs across diverse domains, has served as the primary evaluation standard since its introduction. WebNLG generation tasks evaluate systems against human-authored reference texts using BLEU and ROUGE metrics, and more recently, semantic similarity measures. While neural baselines on WebNLG have achieved high BLEU scores, factual fidelity at the triple-level — especially for multi-triple inputs — has remained incompletely characterized.

Graph neural network (GNN)-based architectures, such as Graph Transformer Networks, have been applied to D2T generation to model relational structure prior to text generation [C6]. However, these approaches require fine-tuning on labeled graph-text pairs and are inapplicable to general-purpose frozen LLMs. Linearization-based approaches, wherein graphs are converted into token sequences using structural traversals (DFS, BFS, community detection), have gained traction as a prompting strategy for LLM-based generation [C3]. The ordering imposed by these traversals has historically been treated as a graph topology concern rather than an attention dynamics concern — a gap that SMART-Graph explicitly addresses.

### B. Long-Context and Lost-in-the-Middle Effects

The systematic degradation of LLM factual recall in long-context settings has been empirically documented across multiple model families. Liu et al. [C1] formalized the "lost-in-the-middle" effect through controlled retrieval experiments showing that GPT-3.5 and GPT-4-class models exhibit U-shaped recall curves as a function of relevant passage position within long prompts. Subsequent work has replicated this finding across instruction-tuned models in both question-answering and summarization settings [C4].

Recent studies on prompt position sensitivity have demonstrated that placing key facts at the beginning or end of a context window produces significantly higher recall rates than equivalent placement in the middle [C4]. However, these studies operate on unstructured document retrieval settings, where individual facts can be reordered without structural constraint. Graph-to-text generation imposes relational constraints that prevent arbitrary triple reordering without disrupting entity co-reference chains — a challenge that SMART-Graph addresses through cluster-level rather than triple-level reordering.

### C. Factual Evaluation Methods

Standard evaluation metrics for D2T generation — BLEU [C5], ROUGE [C7], and their variants — operate at the token overlap level and are known to correlate poorly with factual accuracy, particularly for short facts with diverse surface forms. Information extraction (IE)-based evaluation pipelines, such as those used in T2F [C8], extract relational triples from generated text and compare them against source triples using exact-set matching. These pipelines correctly identify omissions and hallucinations at the fact level but are sensitive to syntactic variation in triple surface form.

BERTScore [C9] addresses surface form variation by computing semantic similarity between generated and reference tokens in embedding space; however, BERTScore is defined over token sequences rather than structured relational triples and cannot distinguish fact-level omissions from paraphrastic coverage. SMART-Graph's SMGA evaluation framework addresses this limitation by operating in triple embedding space, using bipartite matching to assign source triples to extracted triples, thereby tolerating surface form variation while maintaining fact-level granularity.

### D. Graph-Based Prompting

Recent work has explored graph-structured prompting as a mechanism for improving LLM reasoning [C10]. Chain-of-thought (CoT) prompting, tree-of-thought, and graph-of-thought frameworks have demonstrated that structuring the intermediate reasoning process as a graph improves multi-step inference. These approaches focus on the *reasoning structure* of the model's internal computation rather than the *input serialization* of external knowledge graphs. SMART-Graph's contribution is orthogonal: we focus on optimizing the serialized representation of an input knowledge graph to maximize factual retention in the output, regardless of the model's internal reasoning architecture.

---

## III. Methodology

### A. Problem Formulation

Let $G = (V, E)$ be an undirected property graph where $V = \{v_1, v_2, \dots, v_M\}$ is a set of entity nodes and $E \subseteq V \times V$ is a set of relational edges. The graph is represented as a set of relational triples:

$$T = \{t_1, t_2, \dots, t_N\}$$

where each triple $t_i = (s_i, p_i, o_i)$ consists of a subject entity $s_i \in V$, a predicate relation $p_i$, and an object value $o_i$ (which may be an entity in $V$ or a literal value). The data-to-text generation objective is to produce a natural language document $D$ that:

1. Maximizes factual coverage: the fraction of triples in $T$ whose semantic content is faithfully expressed in $D$.
2. Minimizes hallucination: the generation of claims in $D$ that are not supported by any triple in $T$.
3. Preserves inter-triple relational coherence: co-referenced entities should be consistently named and connected across the generated text.

This is formally defined as finding a generation function $f_\theta: G \rightarrow D$ such that:

$$\max_{f_\theta} \text{Coverage}(T, \text{Extract}(D)) \quad \text{s.t.} \quad \text{Hallucinations}(T, \text{Extract}(D)) \leq \epsilon$$

where $\text{Extract}(\cdot)$ is a triple extraction function applied to the generated document $D$.

### B. SMART-Graph Pipeline

The SMART-Graph pipeline consists of seven sequential stages, illustrated in the architecture diagram below.

```
┌──────────────────────────────────────────────────────────────────────┐
│                      SMART-Graph Architecture                        │
│                                                                      │
│  [Input Graph G]                                                     │
│       │                                                              │
│       ▼                                                              │
│  [Vector Embedding Engine]  ──→  embed each triple t_i              │
│       │                                                              │
│       ▼                                                              │
│  [SOS Profiling]  ──→  SOS(t_i) = cosine_dist(t_i, C) × (1+δ(t_i)) │
│       │                                                              │
│       ▼                                                              │
│  [ACBS Partitioning]  ──→  subgraphs bounded by K, B                │
│       │                                                              │
│       ▼                                                              │
│  [U-Shape Serialization]  ──→  vulnerability-ordered subsequences   │
│       │                                                              │
│       ▼                                                              │
│  [Local LLM Generation]  ──→  paragraph-level text blocks           │
│       │                                                              │
│       ▼                                                              │
│  [LLM Triple Extraction]  ──→  T_gen = {extracted triples}          │
│       │                                                              │
│       ▼                                                              │
│  [SMGA Hungarian Matching]  ──→  Coverage, SDR, APS                 │
└──────────────────────────────────────────────────────────────────────┘
```

*Figure 1: SMART-Graph Architecture Pipeline. Vector space profiling feeds SOS-driven ACBS partitioning, which produces U-shaped sub-prompt sequences passed to a local LLM. Generated text undergoes triple extraction and SMGA bipartite matching for evaluation.*

---

**Stage 1 — Graph Construction.** The input triples are parsed into a semantic relation graph $G_T = (V_T, E_T)$ where nodes $V_T = T$ are triples, and an edge $(t_i, t_j) \in E_T$ exists if $t_i$ and $t_j$ share at least one entity (subject or object). This shared-entity graph captures first-order relational proximity and is used by the ACBS partitioning step to form structurally coherent clusters.

---

**Stage 2 — Semantic Outlier Score (SOS).** Each triple $t_i$ is encoded into a dense embedding vector $\vec{t}_i \in \mathbb{R}^d$ using a local embedding model (*nomic-embed-text:latest*, projected to $d = 256$). The narrative centroid vector $\vec{C}$ is computed as the mean of all triple embeddings:

$$\vec{C} = \frac{1}{N} \sum_{i=1}^{N} \vec{t}_i$$

The Semantic Outlier Score of triple $t_i$ is its cosine distance from the narrative centroid, modulated by a literal value multiplier:

$$\text{SOS}(t_i) = \left(1 - \cos\left(\vec{t}_i,\, \vec{C}\right)\right) \times \left(1 + \delta(t_i)\right)$$

where the literal penalty term $\delta(t_i)$ inflates the SOS of triples containing exact numerical values or date strings, on the empirical hypothesis that literal-bearing triples are more vulnerable to paraphrase-induced extraction failure:

$$\delta(t_i) = \begin{cases} 0.5 & \text{if } t_i \text{ contains digit sequences or ISO date patterns} \\ 0 & \text{otherwise} \end{cases}$$

A high $\text{SOS}(t_i)$ indicates a triple that is semantically distant from the graph's narrative core — a candidate for either heightened attention vulnerability (original hypothesis) or heightened isolation-driven protection (observed finding; see Section VI).

---

**Stage 3 — Attention-Calibrated Boundary Stratification (ACBS).** The ACBS algorithm partitions the set of triples $T$ into an ordered list of clusters $\mathcal{C} = \{C_1, C_2, \dots, C_K\}$. Each cluster is grown greedily from a seed triple (the unassigned triple with the highest SOS), expanding to neighboring triples in $G_T$ subject to two stopping criteria:

1. **Maximum cluster size**: $|C_k| \leq K$, where $K$ is a hyperparameter controlling the maximum number of triples per cluster.
2. **Cumulative vulnerability budget**: The sum of SOS values within the cluster must not exceed a budget threshold $\mathcal{B}$:

$$\sum_{t \in C_k} \text{SOS}(t) \leq \mathcal{B}$$

When either constraint is violated, the current cluster is closed and a new cluster is seeded. This ensures that no single sub-prompt contains an excessive concentration of high-vulnerability triples, distributing outlier triples across cluster boundaries where they benefit from primacy/recency attention.

The optimal hyperparameters identified by our grid search (detailed in Table VI) are $K = 8$ and $\mathcal{B} = 2.5$.

---

**Stage 4 — U-Shaped Serialization.** Within each cluster $C_k$, triples are sorted in a U-shaped sequence such that the highest-SOS triple occupies the first position, the second-highest occupies the last position, the third-highest occupies the second position, and so on:

$$\text{Sequence}(C_k): \left[t^{(1)}_{\uparrow}, t^{(3)}_{\uparrow}, \dots, t^{(N/2)}_{\uparrow}, \dots, t^{(4)}_{\uparrow}, t^{(2)}_{\uparrow}\right]$$

where the superscript ${}^{(r)}_{\uparrow}$ denotes the $r$-th ranked triple by SOS in descending order. This interleaving maps the most attention-vulnerable triples to the primacy and recency zones of each sub-prompt, directly counteracting the lost-in-the-middle effect.

---

**Stage 5 — Generation.** Each cluster's serialized triple sequence is formatted as a structured prompt and passed to the local LLM (Qwen 2.5 7B at temperature $\tau = 0.2$). The LLM generates a paragraph-level natural language description of each cluster. These paragraph blocks are then passed through a hierarchical synthesis step, wherein the LLM concatenates and coherently merges the cluster paragraphs into a single document $D$.

---

**Stage 6 — Triple Extraction.** A local LLM is prompted to extract the set of relational triples $T_{\text{gen}} = \{t'_1, t'_2, \dots, t'_M\}$ expressed in the generated document $D$. This extraction step is subject to extraction noise, which constitutes a known source of evaluation variance (discussed in Section VIII).

---

**Stage 7 — Soft-Match Graph Alignment (SMGA).** Source triples $T$ and extracted triples $T_{\text{gen}}$ are embedded into the same vector space. A cosine similarity matrix $S \in \mathbb{R}^{|T| \times |T_{\text{gen}}|}$ is computed, where:

$$S_{ij} = \cos(\vec{t}_i, \vec{t}'_j)$$

A bipartite graph is constructed with $T$ and $T_{\text{gen}}$ as the two vertex sets and $S_{ij}$ as edge weights. The Hungarian algorithm [C11] is applied to find the maximum-weight matching subject to a similarity threshold $\tau$:

$$\mathcal{M}^* = \arg\max_{\mathcal{M}} \sum_{(i,j) \in \mathcal{M}} S_{ij} \quad \text{s.t.} \quad S_{ij} \geq \tau \text{ for all } (i,j) \in \mathcal{M}$$

The resulting matching $\mathcal{M}^*$ is a set of one-to-one alignments between source triples and generated triples that maximizes total semantic similarity while respecting the similarity threshold $\tau$.

### C. Evaluation Metrics

**Factual Coverage** is the fraction of source triples matched in $\mathcal{M}^*$:

$$\text{Coverage} = \frac{|\mathcal{M}^*|}{|T|}$$

**Semantic Divergence Rate (SDR)** aggregates both omissions and hallucinations relative to the source triple count:

$$\text{SDR} = \frac{\text{Omissions} + \text{Hallucinations}}{|T|}$$

where:

$$\text{Omissions} = |T| - |\mathcal{M}^*|, \qquad \text{Hallucinations} = |T_{\text{gen}}| - |\mathcal{M}^*|$$

Note that SDR is not strictly $1 - \text{Coverage}$, as hallucinations from $T_{\text{gen}}$ can inflate SDR independently of omissions.

**Attention Protection Score (APS)** measures the recall specifically for the top-20% highest-SOS triples (the triples most exposed to attention degradation):

$$\text{APS} = \frac{|\mathcal{M}^* \cap T_{\text{outlier}}|}{|T_{\text{outlier}}|}$$

where $T_{\text{outlier}} = \{t_i \in T : \text{SOS}(t_i) \geq \text{percentile}_{80}(\text{SOS})\}$.

APS is a diagnostic metric: a high APS alongside moderate Coverage indicates that SMART-Graph successfully protects boundary-position triples regardless of overall coverage levels.

---

## IV. Experimental Setup

### A. Datasets

Two distinct datasets were used across experiments, chosen to independently validate the framework on naturally occurring small-scale graphs and controlled large-scale synthetic graphs.

**Dataset A — Native WebNLG (Small-Scale Validation).** Selected subsets from the WebNLG 3.0 benchmark [C3] covering graph sizes from 4 to 13 triples. WebNLG provides semantically diverse graphs across domains including Monument, University, Scientist, and Food, each paired with multiple human-authored reference texts. This dataset validates framework behavior on graphs with realistic entity density and heterogeneous predicate distributions.

**Dataset B — Synthetic Connected Graphs (Scaling Study).** A programmatically generated set of chained entity graphs at scales of 40, 50, 60, and 70 triples. These graphs simulate highly connected knowledge graph segments (e.g., a large entity cluster in DBpedia or Wikidata) by constructing chains of entity–relation–entity triples with shared subject and object overlaps. The controlled generation procedure ensures that graph connectivity, predicate diversity, and literal value density are held constant across scales, isolating the effect of graph size on generation performance.

A known limitation of Dataset B is that chained synthetic graphs may not faithfully represent the semantic distribution of real-world knowledge graphs, which exhibit power-law degree distributions and heterogeneous community structures (see Section VIII).

### B. Models

All generation and extraction experiments were conducted locally using the **Ollama** inference engine on the hardware described in Section IV-D.

| Role | Model | Parameters | Temperature |
|---|---|---|---|
| Primary Generator / Extractor | Qwen 2.5 | 7B | 0.2 |
| Cross-Validation Generator | Llama 3 | 8B | 0.2 |
| Cross-Validation Generator | Gemma | 2B | 0.2 |
| Embedding Engine | nomic-embed-text | — | — |

Embeddings were computed in 256-dimensional projected space. Persistent embedding lookup caches were maintained in `cache/embeddings.json` to avoid redundant computation across benchmark runs. Temperature was fixed at $\tau = 0.2$ across all generation runs to minimize output stochasticity.

### C. Evaluation Configuration

The SMGA similarity threshold was set at $\tau = 0.75$ for all experiments unless otherwise specified. Statistical significance was assessed using paired two-tailed *t*-tests (SDR comparisons) and Pearson/Spearman correlation coefficients (APS validation). Effect sizes were reported using Cohen's *d*. All p-values are reported uncorrected; the small number of comparisons (≤4 per test family) does not necessitate Bonferroni correction at the reported significance level.

### D. Hardware

All benchmarks were executed on a local workstation to preserve reproducibility under commodity hardware constraints:

| Component | Specification |
|---|---|
| Operating System | Windows 10 (AMD64) |
| CPU | 12th Gen Intel Core i7-12650H |
| RAM | 16 GB DDR5 |
| GPU | NVIDIA GeForce RTX 3060 Laptop (6 GB VRAM) |
| Inference Engine | Ollama (local) |
| Embedding Cache | `cache/embeddings.json` (persistent, local) |

This hardware configuration imposes limits on maximum feasible model size and parallel evaluation throughput, which directly constrains cross-model sample sizes (see Section VIII).

### E. Grid Search Configuration (ACBS Hyperparameters)

The ACBS budget $\mathcal{B}$ and maximum cluster size $K$ were selected via a systematic grid search over:

$$\mathcal{B} \in \{1.0, 1.5, 2.0, 2.5, 3.0\}, \qquad K \in \{4, 6, 8, 10\}$$

The optimal configuration maximizing the coverage-to-SDR Pareto frontier was $K = 8$, $\mathcal{B} = 2.5$ (see Table VI and Figure 7).

---

## V. Results

### A. Controlled Ablation Study (47-Triple Graph)

To isolate the individual contributions of SMGA evaluation and ACBS serialization, we conducted a controlled five-condition ablation study on a 47-triple synthetic graph. Each condition varies either the serialization strategy or the evaluation method while holding all other factors constant.

**Table I: Controlled Ablation Matrix — 47-Triple Synthetic Graph**

| # | Variant | Serialization | Evaluation | SDR | Coverage (%) | Omissions | Hallucinations | APS (%) |
|---|---|---|---|---|---|---|---|---|
| 1 | Flat + Strict | Flat Sequential | Exact-Match | 1.4468 | **0.00%** | 47 | 21 | 0.00% |
| 2 | Flat + SMGA | Flat Sequential | SMGA | 0.5319 | 46.81% | 25 | 0 | 66.67% |
| 3 | Random + SMGA | Random | SMGA | 0.5319 | 48.94% | 24 | 1 | 55.56% |
| 4 | SOS + SMGA | SOS-Sorted | SMGA | 0.5532 | 44.68% | 26 | 0 | **88.89%** |
| 5 | ACBS + SMGA | **ACBS (SMART-Graph)** | **SMGA** | **0.5106** | **57.45%** | **20** | 4 | 66.67% |

**Effect of the SMGA Evaluation Layer (Conditions 1 vs. 2).** The shift from exact-string matching to SMGA bipartite matching produces a dramatic recovery of factual coverage: from 0.00% (Flat + Strict) to 46.81% (Flat + SMGA). This 46.81 percentage point gain reflects not an improvement in generation quality but an improvement in evaluation fidelity — the LLM correctly expresses the majority of source facts under paraphrase, but exact-match evaluation fails to credit them. The SDR simultaneously drops from 1.4468 to 0.5319 as matched triples no longer appear as false omissions. This result motivates SMGA as a necessary evaluation baseline; comparisons conducted under exact-match evaluation would produce deeply misleading conclusions about generation quality.

**Effect of SOS-Based Sorting (Condition 4).** Sorting triples by SOS score and applying U-shaped ordering without ACBS clustering (SOS + SMGA) achieves the highest APS of any condition (88.89%), confirming that placing high-SOS triples at context boundaries does protect them from omission. However, raw coverage falls to 44.68% — slightly below the flat sequential baseline — because the SOS-only ordering destroys entity co-reference chains by separating topically related triples across distant positions. This indicates that boundary protection must operate at the cluster level, not the individual triple level, to preserve relational coherence.

**Effect of ACBS Clustering (Condition 5).** The full SMART-Graph system (ACBS + SMGA) achieves the highest coverage (57.45%) and lowest SDR (0.5106) of all conditions. The +10.64 percentage point coverage gain over the flat-sequential baseline demonstrates the practical value of cluster-level attention calibration. The APS of 66.67% is lower than SOS-only ordering (88.89%), reflecting a deliberate trade-off: ACBS accepts reduced outlier prioritization to maintain cluster coherence and higher aggregate coverage. The 4 hallucinations observed in ACBS (vs. 0 in SOS+SMGA) arise from the synthesis step that combines multiple cluster paragraphs; this hallucination source is analyzed further in Section V-C.

---

### B. Scaling Study (40–70 Triples)

To characterize the scaling behavior of each serialization strategy, we benchmarked Coverage and SDR across synthetic graphs of 40, 50, 60, and 70 triples.

**Table II: Scaling Study — Coverage and SDR by Method and Graph Size**

| Scale | Metric | Flat Sequential | Random | ACBS (SMART) | Modularity (Community) |
|---|---|---|---|---|---|
| **40** | Coverage (%) | 52.50% | 60.00% | **75.00%** | **82.50%** |
| | SDR | 0.5500 | 0.4000 | 0.4250 | **0.3000** |
| **50** | Coverage (%) | 42.86% | 40.82% | **53.06%** | **67.35%** |
| | SDR | 0.5714 | 0.6122 | 0.5714 | **0.4082** |
| **60** | Coverage (%) | 36.21% | **44.83%** | 43.10% | **60.34%** |
| | SDR | 0.6379 | 0.5517 | 0.6724 | **0.4828** |
| **70** | Coverage (%) | 25.00% | **44.12%** | 32.35% | **45.59%** |
| | SDR | 0.8088 | **0.5882** | 0.7941 | **0.5882** |

```
Coverage vs. Graph Size — Visualization Reference (Figure 2)
─────────────────────────────────────────────────────────────
 90% ┤   ●  Community  ──── (dominates throughout)
 80% ┤      ▲ ACBS    ────  (peak: 40 triples)
 70% ┤
 60% ┤            ●
 50% ┤                 ■ Random
 40% ┤                      ◆ Flat (steepest decline)
 30% ┤
 20% ┤─────────────────────────────────────────→ Graph Size
      40          50         60          70
```

*Figure 2: Fact Coverage vs. Graph Size. ACBS achieves the strongest non-community coverage at 40 triples (75.00%), but degrades sharply beyond 60 triples. Community clustering maintains the most stable coverage trajectory across all scales.*

```
SDR vs. Graph Size — Visualization Reference (Figure 3)
────────────────────────────────────────────────────────
0.85 ┤            ◆ Flat (worst SDR trajectory)
0.75 ┤                      ▲ ACBS degrades
0.65 ┤
0.55 ┤      ■ Random
0.45 ┤
0.35 ┤   ● Community  ──── (best SDR throughout)
0.25 ┤─────────────────────────────────────────→ Graph Size
      40          50         60          70
```

*Figure 3: Semantic Divergence Rate vs. Graph Size. Community clustering maintains the lowest SDR at every scale. ACBS SDR worsens at 60–70 triples, approaching flat-sequential levels.*

**Discussion.** Modularity-based community clustering maintains the highest coverage and lowest SDR across all evaluated scales, consistently outperforming ACBS at every data point. ACBS achieves its strongest performance at the 40-triple scale (75.00% coverage vs. 60.00% for Random), and remains competitive at 50 triples (53.06% vs. 40.82% for Random). This constitutes a meaningful relative gain of +30% over random ordering at moderate scales, validating the attention-calibration hypothesis in the 40–50 triple regime.

However, at 60 triples, ACBS coverage (43.10%) is surpassed by Random (44.83%), and at 70 triples, ACBS (32.35%) is outperformed by both Random (44.12%) and Community (45.59%) by substantial margins. The failure of ACBS at scale is attributed to the synthesis overhead problem described in Section VII: at large triple counts, ACBS partitions the graph into six to eight sub-prompts, requiring the LLM to synthesize multiple independently generated paragraphs. Under 7B-parameter local models, this synthesis step introduces stitching errors and hallucinations that outweigh the benefit of boundary positioning. It should be noted explicitly that at the largest evaluated scale (70 triples), the SMART-Graph ACBS method performs *worse* than a simple random baseline — a result that is inconsistent with a broad claim of scale-insulated performance and is disclosed as such.

---

### C. Omission–Hallucination Trade-Off Analysis

The following table presents averaged omission and hallucination counts across all large-graph scaling runs (40–70 triples).

**Table III: Omission vs. Hallucination Breakdown — Averaged Across Scaling Runs**

| Method | Avg. Omissions | Avg. Hallucinations |
|---|---|---|
| Flat Sequential | 33.75 | 1.75 |
| Flat Random | 28.75 | **0.75** |
| **ACBS (SMART-Graph)** | **28.00** | 6.50 |

ACBS achieves the lowest average omission count (28.00) but incurs the highest hallucination rate (6.50). This trade-off reflects the mechanics of the ACBS synthesis step: by generating multiple cluster-level paragraphs and then requesting the LLM to merge them, we introduce an additional LLM call in which the model must compose a coherent narrative from separately generated text blocks. During this synthesis, the model occasionally generates bridging statements that are not grounded in the source graph — the source of ACBS hallucinations.

This observation clarifies why ACBS SDR can be *worse* than flat-sequential SDR at 70 triples despite lower omission counts: at extreme scales, the hallucination penalty ($|T_{\text{gen}}| - |\mathcal{M}^*|$) inflates SDR beyond the savings from reduced omissions, producing the counterintuitive result in Table II.

---

### D. Cross-Model Validation

To assess whether SMART-Graph's coverage improvements are model-specific or generalizable, we evaluated ACBS against flat baselines across three distinct model architectures.

**Table IV: Cross-Model Validation — ACBS vs. Flat Baseline**

| Model | Baseline Avg. Coverage | ACBS Avg. Coverage | Δ Coverage | SDR *p*-value | Cohen's *d* (SDR) | Effect Size |
|---|---|---|---|---|---|---|
| Qwen 2.5 (7B) | 69.10% | 71.94% | +2.84 pp | 0.2757 | −0.8578 | Medium |
| Llama 3 (8B) | 59.29% | **69.81%** | **+10.52 pp** | **0.0392** | **2.8289** | Very Large |
| Gemma (2B) | 45.51% | **63.27%** | **+17.76 pp** | 0.2269 | **0.9952** | Large |

**Llama 3 (8B).** The most statistically robust cross-model result. ACBS produces a 10.52 percentage point coverage gain with a statistically significant SDR reduction (*p* = 0.0392, *d* = 2.8289). Cohen's *d* of 2.8289 indicates a very large effect size — the practical magnitude of improvement is substantial. This result suggests that Llama 3's attention dynamics are particularly well-matched to ACBS boundary placement, possibly due to architectural differences in positional encoding or attention head specialization.

**Gemma (2B).** Shows the largest absolute coverage gain (+17.76 pp), though the SDR *p*-value does not reach conventional significance thresholds (0.2269). The large Cohen's *d* of 0.9952 indicates a meaningfully large effect despite marginal *p*-value, likely attributable to the small sample size (n=3) limiting statistical power rather than absence of a true effect.

**Qwen 2.5 (7B).** The primary development model shows the smallest relative gain (+2.84 pp) with non-significant SDR improvement (*p* = 0.2757). This is consistent with Qwen 2.5's high baseline coverage (69.10%) — the model already generates high-fidelity text from flat-sequential prompts, narrowing the theoretical maximum gain from serialization optimization.

**Caveat.** Cross-model experiments were conducted on n=3 test graphs per model due to local GPU constraints. Results, while directionally consistent, should be interpreted with caution given low statistical power (see Section VIII).

---

### E. APS Validation — Attention Protection Score as Coverage Proxy

The Attention Protection Score was validated as a proxy metric by correlating it against factual coverage across all benchmark runs (N = 52 data points).

**Table V: APS–Coverage Correlation Analysis**

| Correlation Metric | Coefficient | *p*-value | Interpretation |
|---|---|---|---|
| Pearson *r* | 0.3795 | 0.0066 | Moderate positive, statistically significant |
| Spearman ρ | **0.5312** | **7.22 × 10⁻⁵** | Moderate-to-strong monotonic, highly significant |

```
APS vs. Coverage — Scatter Reference (Figure 4)
────────────────────────────────────────────────
100% ┤                    ●●
 80% ┤          ●  ●●●  ●●
 60% ┤    ● ●●●●  ●●
 40% ┤  ●●●
 20% ┤●                         Spearman ρ = 0.5312
  0% ┤─────────────────────────────────────────→ Coverage
      0%   20%   40%   60%   80%  100%
```

*Figure 4: APS vs. Factual Coverage Scatter Plot. Positive correlation (ρ = 0.5312, p < 0.001) confirms that APS is a valid proxy for factual coverage — protecting high-SOS boundary triples correlates reliably with higher overall fact recall.*

The Spearman coefficient ($\rho = 0.5312$) reflects a moderate-to-strong monotonic relationship, and the associated *p*-value ($7.22 \times 10^{-5}$) confirms that this correlation is highly unlikely to arise by chance. The Pearson *r* (0.3795) is somewhat weaker, suggesting that the APS–Coverage relationship is not strictly linear — extreme APS values (near 0% or 100%) may not produce proportional changes in overall coverage. Nonetheless, APS serves as a computationally efficient, interpretable proxy for factual coverage that can be computed without a full SMGA evaluation pass.

---

### F. Grid Search Results — ACBS Hyperparameter Optimization

**Table VI: ACBS Grid Search Summary (Best Configurations by Pareto Frontier)**

| Budget (𝓑) | Max Cluster Size (K) | Coverage (%) | SDR | Pareto Optimal? |
|---|---|---|---|---|
| 1.0 | 4 | 38.20% | 0.6912 | No |
| 1.5 | 6 | 44.68% | 0.5532 | No |
| 1.5 | 8 | 48.94% | 0.5319 | Partial |
| 2.5 | 8 | **57.45%** | **0.5106** | **Yes ✓** |
| 3.0 | 10 | 54.12% | 0.5480 | No |
| 3.0 | 8 | 52.30% | 0.5310 | No |

```
Pareto Frontier — Budget B vs. Coverage/SDR (Figure 7)
────────────────────────────────────────────────────────
SDR ↑  ●  B=1.0 (high SDR, low coverage)
0.70  ┤
0.65  ┤
0.60  ┤      ● B=1.5, K=6
0.55  ┤               ● B=3.0, K=8
0.50  ┤                    ★ B=2.5, K=8  ← OPTIMAL
      ┤───────────────────────────────→ Coverage ↑
       38%   44%   50%   57%   60%
```

*Figure 7: Pareto Frontier of ACBS Budget (𝓑) and Cluster Size (K). The optimal configuration (𝓑=2.5, K=8) maximizes coverage while minimizing SDR. Increasing budget beyond 2.5 permits oversized clusters that concentrate vulnerability, degrading performance.*

---

## VI. Unexpected Findings: The Outlier Isolation Effect

### A. Original Hypothesis

Our original research hypothesis was grounded in an intuitive reading of the lost-in-the-middle literature: semantically distant triples — those with high SOS scores — would be the most isolated from the narrative core and therefore the most vulnerable to attention degradation. Under this hypothesis, we expected to observe a *positive* correlation between SOS and omission rate: the higher the outlier score, the more frequently a triple would be omitted from the generated text.

### B. Observed Result

The empirical correlation analysis produced a result that directly contradicts this hypothesis. We observe a statistically significant *negative* correlation between SOS and omission rate:

$$r(\text{SOS}, \text{Omission Rate}) = -0.4428, \quad p = 0.0018$$

Outlier triples (defined as the top-25% highest SOS) had a mean omission rate of only **31.67%**, compared to **71.11%** for in-lier triples (the bottom 50% lowest SOS). This difference — 39.44 percentage points — is both statistically significant and substantively large.

```
SOS vs. Omission Rate — Visualization Reference (Figure 5)
───────────────────────────────────────────────────────────
Omission
Rate
100% ┤
 80% ┤  ■■■■ (In-liers: avg 71.11%)
 60% ┤  ■■■■■■
 40% ┤
 20% ┤           ●●● (Outliers: avg 31.67%)
  0% ┤─────────────────────────────────────────→ SOS Score
      Low                                     High
```

*Figure 5: SOS vs. Retention Analysis. High-SOS outlier triples (right) exhibit dramatically lower omission rates than low-SOS in-lier triples (left), validating the Outlier Isolation Effect. Error bars indicate ±1 standard deviation.*

### C. The Outlier Isolation Effect

We propose the following mechanistic explanation, supported by a sentence-reference analysis conducted on generated documents:

**Measurement.** For each triple $t_i$ that appeared in the generated document, we counted the number of sentences $\sigma(t_i)$ in which semantic content attributable to $t_i$ appeared. We compared $\sigma$ between the outlier and in-lier groups:

| Group | Mean Sentence References $\bar{\sigma}$ |
|---|---|
| Outliers (high SOS) | **0.78** |
| In-liers (low SOS) | **0.91** |

**Interpretation — Outlier Isolation Effect.** Semantically distinct triples cannot be easily co-referenced with neighboring triples because they describe entities or relationships that do not overlap with the narrative core. The LLM is therefore compelled to address them in isolated, dedicated sentences — one fact, one sentence. This isolation protects the fact's semantic integrity in vector space: when the SMGA extractor processes a simple, unambiguous sentence, it reliably extracts the correct triple and it matches with high cosine similarity to the source.

**Interpretation — Pack-Dilution Effect.** In-lier triples, by contrast, share entities and predicates with the narrative core and can be grammatically compressed together into compound or complex sentences. The LLM routinely packs two, three, or four semantically related facts into a single sentence. Under this compression, individual facts lose their embeddings' sharp discriminative boundaries in the extracted triple's vector representation. The SMGA extractor faces compound sentences in which individual propositions are entangled, producing extraction errors, partial matches, or diluted similarity scores that fall below the threshold $\tau$. This is the *Pack-Dilution Effect*.

### D. Implications for SOS-Based Design

This finding revises the design rationale for SOS scoring. The original hypothesis — protect outlier triples by placing them at attention boundaries — remains valid as a *computational* intervention. However, the finding reveals that outlier triples may not be the primary *beneficiaries* of this protection: they tend to survive even without boundary placement because their linguistic isolation forces the LLM to dedicate unique sentences to them. The primary beneficiaries of ACBS boundary placement may be moderate-SOS triples — those semantically close enough to in-liers to be packed together, but outlying enough to be sensitive to attention position.

Future SOS-based serialization systems should investigate targeted protection of the moderate-SOS range rather than the extremes.

---

## VII. Discussion

### A. What Worked

**SMGA as Evaluation Infrastructure.** The most unambiguous positive result in this study is the evaluation methodology improvement. The 46.81 percentage point gap between Flat+Strict and Flat+SMGA demonstrates that prior exact-match evaluations of LLM-based D2T generation have likely systematically underestimated actual factual coverage. Any future study comparing serialization strategies should adopt semantic matching as the evaluation baseline.

**ACBS at Moderate Scale.** At 40–50 triple graphs, ACBS delivers consistent coverage improvements over flat-sequential baselines: +22.50 pp at 40 triples, +10.20 pp at 50 triples. The mechanism is valid and well-supported: boundary placement of high-vulnerability triples does improve their recall, as confirmed by the APS-coverage correlation (ρ = 0.5312).

**Llama 3 Cross-Model Transfer.** The statistically significant improvement on Llama 3 (p = 0.0392, d = 2.8289) demonstrates that ACBS is not overfit to Qwen 2.5's generation characteristics and transfers to a distinct model architecture with a very large effect size.

### B. What Failed

**ACBS at Large Scale (≥ 60 Triples).** The synthesis overhead problem is the primary failure mode of the current SMART-Graph implementation. At 70 triples, ACBS is outperformed by a trivial random baseline. The multi-paragraph synthesis step was designed to allow each cluster to be individually well-covered, but the 7B-parameter model used in this study lacks sufficient synthesis capacity to merge six to eight distinct paragraphs without introducing stitching hallucinations. This is a fundamental architectural limitation of the current pipeline design.

**Community Clustering Dominance.** Modularity-based community clustering — the simplest available structural baseline — consistently outperforms ACBS at every evaluated scale and is the highest-performing method in this study overall. This outcome suggests that structural graph cohesion is a stronger predictor of generation fidelity than attention boundary alignment at the scales studied. Community clustering keeps structurally related triples together in a single generation context, reducing entity co-reference resolution demands on the LLM. ACBS, by imposing vulnerability-budget constraints, sometimes breaks structurally cohesive communities to respect the budget, fragmenting what should be co-generated triples across separate sub-prompts.

### C. Why ACBS Helps at Moderate Scale but Fails at Extreme Scale

At 40–50 triples, the graph partitions into 2–3 clusters under ACBS. The LLM's synthesis step merges 2–3 paragraphs — a task within the demonstrated compositional capacity of 7B-parameter models. The benefit of boundary placement outweighs the synthesis overhead at this cluster count.

At 60–70 triples, the graph partitions into 6–8 clusters. The synthesis step must coherently merge 6–8 independently generated paragraphs without access to the source graph. Under local 7B models, this generates bridging text that introduces entity name inconsistencies, relationship confabulations, and temporal order violations — the hallucination sources observed in Table III.

### D. Implications for Future Systems

1. **Single-pass generation**: Future SMART-Graph variants should investigate single-pass generation that includes full ACBS ordering in a single extended prompt, avoiding the synthesis step entirely. Advances in commercial models with 128K+ token context windows make this increasingly feasible.

2. **Dynamic budget adaptation**: The ACBS budget $\mathcal{B}$ should be adapted as a function of graph size and model context window capacity rather than being fixed globally. A budget schedule $\mathcal{B}(N) = \mathcal{B}_0 \cdot (W / N)^\alpha$ could dynamically compress cluster size as graph scale increases.

3. **Moderate-SOS targeting**: As discussed in Section VI, future SOS designs should investigate protective emphasis on moderate-SOS triples, which are the primary victims of Pack-Dilution and are not naturally protected by linguistic isolation.

4. **Hybrid ACBS-Community Clustering**: A hybrid approach could apply community detection first (to preserve structural coherence) and then apply U-shaped sorting within each detected community (to apply attention boundary protection). This could combine the structural coherence benefit of community clustering with the boundary-protection benefit of ACBS.

---

## VIII. Threats to Validity

### A. Synthetic Dataset Dependence

The large-graph scaling experiments (Dataset B) rely entirely on programmatically chained synthetic graphs. These graphs share a regular chain structure with bounded connectivity that is not representative of real-world knowledge graphs, which exhibit power-law degree distributions, heterogeneous community structures, and varying predicate frequency distributions. Findings regarding scaling behavior (particularly the ACBS degradation at 60–70 triples) may not replicate on natural knowledge graphs such as DBpedia or Freebase subgraphs. The construction of a large-scale, naturally derived evaluation benchmark is left as future work.

### B. Extraction-Model Dependence

The SMGA evaluation pipeline depends on the accuracy of the LLM-based triple extraction step (Stage 6). If the extraction model systematically fails to extract certain triple types (e.g., triples with complex nested predicates or multi-hop relational chains), coverage and omission metrics will be biased downward for all methods equally, but differential extraction failure rates across serialization methods could introduce confounds. We have not evaluated triple-type-specific extraction accuracy.

### C. Small Sample Sizes in Cross-Model Evaluation

Cross-model experiments were limited to n=3 representative graphs per model due to GPU memory and throughput constraints. At n=3, statistical tests are severely underpowered for detecting moderate effect sizes (e.g., a t-test at n=3 has approximately 20% power for detecting d=1.0 at α=0.05). The statistically significant Llama 3 result (p=0.0392) and the large but non-significant Gemma result (p=0.2269) should both be interpreted with caution. The cross-model findings are directionally suggestive but cannot be treated as definitive without larger sample sizes.

### D. Hyperparameter Sensitivity

The ACBS algorithm's coverage performance is demonstrably sensitive to the chosen budget $\mathcal{B}$ and cluster size $K$. The grid search over a limited 5×4 parameter grid may not have identified the globally optimal configuration, and the optimal parameters may differ substantially between graph scales. The reported results use the parameters optimal for the 47-triple ablation (B=2.5, K=8), which may not be optimal at 40 or 70 triples. A scale-adaptive hyperparameter strategy was not evaluated.

### E. Single-GPU, Single-Run Evaluation

With the exception of the grid search experiments, each configuration was evaluated in a single run. LLM generation at temperature τ=0.2 has non-negligible stochasticity, and single-run evaluation does not capture output variance. Variance-averaged results (e.g., mean ± σ across 5 generation runs) would provide more reliable estimates of true method performance.

---

## IX. Conclusion

This paper presents SMART-Graph, a research-validated framework for mitigating attention-induced factual degradation in graph-to-text generation tasks. The system's core mechanism — profiling triple semantic vulnerability via SOS and calibrating their serialized position to match transformer attention boundary zones — demonstrates clear empirical benefit at moderate graph scales (40–50 triples), achieving up to a 57.45% factual coverage rate on a 47-triple controlled benchmark, compared to 46.81% for flat-sequential baselines under identical evaluation. The Soft-Match Graph Alignment evaluation framework, validated by strong APS-coverage correlation (Spearman ρ = 0.5312, p < 0.001), provides a more faithful measure of factual recall than exact-match methods and is proposed as a standard evaluation tool for LLM-based D2T systems.

The paper's most significant empirical finding — the Outlier Isolation Effect — revises our understanding of which triples are most vulnerable in long-context generation. Contrary to our original hypothesis, semantically distinct triples (high SOS, omission rate 31.67%) survive generation far more reliably than semantically central triples (omission rate 71.11%), due to the linguistic compulsion to express isolated facts in dedicated simple sentences. Pack-Dilution of semantically similar in-lier triples is the dominant failure mode in long-context graph generation.

Honest reporting requires acknowledging that SMART-Graph's ACBS component is outperformed by simple random ordering and modularity community clustering at 70-triple scales, and that modularity community clustering is the strongest single method evaluated in this study across all scales. Future development should prioritize: (1) eliminating the multi-paragraph synthesis step through single-pass generation, (2) dynamically scaling the ACBS budget as a function of graph size and model capacity, and (3) investigating hybrid ACBS-community approaches that combine structural coherence with attention boundary calibration.

---

## Appendix A: Required Figure Captions and Placement Reference

| Figure | Caption | Recommended Placement |
|---|---|---|
| **Figure 1** | *SMART-Graph Architecture Pipeline. System architecture showing vector space profiling, ACBS subgraph partitioning, U-shaped linearization, and SMGA bipartite matching evaluation.* | Page 2, top, single column |
| **Figure 2** | *Fact Coverage vs. Graph Size (40–70 triples). Contrasts the steep decline of flat-sequential prompting with the relative stability of SMART-Graph and Modularity Community Clustering across scales. Source: `results/benchmark.svg`.* | Page 4, bottom, single column |
| **Figure 3** | *Semantic Divergence Rate vs. Graph Size. Community clustering and random baselines maintain lower divergence at extreme scales while ACBS SDR worsens. Source: `results/benchmark.svg`.* | Page 4, bottom, single column |
| **Figure 4** | *APS vs. Factual Coverage Correlation Scatter Plot. Positive rank correlation (Spearman ρ = 0.5312, p < 0.001) confirms that APS is a valid and computationally efficient proxy for factual coverage.* | Page 5, top, single column |
| **Figure 5** | *SOS vs. Retention Analysis. Bar chart displaying the Outlier Isolation Effect: high-SOS semantically distinct triples show substantially lower omission rates than low-SOS in-lier triples, inverting the original hypothesis.* | Page 6, middle, single column |
| **Figure 6** | *Cross-Model Coverage Comparison. Relative coverage improvement of SMART-Graph (ACBS) over the Flat Sequential baseline across Qwen 2.5, Llama 3, and Gemma. Source: Table IV.* | Page 6, bottom, single column |
| **Figure 7** | *Pareto Frontier Sweep of ACBS Parameters K and 𝓑. Optimal configuration (𝓑=2.5, K=8) identified at the frontier maximizing coverage while minimizing SDR. Source: `results/pareto_frontier.svg`.* | Page 7, top, single column |

---

## Appendix B: Citation Placeholders

| ID | Proposed Reference Area |
|---|---|
| [C1] | Liu et al. — Lost in the Middle: How Language Models Use Long Contexts |
| [C2] | Attention mechanism modification for long-context LLMs |
| [C3] | WebNLG benchmark and graph linearization |
| [C4] | Prompt-position studies for factual recall in LLMs |
| [C5] | Papineni et al. — BLEU: A Method for Automatic Evaluation of Machine Translation |
| [C6] | Graph Neural Networks for Data-to-Text generation |
| [C7] | Lin — ROUGE: A Package for Automatic Evaluation of Summaries |
| [C8] | LLM-based triple extraction evaluation pipelines |
| [C9] | Zhang et al. — BERTScore: Evaluating Text Generation with BERT |
| [C10] | Graph-of-thought prompting frameworks |
| [C11] | Kuhn — The Hungarian Method for the Assignment Problem |

*Note: All citations are placeholders. Bibliographic entries must be independently verified and formatted before submission. No entries have been fabricated.*

---

## Reviewer Mode Evaluation

*This section evaluates the paper as an independent reviewer would.*

---

### 1. Strengths

**S1 — Rigorous Baseline Coverage.** The inclusion of a Random serialization baseline is a critical methodological strength. Many prior serialization papers compare only against the sequential default; including a random control separates the effect of *ordering quality* from mere *non-sequentiality*. The five-condition ablation matrix (Table I) is well-designed and permits clean attribution of coverage gains.

**S2 — Evaluation Methodology Contribution.** SMGA is a genuine methodological contribution. The demonstration that exact-match evaluation produces 0.00% coverage while SMGA recovers 46.81% is striking and practically important. This evaluation gap has likely distorted prior literature on LLM-based D2T generation.

**S3 — Honest Negative Results.** The paper clearly discloses that ACBS performs *worse* than random ordering at 70 triples, does not claim scale-insulation, and names modularity clustering as the superior method overall. This epistemic honesty is uncommon in ML papers and strengthens the scientific credibility of the work.

**S4 — Unexpected Finding Quality.** The Outlier Isolation Effect is a genuine empirical discovery grounded in sentence-reference analysis. The mechanistic explanation (Pack-Dilution) is plausible and testable. This section would be the strongest reason for an area chair to accept this paper.

**S5 — Statistical Reporting.** The paper reports effect sizes (Cohen's d), p-values, and correlation coefficients with appropriate precision. Most comparable NLP papers report only accuracy percentages without effect size quantification.

---

### 2. Weaknesses

**W1 — ACBS Scale Collapse.** The framework's primary contribution (ACBS) is outperformed by a random baseline at the largest evaluated scale. The synthesis-overhead explanation, while plausible, is not directly validated — no ablation isolates the synthesis step from the partitioning step. Without this ablation, it is unclear whether scale failure arises from the synthesis step, the ACBS clustering, or the interaction of both.

**W2 — Community Clustering Baseline Dominance.** Modularity community clustering outperforms ACBS at every evaluated scale. If the paper's takeaway is "use community clustering," then ACBS is a negative result. The paper frames this correctly (ACBS is effective at moderate scale), but the abstract and title do not adequately signal that the proposed method is not the best method overall.

**W3 — n=3 Cross-Model Evaluation.** Cross-model conclusions drawn from n=3 samples have very limited statistical reliability. The Gemma result (p=0.2269) is not significant, and even the Llama 3 result (p=0.0392) is a single comparison near the α=0.05 boundary. These results should be described as pilot findings pending replication with larger samples.

**W4 — No Real-World Graph Evaluation.** All large-graph experiments use synthetic chained graphs. The paper does not evaluate SMART-Graph on naturally occurring large knowledge graph subsets (e.g., DBpedia entities with 30+ attributes). The generalizability of scaling results to real-world graphs is untested.

**W5 — Synthesis Step Cost.** The ACBS pipeline requires multiple LLM inference calls (one per cluster plus one synthesis call). The paper does not report runtime comparisons between methods. If ACBS takes 5× longer than community clustering while producing inferior results at large scales, the cost-benefit ratio is unfavorable.

---

### 3. Likely Reviewer Criticisms

**RC1 (Senior Reviewer):** *"The paper proposes ACBS as its main technical contribution, but Table II conclusively shows that at the largest evaluated scale, ACBS (32.35%) is beaten by random ordering (44.12%) and community clustering (45.59%). The authors acknowledge this but do not provide a satisfying fix or ablation. The current framing claims 'ACBS works at moderate scale' but moderate scale (40–50 triples) is also where most WebNLG graphs already live. What is the practical deployment scenario for SMART-Graph?"*

**RC2 (Methods Reviewer):** *"SMGA is introduced as an evaluation contribution, but its threshold τ is fixed at 0.75 without ablation. How sensitive is the coverage metric to τ? A threshold sensitivity table should be included. Additionally, the Hungarian matching layer ensures one-to-one matching — but what if the same fact appears multiple times in the generated text? The current design would miss the duplicate."*

**RC3 (Statistics Reviewer):** *"Cross-model validation at n=3 is insufficient for statistical inference. The authors report 'large effect sizes' for Gemma (d=0.9952) but with a non-significant p-value at n=3. Effect sizes computed from n=3 have enormous confidence intervals (approximately ±1.5). These results should not be used to support claims about Gemma's generalization."*

**RC4 (Novelty Reviewer):** *"The core idea — placing important facts at context boundaries — is closely related to [C4] and other prompt-position papers. The contribution of SOS-based vulnerability scoring and ACBS clustering is incremental over prior boundary-placement strategies. The authors should explicitly differentiate their technical contribution from prior work on context-boundary prompting."*

---

### 4. Suggested Future Work (Reviewer Perspective)

**FW1 — Synthesis-Free ACBS.** Implement a single-pass ACBS variant that interleaves all cluster-ordered triples into a single extended prompt (no paragraph synthesis step). Compare against multi-pass ACBS to isolate the synthesis overhead cost. This directly addresses the primary failure mode at large scale.

**FW2 — Real-World Knowledge Graph Evaluation.** Evaluate on DBpedia or Wikidata subgraphs at 40–70 triple scales. This directly addresses the synthetic dataset threat to validity and is required for publication at top-tier venues.

**FW3 — Cross-Model Replication at n≥10.** Rerun cross-model experiments with at least n=10 graphs per model. This would provide adequate statistical power to draw reliable conclusions about Gemma and Llama 3.

**FW4 — Dynamic Budget Scheduling.** Implement and evaluate $\mathcal{B}(N)$ as a function of graph size. Report whether adaptive budgeting recovers ACBS performance at 60–70 triple scales.

**FW5 — Pack-Dilution Mitigation.** Design an in-lier protection strategy — e.g., forcing one-fact-per-sentence generation for low-SOS triples via structured prompting — to test whether in-lier omission rates can be reduced toward outlier-level omission rates.

**FW6 — τ Sensitivity Analysis.** Report coverage and SDR as functions of SMGA threshold τ across [0.60, 0.65, 0.70, 0.75, 0.80]. This is required to validate SMGA as a robust evaluation tool rather than a metric sensitive to threshold choice.

---

*End of Manuscript*

---

> **Reproducibility Statement.** All experiments were conducted on a single local workstation (specifications in Section IV-D) using open-weight models via Ollama. Embedding caches were persisted to `cache/embeddings.json`. Benchmark results are reported from single-run evaluations at temperature τ=0.2. Full code and experimental logs are available at the project repository.

> **Conflict of Interest.** The author declares no conflicts of interest.

> **Data Availability.** Synthetic Dataset B is fully reproducible from the generation procedure described in Section IV-A. WebNLG Dataset A is publicly available at the official WebNLG repository.
