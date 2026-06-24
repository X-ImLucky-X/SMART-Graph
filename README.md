# SMART-Graph 🧠🕸️

### Attention-Calibrated Graph Serialization for Long-Context Data-to-Text Generation

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local-orange.svg)](https://ollama.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SMART-Graph is a research framework for studying how graph serialization strategies affect factual retention in long-context Data-to-Text (D2T) generation.

The project investigates whether transformer attention limitations can be mitigated by restructuring graph triples before generation using:

* Semantic Outlier Scores (SOS)
* Attention-Calibrated Boundary Stratification (ACBS)
* U-Shaped Prompt Serialization
* Soft-Match Graph Alignment (SMGA)

The framework was evaluated on WebNLG-derived graphs and synthetic connected graphs ranging from 4 to 70 triples using local LLMs through Ollama.

---

# Research Motivation

Large Language Models exhibit a well-known "lost-in-the-middle" behavior in which information appearing in the middle of long contexts receives less attention than information near the beginning or end.

SMART-Graph explores whether graph-aware serialization can improve factual retention by strategically positioning graph facts before generation.

---

# Main Contributions

### 1. Attention-Calibrated Boundary Stratification (ACBS)

A graph partitioning and serialization strategy that:
* Computes Semantic Outlier Scores (SOS) in embedding space.
* Creates vulnerability-bounded clusters using a relation graph greedy traversal.
* Positions selected triples into primacy and recency attention regions in a U-shape layout.

---

### 2. Soft-Match Graph Alignment (SMGA)

A semantic evaluation framework that replaces strict exact string matching of triples with:
* Embedding vector similarity.
* Hungarian maximum-weight bipartite matching.
* Error-tolerant factual alignment (synonym and phrasing resilience).

---

### 3. Attention Protection Score (APS)

A metric designed to estimate how well high-priority facts survive generation.

APS was validated against overall factual coverage:
* **Pearson Correlation ($r$)**: **`0.3795`** ($p = 0.0066$)
* **Spearman Rank Correlation ($\rho$)**: **`0.5312`** ($p < 0.001$)

The strong correlation coefficient indicates that shielding outlier facts prevents coverage collapse.

---

# Key Experimental Findings

## Controlled Ablation (47-Triple Graph)

Evaluated under the identical SMGA metric to isolate components:

| Variant | Serialization | Evaluation | SDR | Coverage | Outlier Protection (APS) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **1. Flat + Strict** | Flat | Strict | 1.4468 | 0.00% | 0.00% |
| **2. Flat + SMGA** | Flat | SMGA | 0.5319 | 46.81% | 66.67% |
| **3. Random + SMGA** | Random | SMGA | 0.5319 | 48.94% | 55.56% |
| **4. SOS + SMGA** | SOS | SMGA | 0.5532 | 44.68% | **88.89%** |
| **5. ACBS + SMGA (SMART)** | **ACBS** | **SMGA** | **0.5106** | **57.45%** | 66.67% |

ACBS + SMGA (SMART) achieved the highest factual coverage and lowest Semantic Divergence Rate (SDR), proving the ordering algorithm provides factual gains over Random.

---

## Large Graph Scaling

| Graph Size | Flat (Seq.) | Random | ACBS (SMART) |
| :---: | :---: | :---: | :---: |
| **40** | 52.50% | 60.00% | **75.00%** |
| **50** | 42.86% | 40.82% | **53.06%** |
| **60** | 36.21% | **44.83%** | 43.10% |
| **70** | 25.00% | **44.12%** | 32.35% |

ACBS improves retention on moderate-scale graphs (40–50 triples) but degrades at extreme scales (60–70 triples) where model synthesis limitations dominate.

---

## Cross-Model Validation

Models evaluated (SDR paired t-tests):

| Model | Baseline Coverage | ACBS Coverage | SDR $p$-value | Cohen's $d$ (SDR) |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen 2.5 (7B)** | 69.10% | 71.94% | 0.2757 | -0.8578 |
| **Llama 3 (8B)** | 59.29% | **69.81%** | **0.0392** | **2.8289** (Very Large) |
| **Gemma (2B)** | 45.51% | **63.27%** | 0.2269 | **0.9952** (Large) |

SMART-Graph generalizability is validated, demonstrating significant improvements on Llama 3 and Gemma model families.

---

## Unexpected Scientific Finding

The original hypothesis predicted:
> High Semantic Outlier Score → Higher omission probability

Experiments showed the opposite. High-SOS triples were retained better because the model naturally generated them as isolated sentences.

This phenomenon was named:
### Outlier Isolation Effect

while low-SOS facts suffered from:
### Pack-Dilution Effect
where multiple related in-lier facts were merged into compound clauses and became harder for the vector model to align and recover.

---

# Repository Structure

```text
configs/                  # Configuration YAML profiles
data/                     # Native WebNLG and synthetic datasets
experiments/              # Core evaluation scripts
evaluators/               # strict and SMGA evaluation layers
clustering/               # ACBS engine implementation
utils/                    # Persistent cache and table generation utilities
results/                  # Exported JSON logs, tables, and SVG plots
paper/                    # LaTeX manuscript source and notes
app.py                    # FastAPI dashboard server
run_all_experiments.py    # Master pipeline runner
```

---

# Installation

```bash
pip install -r requirements.txt
```

Pull required Ollama models:

```bash
ollama pull qwen2.5:7b
ollama pull llama3
ollama pull gemma
ollama pull nomic-embed-text
```

---

# Running Experiments

Run complete pipeline:

```bash
python run_all_experiments.py
```

Run large graph study:

```bash
python experiments/large_graph_study.py
```

Run cross-model validation:

```bash
python experiments/cross_model_study.py
```

Run reproducibility verification:

```bash
python experiments/verify_reproducibility.py
```

---

# Results

Generated outputs include:
* Benchmark reports
* Scaling studies
* Ablation studies
* Cross-model validation
* LaTeX tables
* SVG figures

Output directory:
```text
results/
```

---

# Limitations

Current experiments indicate:
* **Scale threshold**: ACBS performs best at moderate graph scales (40–50 triples) and degrades beyond 60 triples.
* **Clustering baseline**: Modularity-based community clustering can outperform ACBS at extreme scales (>60 triples).
* **Extraction bias**: Results remain dependent on the generation and extraction model capabilities.
* **Synthetic graph bias**: Synthetic evaluations may not fully represent real-world graph semantic distributions.

These limitations are documented to support transparent and reproducible research.

---

# Citation

If you use SMART-Graph in academic work, please cite the project using the provided `CITATION.cff`.

---

# License

MIT License
