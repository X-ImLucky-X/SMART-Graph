# SMART-Graph: Attention-Calibrated Graph Serialization for Reducing Semantic Divergence in Long-Context Data-to-Text Generation

SMART-Graph is a mathematically rigorous, local open-source framework designed to mitigate the "lost-in-the-middle" attention degradation in transformer-based Large Language Models (LLMs) during data-to-text generation. 

By decomposing large input graphs into semantically cohesive subgraphs and serializing them in a attention-calibrated V-shape (U-shape) context layout, the framework ensures vulnerable facts are placed in high-attention primacy and recency zones. 

---

## 1. Core Taxonomy & Innovations

The framework is organized under a unified taxonomy to ensure clean scientific evaluation:

```
SMART-Graph
├── ACBS (Attention-Calibrated Boundary Stratification)
│   ├── SOS Profiling (Semantic Outlier Scores)
│   ├── Vulnerability Budgeting (Aggregate cluster constraints)
│   └── U-Shaped Serialization (Boundary context mapping)
└── SMGA Evaluation (Soft-Match Graph Alignment)
    ├── Hungarian Bipartite Matching
    └── Attention Protection Score
```

### 1.1 Semantic Outlier Score (SOS) Profiling
To identify which triples in a global graph $G$ are most vulnerable to context omission, we compute their semantic distance from the global narrative centroid $\vec{C}$. Given a set of triples $T$, the centroid vector is:
$$\vec{C} = \frac{1}{|T|} \sum_{t_i \in T} \text{embed}(t_i)$$

The $SOS$ for any individual triple $t_i$ is its cosine distance from the centroid, scaled by a literal complexity penalty $\delta(t_i)$:
$$SOS(t_i) = \left( 1 - \cos(\vec{t}_i, \vec{C}) \right) \times \left(1 + \delta(t_i)\right)$$
$$\delta(t_i) = \begin{cases} 0.5 & \text{if } t_i \text{ contains digits or numeric dates} \\ 0 & \text{otherwise} \end{cases}$$

### 1.2 Attention-Calibrated Boundary Stratification (ACBS)
Triples are partitioned into subgraphs using a greedy, neighborhood-expanding traversal on the semantic relation graph (where nodes are triples sharing entities). Expansion is strictly bound by:
1. Max Cluster Size $K$ (e.g., $K=8$ triples).
2. Aggregate Vulnerability Budget $\mathcal{B}$:
   $$\sum_{t \in \text{cluster}} SOS(t) \le \mathcal{B}$$

Once a cluster is formed, triples are sorted by $SOS$ and distributed in a **U-shape** across the prompt window, mapping the highest-risk facts to the primacy and recency boundaries where attention remains near 100%:
$$\text{Sequence: } [t_{\text{highest}}, t_{\text{3rd\_highest}}, \dots, t_{\text{lowest}}, \dots, t_{\text{4th\_highest}}, t_{\text{2nd\_highest}}]$$

### 1.3 Soft-Match Graph Alignment (SMGA)
To evaluate fact retention without string matching sensitivities (such as "Apple Inc" vs "Apple"), we construct a weighted bipartite graph between input triples ($T_{\text{input}}$) and extracted triples ($T_{\text{extracted}}$). Edges represent cosine similarities of their nomic embeddings. We apply the **Hungarian Algorithm** (`linear_sum_assignment`) to find the optimal 1-to-1 matchings:
$$\max \sum_{(i,j) \in \text{Matches}} \cos(\vec{t}_i, \vec{t}_j) \quad \text{s.t. } \cos(\vec{t}_i, \vec{t}_j) \ge \tau$$
where $\tau$ is the soft-match threshold (default $0.75$).

---

## 2. Experimental Repository Structure

The codebase is organized as follows:

```
c:\ME\PROJECT\SMART-Graph\
├── configs\
│   ├── baseline.yaml               # Baseline configs (models, seeds, runs)
│   └── smart_graph.yaml            # Configs for ACBS + SMGA parameters
├── data\
│   ├── webnlg\                     # Native WebNLG task files (small scale)
│   └── generated_large\            # Programmatically chained synthetic large graphs (40-70 triples)
├── cache\
│   └── embeddings.json             # Persistent local embedding cache
├── research\
│   ├── hypotheses.md               # Formal H0/H1-H3 hypotheses
│   ├── contributions.md            # Documented research claims
│   └── risks.md                    # Risk register and early kill criteria
├── utils\
│   ├── hardware.py                 # System hardware logger
│   └── vector_engine.py            # Local embedding layer and SOS calculations
├── clustering\
│   └── acbs_engine.py              # ACBS engine (SOS, budgeting, U-shape sorting)
├── evaluators\
│   ├── strict.py                   # Strict set-difference evaluator
│   └── smga.py                     # Soft-Match Graph Alignment evaluator
├── experiments\
│   ├── runs\                       # Logs of individual experiment runs (run_XXX.json)
│   ├── ablation.py                 # Ablation study runner (Variants 1-4)
│   ├── threshold_sweep.py          # Parametric sweep of soft-match thresholds
│   └── sos_correlation.py          # Primary correlation study (SOS vs Omissions)
├── run_benchmark.py                # Comprehensive comparative benchmark suite
├── app.py                          # FastAPI backend service
├── static\
│   └── index.html                  # Dark glassmorphism dashboard UI
└── results\
    ├── sos_correlation.json        # Output of the correlation study
    ├── threshold_sweep.json        # Output of the threshold sweep
    ├── ablation_study.json         # Output of the ablation matrix
    ├── benchmark.csv               # Raw benchmark runs data
    ├── benchmark.svg               # SVG plots (SDR vs Scale, Attention Curve)
    └── report.md                   # Auto-generated LaTeX-ready research report
```

---

## 3. Running the Research Pipeline

### 3.1 Pre-requisites & Local Models
Ensure Ollama is running locally and has pulled the required models:
```powershell
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

Install the required mathematical and graph libraries:
```powershell
pip install numpy scipy networkx pyyaml uvicorn fastapi
```

### 3.2 Executing Experiments

1. **Initialize Datasets**:
   ```powershell
   python data/build_datasets.py
   ```

2. **Step 1: SOS Omission Correlation Study** ($H_1$ Validation):
   Runs flat sequential generation 20 times and tests if outlier scores correlate with omissions.
   ```powershell
   python experiments/sos_correlation.py
   ```
   *Result saved to*: [sos_correlation.json](file:///c:/ME/PROJECT/SMART-Graph/results/sos_correlation.json)

3. **Step 2: Threshold Sweeping**:
   Finds the optimal similarity matching threshold for SMGA.
   ```powershell
   python experiments/threshold_sweep.py
   ```
   *Result saved to*: [threshold_sweep.json](file:///c:/ME/PROJECT/SMART-Graph/results/threshold_sweep.json)

4. **Step 3: Ablation Study**:
   Compares Baseline vs. SOS Only vs. SOS + ACBS vs. Full SMART-Graph.
   ```powershell
   python experiments/ablation.py
   ```
   *Result saved to*: [ablation_study.json](file:///c:/ME/PROJECT/SMART-Graph/results/ablation_study.json)

5. **Step 4: Comparative Benchmarks**:
   Evaluates all baselines across graph scales, calculates paired t-tests, Cohen's $d$, and outputs charts.
   ```powershell
   python run_benchmark.py
   ```
   *Results saved to*:
   - Tabular dataset: [benchmark.csv](file:///c:/ME/PROJECT/SMART-Graph/results/benchmark.csv)
   - Visual plots: [benchmark.svg](file:///c:/ME/PROJECT/SMART-Graph/results/benchmark.svg)
   - LaTeX-ready report: [report.md](file:///c:/ME/PROJECT/SMART-Graph/results/report.md)

### 3.3 Launching the Dashboard
To start the interactive research server and view pipeline flows and metrics on a premium web interface:
```powershell
python app.py
```
Open your browser and navigate to `http://127.0.0.1:8000/`.