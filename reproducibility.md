# SMART-Graph Reproducibility Package

This document outlines the steps required to replicate the experimental results and run the interactive dashboard for the SMART-Graph project.

## 1. Prerequisites & Environment Setup

### 1.1 Local LLM Environment (Ollama)
Ensure you have **Ollama** installed on your host machine.
1. Download and install Ollama from [ollama.com](https://ollama.com).
2. Start the Ollama local daemon.
3. Pull the required models by executing the following commands in your shell:
   ```bash
   ollama pull nomic-embed-text:latest
   ollama pull qwen2.5:7b
   ollama pull llama3:latest
   ollama pull gemma:latest
   ```

### 1.2 Python Environment
Create a clean virtual environment and install the required dependencies:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Running the Benchmarks

All benchmark results can be reproduced by running the master script, which executes the entire pipeline sequentially:

```bash
python run_all_experiments.py
```

This master script runs the following steps (which can also be executed individually):

1. **SOS Correlation Study**:
   ```bash
   python experiments/sos_correlation.py
   ```
   Validates hypothesis H1 by correlating Semantic Outlier Scores (SOS) with omission frequencies in flat prompting layouts.

2. **Large Graph Scaling Study**:
   ```bash
   python experiments/large_graph_study.py
   ```
   Tests boundaries of factual recall at scales of 40, 50, 60, and 70 triples.

3. **Cross-Model Generality Validation**:
   ```bash
   python experiments/cross_model_study.py
   ```
   Benchmarks Qwen, Llama 3, and Gemma on SDR and Fact Coverage metrics.

4. **Sentence-Isolation Study**:
   ```bash
   python experiments/sentence_isolation_study.py
   ```
   Analyzes the Outlier Isolation vs. Pack-Dilution phenomenon.

5. **Grid Search Sweep**:
   ```bash
   python experiments/grid_search.py
   ```
   Sweeps cluster limit $K$ and vulnerability budget $\mathcal{B}$ to identify the Pareto frontier of optimal settings.

6. **LaTeX Table Compilation**:
   ```bash
   python utils/compile_latex.py
   ```
   Processes JSON outputs from all studies and compiles them into clean LaTeX tables.

All outputs (data files, Pareto plots, LaTeX tables) will be written to the `results/` directory.

---

## 3. Interactive Web Dashboard

To run the interactive web application, execute:

```bash
python app.py
```

Once running, navigate your web browser to:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

Using the UI, you can:
- Load synthetic large graphs or select native WebNLG graphs.
- Configure active hyperparameters ($K$, $\mathcal{B}$).
- Trigger live graph serializations and view the resulting soft-matched alignments.
