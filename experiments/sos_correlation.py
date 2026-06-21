import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import re
import random
import yaml
import numpy as np
import ollama
from scipy.stats import pearsonr, spearmanr
from utils.vector_engine import VectorSpaceProfiler
from evaluators.smga import SMGAEvaluator
from experiments.threshold_sweep import extract_triples_from_text

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

def split_sentences(text):
    # Splits text into sentences based on punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def run_correlation_study():
    print("=== Starting SOS Correlation Study ===")
    
    # 1. Load configuration
    with open(os.path.join("configs", "baseline.yaml"), "r") as f:
        config = yaml.safe_load(f)
        
    model = config.get("model", "qwen2.5:7b")
    embedding_model = config.get("embedding_model", "nomic-embed-text:latest")
    base_seed = config.get("seed", 42)
    runs = config.get("correlation_runs", 20)
    sim_threshold = config.get("similarity_threshold", 0.75)
    
    set_seeds(base_seed)
    
    # 2. Load dataset
    with open(os.path.join("data", "generated_large", "synthetic_large.json"), "r") as f:
        large_data = json.load(f)
    
    triples = large_data[0]["triples"]
    print(f"Loaded synthetic large graph with {len(triples)} triples.")
    
    # 3. Calculate SOS scores
    profiler = VectorSpaceProfiler(model_name=embedding_model)
    print("Calculating Semantic Outlier Scores (SOS) for triples...")
    sos_scores = profiler.calculate_sos(triples)
    
    # 4. Generate texts using flat sequential prompting (with varying seeds for diversity)
    flat_prompt = (
        "Write a fluent, cohesive, and comprehensive text that incorporates ALL of the following facts "
        "without omitting any details. Make sure to describe every single fact explicitly:\n\n" +
        "\n".join(f"- {s} {p} {o}" for s, p, o in triples)
    )
    
    print(f"Running {runs} generation trials via Ollama '{model}'...")
    generated_texts = []
    
    for i in range(runs):
        print(f"Trial {i+1}/{runs}...")
        try:
            # Vary seed per run for diverse outputs, but keep it reproducible
            run_seed = base_seed + i
            response = ollama.generate(
                model=model,
                prompt=flat_prompt,
                options={
                    "temperature": 0.7,
                    "seed": run_seed,
                    "num_ctx": 4096
                }
            )
            generated_texts.append(response["response"])
        except Exception as e:
            print(f"Error during Ollama generation: {e}")
            return
            
    # 5. Determine Omission Rates using extraction + SMGA Evaluator
    smga_evaluator = SMGAEvaluator(embedding_model=embedding_model, similarity_threshold=sim_threshold)
    omission_matrix = np.zeros((len(triples), runs))  # 1 for omitted, 0 for retained
    
    print("Analyzing omissions across trials using triple extraction and SMGA Bipartite Alignment...")
    for trial_idx, text in enumerate(generated_texts):
        print(f"Extracting and aligning for trial {trial_idx+1}/{runs}...")
        extracted = extract_triples_from_text(text, model)
        eval_res = smga_evaluator.evaluate(triples, extracted)
        
        omitted_normalized = {tuple(str(item).strip().lower() for item in t) for t in eval_res["omitted_triples"]}
        
        for triple_idx, t in enumerate(triples):
            t_norm = tuple(str(item).strip().lower() for item in t)
            if t_norm in omitted_normalized:
                omission_matrix[triple_idx, trial_idx] = 1
                
    omission_rates = np.mean(omission_matrix, axis=1)
    
    # 6. Calculate Pearson and Spearman Correlation
    p_corr, p_val = pearsonr(sos_scores, omission_rates)
    s_corr, s_val = spearmanr(sos_scores, omission_rates)
    
    print("\n=== Study Results ===")
    print(f"Pearson Correlation (r): {p_corr:.4f} (p-value: {p_val:.4e})")
    print(f"Spearman Correlation (rho): {s_corr:.4f} (p-value: {s_val:.4e})")
    
    # Calculate average omission rate of top 20% outlier triples vs bottom 20%
    sorted_indices = np.argsort(sos_scores)
    k_20 = max(1, len(triples) // 5)
    bottom_20_idx = sorted_indices[:k_20]
    top_20_idx = sorted_indices[-k_20:]
    
    avg_omission_top = np.mean(omission_rates[top_20_idx])
    avg_omission_bottom = np.mean(omission_rates[bottom_20_idx])
    print(f"Avg Omission Rate for Top 20% Outlier Triples (High SOS): {avg_omission_top*100:.2f}%")
    print(f"Avg Omission Rate for Bottom 20% In-lier Triples (Low SOS): {avg_omission_bottom*100:.2f}%")
    
    # 7. Check Kill Criteria
    hypothesis_supported = True
    kill_triggered = False
    
    if p_corr < 0.2 and s_corr < 0.2:
        print("\n[WARNING] early kill criterion triggered: SOS-omission correlation is below 0.2!")
        print("The scientific foundation of the outlier score hypothesis is shaky.")
        hypothesis_supported = False
        kill_triggered = True
        
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    output_data = {
        "pearson_r": float(p_corr),
        "pearson_p": float(p_val),
        "spearman_rho": float(s_corr),
        "spearman_p": float(s_val),
        "avg_omission_high_sos": float(avg_omission_top),
        "avg_omission_low_sos": float(avg_omission_bottom),
        "hypothesis_supported": hypothesis_supported,
        "kill_triggered": kill_triggered,
        "triples_analyzed": [
            {
                "triple": triples[i],
                "sos": sos_scores[i],
                "omission_rate": float(omission_rates[i])
            }
            for i in range(len(triples))
        ]
    }
    
    with open(os.path.join(results_dir, "sos_correlation.json"), "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print(f"Results exported to {os.path.join(results_dir, 'sos_correlation.json')}")

if __name__ == "__main__":
    run_correlation_study()
