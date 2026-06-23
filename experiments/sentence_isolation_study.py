import os
import sys
import json
import re
import yaml
import numpy as np
import ollama
from scipy.stats import pearsonr, spearmanr

# Path injection
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.vector_engine import VectorSpaceProfiler

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def run_sentence_isolation_study():
    print("=== Starting Sentence Isolation Study ===")
    
    # 1. Load configuration
    with open(os.path.join("configs", "baseline.yaml"), "r") as f:
        config = yaml.safe_load(f)
        
    model = config.get("model", "qwen2.5:7b")
    embedding_model = config.get("embedding_model", "nomic-embed-text:latest")
    base_seed = config.get("seed", 42)
    sim_threshold = config.get("similarity_threshold", 0.75)
    
    # Run 5 trials to keep local runtime fast but statistically representative
    trials = 5
    
    # 2. Load dataset
    with open(os.path.join("data", "generated_large", "synthetic_large.json"), "r") as f:
        large_data = json.load(f)
    triples = large_data[0]["triples"]
    
    # 3. Calculate SOS scores
    profiler = VectorSpaceProfiler(model_name=embedding_model)
    sos_scores = profiler.calculate_sos(triples)
    
    # 4. Generate texts (flat sequential)
    flat_prompt = (
        "Write a fluent, cohesive, and comprehensive text that incorporates ALL of the following facts "
        "without omitting any details. Make sure to describe every single fact explicitly:\n\n" +
        "\n".join(f"- {s} {p} {o}" for s, p, o in triples)
    )
    
    print(f"Generating {trials} texts via Ollama '{model}'...")
    generated_texts = []
    for i in range(trials):
        print(f"  Trial {i+1}/{trials}...")
        try:
            response = ollama.generate(
                model=model,
                prompt=flat_prompt,
                options={"temperature": 0.7, "seed": base_seed + i, "num_ctx": 4096}
            )
            generated_texts.append(response["response"])
        except Exception as e:
            print(f"Error during Ollama generation: {e}")
            return

    # 5. Track Sentence Reference Counts
    triple_texts = [f"{s} {p} {o}" for s, p, o in triples]
    triple_embeddings = [profiler.get_embedding(t) for t in triple_texts]
    
    # Matrix of shape (len(triples), trials) to store reference counts
    ref_counts = np.zeros((len(triples), trials))
    
    print("Analyzing sentence references...")
    for t_idx, text in enumerate(generated_texts):
        sentences = split_sentences(text)
        if not sentences:
            continue
            
        sentence_embeddings = [profiler.get_embedding(s) for s in sentences]
        
        for tr_idx, tr_emb in enumerate(triple_embeddings):
            tr_norm = np.linalg.norm(tr_emb)
            ref_count = 0
            
            for s_emb in sentence_embeddings:
                s_norm = np.linalg.norm(s_emb)
                if tr_norm > 0 and s_norm > 0:
                    sim = np.dot(tr_emb, s_emb) / (tr_norm * s_norm)
                    if sim >= sim_threshold:
                        ref_count += 1
            ref_counts[tr_idx, t_idx] = ref_count
            
    # Calculate average reference counts for each triple
    avg_ref_counts = np.mean(ref_counts, axis=1)
    
    # 6. Calculate Correlations
    p_corr, p_val = pearsonr(sos_scores, avg_ref_counts)
    s_corr, s_val = spearmanr(sos_scores, avg_ref_counts)
    
    # Analyze outliers vs in-liers
    sorted_indices = np.argsort(sos_scores)
    k_20 = max(1, len(triples) // 5)
    bottom_20_idx = sorted_indices[:k_20]
    top_20_idx = sorted_indices[-k_20:]
    
    avg_ref_top = np.mean(avg_ref_counts[top_20_idx])
    avg_ref_bottom = np.mean(avg_ref_counts[bottom_20_idx])
    
    # Percent of trials where the triple has exactly 1 reference
    exact_one_rate = np.mean(ref_counts == 1, axis=1)
    avg_exact_one_top = np.mean(exact_one_rate[top_20_idx])
    avg_exact_one_bottom = np.mean(exact_one_rate[bottom_20_idx])
    
    print("\n=== Sentence Isolation Study Results ===")
    print(f"Pearson Correlation (SOS vs. Reference Count): {p_corr:.4f} (p-value: {p_val:.4e})")
    print(f"Spearman Correlation (SOS vs. Reference Count): {s_corr:.4f} (p-value: {s_val:.4e})")
    print(f"Avg Sentence References for Top 20% Outlier Triples (High SOS): {avg_ref_top:.2f}")
    print(f"Avg Sentence References for Bottom 20% In-lier Triples (Low SOS): {avg_ref_bottom:.2f}")
    print(f"Exact-1 Sentence Reference Probability (Outliers): {avg_exact_one_top*100:.2f}%")
    print(f"Exact-1 Sentence Reference Probability (In-liers): {avg_exact_one_bottom*100:.2f}%")
    
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    output_data = {
        "pearson_r": float(p_corr),
        "pearson_p": float(p_val),
        "spearman_rho": float(s_corr),
        "spearman_p": float(s_val),
        "avg_references_high_sos": float(avg_ref_top),
        "avg_references_low_sos": float(avg_ref_bottom),
        "exact_one_rate_high_sos": float(avg_exact_one_top),
        "exact_one_rate_low_sos": float(avg_exact_one_bottom),
        "triples_analyzed": [
            {
                "triple": triples[i],
                "sos": sos_scores[i],
                "avg_references": float(avg_ref_counts[i]),
                "exact_one_probability": float(exact_one_rate[i])
            }
            for i in range(len(triples))
        ]
    }
    
    with open(os.path.join(results_dir, "sentence_isolation.json"), "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Results exported to {os.path.join(results_dir, 'sentence_isolation.json')}")

if __name__ == "__main__":
    run_sentence_isolation_study()
