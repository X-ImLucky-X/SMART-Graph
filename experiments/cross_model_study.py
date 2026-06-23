import os
import sys
import json
import yaml
import time
import random
import numpy as np
from scipy.stats import ttest_rel
import ollama

# Path injection
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from clustering.acbs_engine import ACBSEngine
from evaluators.smga import SMGAEvaluator
from experiments.threshold_sweep import extract_triples_from_text

def run_cross_model_study():
    print("=== Starting SMART-Graph Cross-Model Validation ===")
    
    # 1. Load config
    with open(os.path.join("configs", "smart_graph.yaml"), "r") as f:
        config = yaml.safe_load(f)
        
    embedding_model = config.get("embedding_model", "nomic-embed-text:latest")
    seed = config.get("seed", 42)
    max_size = config.get("max_cluster_size", 8)
    budget = config.get("vulnerability_budget", 1.5)
    sim_threshold = config.get("similarity_threshold", 0.75)
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Load test graphs
    with open(os.path.join("data", "webnlg", "native_small.json"), "r") as f:
        small_tasks = json.load(f)
    astronaut_triples = small_tasks[0]["triples"]  # 7 triples
    
    # Combine first two small tasks to form medium scale
    medium_triples = small_tasks[0]["triples"] + small_tasks[1]["triples"]  # 13 triples
    
    # Load synthetic large graph
    with open(os.path.join("data", "generated_large", "synthetic_large.json"), "r") as f:
        large_tasks = json.load(f)
    large_triples = large_tasks[0]["triples"]  # 47 triples
    
    graphs = [
        {"name": "WebNLG_Astronaut", "scale": len(astronaut_triples), "triples": astronaut_triples},
        {"name": "WebNLG_Combined_0", "scale": len(medium_triples), "triples": medium_triples},
        {"name": "Synthetic_Chained_Large", "scale": len(large_triples), "triples": large_triples}
    ]
    
    models = ["qwen2.5:7b", "llama3:latest", "gemma:latest"]
    
    engine = ACBSEngine(max_size=max_size, max_vulnerability_budget=budget, embedding_model=embedding_model)
    smga_evaluator = SMGAEvaluator(embedding_model=embedding_model, similarity_threshold=sim_threshold)
    
    results = {}
    
    for model in models:
        print(f"\nEvaluating Model: {model}...")
        results[model] = []
        
        for g in graphs:
            graph_name = g["name"]
            triples = g["triples"]
            scale = g["scale"]
            print(f"  Graph: {graph_name} (Scale: {scale})...")
            
            sos_scores = engine.compute_sos(triples)
            
            for mode in ["baseline_a", "smart_graph"]:
                print(f"    Mode: {mode}...")
                start_time = time.time()
                clusters = engine.serialize_triples(triples, mode, seed=seed)
                
                # Generations
                paragraphs = []
                for idx, cluster in enumerate(clusters):
                    prompt = (
                        "Write a brief, fluent, and highly factual paragraph describing the following details:\n\n" +
                        "\n".join(f"- {s} {p} {o}" for s, p, o in cluster)
                    )
                    try:
                        response = ollama.generate(
                            model=model,
                            prompt=prompt,
                            options={"temperature": 0.2, "seed": seed, "num_ctx": 4096}
                        )
                        paragraphs.append(response["response"])
                    except Exception as e:
                        print(f"      Generation error: {e}")
                        paragraphs.append("")
                
                # Merge
                if len(paragraphs) > 1:
                    synthesis_prompt = (
                        "Synthesize the following paragraphs into a single, cohesive, fluent document. "
                        "Keep all the facts and details exactly as they are, but make the transitions smooth:\n\n" +
                        "\n\n".join(paragraphs)
                    )
                    try:
                        response = ollama.generate(
                            model=model,
                            prompt=synthesis_prompt,
                            options={"temperature": 0.2, "seed": seed, "num_ctx": 4096}
                        )
                        synthesized_text = response["response"]
                    except Exception:
                        synthesized_text = " ".join(paragraphs)
                else:
                    synthesized_text = paragraphs[0] if paragraphs else ""
                    
                runtime = time.time() - start_time
                
                # Extract & Evaluate
                extracted = extract_triples_from_text(synthesized_text, model)
                eval_res = smga_evaluator.evaluate(triples, extracted, sos_scores)
                
                results[model].append({
                    "graph_name": graph_name,
                    "scale": scale,
                    "mode": mode,
                    "sdr": eval_res["semantic_divergence_rate"],
                    "coverage": eval_res["coverage"],
                    "aps": eval_res["attention_protection_score"],
                    "omissions": eval_res["omissions_count"],
                    "hallucinations": eval_res["hallucinations_count"],
                    "runtime_seconds": round(runtime, 2)
                })
                print(f"      SDR: {eval_res['semantic_divergence_rate']:.4f}, Coverage: {eval_res['coverage']*100:.2f}%, APS: {eval_res['attention_protection_score']*100:.2f}%")
                
    # 5. Compute paired statistics per model
    model_stats = {}
    for model in models:
        run_recs = results[model]
        sdr_a = [r["sdr"] for r in run_recs if r["mode"] == "baseline_a"]
        sdr_smart = [r["sdr"] for r in run_recs if r["mode"] == "smart_graph"]
        
        t_stat, p_val = ttest_rel(sdr_a, sdr_smart)
        diff = np.array(sdr_a) - np.array(sdr_smart)
        std_diff = np.std(diff, ddof=1)
        cohens_d = np.mean(diff) / std_diff if std_diff > 0 else 0.0
        
        model_stats[model] = {
            "t_statistic": float(t_stat) if not np.isnan(t_stat) else 0.0,
            "p_value": float(p_val) if not np.isnan(p_val) else 1.0,
            "cohens_d": float(cohens_d),
            "sdr_baseline_mean": float(np.mean(sdr_a)),
            "sdr_smart_mean": float(np.mean(sdr_smart))
        }
        
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "cross_model_study.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_statistics": model_stats,
            "results": results
        }, f, indent=2, ensure_ascii=False)
        
    print("\n=== Cross-Model Validation Summary ===")
    for model, stats in model_stats.items():
        print(f"Model: {model}")
        print(f"  Baseline Mean SDR: {stats['sdr_baseline_mean']:.4f} vs SMART-Graph SDR: {stats['sdr_smart_mean']:.4f}")
        print(f"  t-statistic: {stats['t_statistic']:.4f}, p-value: {stats['p_value']:.4e}, Cohen's d: {stats['cohens_d']:.4f}")

if __name__ == "__main__":
    run_cross_model_study()
