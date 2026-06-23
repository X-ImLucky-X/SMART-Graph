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
from utils.hardware import get_hardware_info

def run_large_graph_study():
    print("=== Starting SMART-Graph Large Graph Scaling Study ===")
    
    # 1. Load configs
    with open(os.path.join("configs", "smart_graph.yaml"), "r") as f:
        config = yaml.safe_load(f)
        
    model = config.get("model", "qwen2.5:7b")
    embedding_model = config.get("embedding_model", "nomic-embed-text:latest")
    seed = config.get("seed", 42)
    max_size = config.get("max_cluster_size", 8)
    budget = config.get("vulnerability_budget", 1.5)
    sim_threshold = config.get("similarity_threshold", 0.75)
    
    random.seed(seed)
    np.random.seed(seed)
    
    engine = ACBSEngine(max_size=max_size, max_vulnerability_budget=budget, embedding_model=embedding_model)
    smga_evaluator = SMGAEvaluator(embedding_model=embedding_model, similarity_threshold=sim_threshold)
    
    # 2. Load the base synthetic large graph (47 triples)
    with open(os.path.join("data", "generated_large", "synthetic_large.json"), "r") as f:
        large_data = json.load(f)
    base_triples = large_data[0]["triples"]
    
    # 3. Construct Scale Datasets (40, 50, 60, 70 triples)
    # We construct connected scaling graphs by subsetting/expanding the chained graph
    scale_graphs = {}
    
    # Scale 40: Subset of the base triples
    scale_graphs[40] = base_triples[:40]
    
    # Scale 50: Base triples + extra triples
    scale_50 = list(base_triples)
    # Let's add extra connected triples (e.g. Spain currency, Spanish language, Madrid elevation details, etc.)
    extra_50 = [
        ["Brussels, Belgium", "timezone", "Central European Time"],
        ["Adolfo Suarez Airport", "operator", "Aena"],
        ["Steve Jobs", "nationality", "United States"]
    ]
    for t in extra_50:
        if t not in scale_50:
            scale_50.append(t)
    scale_graphs[50] = scale_50[:50]
    
    # Scale 60: Base triples + more extras
    scale_60 = list(scale_50)
    extra_60 = [
        ["Adolfo Suarez Airport", "opened", "1931"],
        ["Adolfo Suarez Airport", "hubFor", "Iberia"],
        ["Spain", "governmentType", "Parliamentary monarchy"],
        ["Madrid", "leaderTitle", "Mayor of Madrid"],
        ["Ankara", "country", "Turkey"],
        ["Turkey", "populationTotal", "84680273"],
        ["Ankara", "established", "Bronze Age"],
        ["Baklava", "origin", "Gaziantep, Turkey"],
        ["Steve Jobs", "died", "October 5, 2011"],
        ["Apple Inc", "industry", "Consumer electronics"]
    ]
    for t in extra_60:
        if t not in scale_60:
            scale_60.append(t)
    scale_graphs[60] = scale_60[:60]
    
    # Scale 70: Expand to 70 triples
    scale_70 = list(scale_60)
    extra_70 = [
        ["Tim Cook", "born", "November 1, 1960"],
        ["Tim Cook", "birthPlace", "Mobile, Alabama"],
        ["Mobile, Alabama", "country", "United States"],
        ["Auburn University", "mascot", "Aubie the Tiger"],
        ["Auburn University", "nickname", "Tigers"],
        ["Alamo Plaza", "architect", "Alamo mission built by Spanish Empire"],
        ["San Antonio, Texas", "populationTotal", "1434625"],
        ["San Antonio, Texas", "timezone", "Central Time Zone"],
        ["Alan_Bean", "rank", "Captain, US Navy"],
        ["NASA", "founded", "July 29, 1958"]
    ]
    for t in extra_70:
        if t not in scale_70:
            scale_70.append(t)
    scale_graphs[70] = scale_70[:70]
    
    # 4. Benchmark Execution Loop
    modes = ["baseline_a", "baseline_b", "baseline_d", "smart_graph"]
    large_results = []
    
    for scale, triples in scale_graphs.items():
        print(f"\nBenchmarking scale: {scale} triples...")
        sos_scores = engine.compute_sos(triples)
        
        for mode in modes:
            print(f"  Running Mode: {mode}...")
            start_time = time.time()
            clusters = engine.serialize_triples(triples, mode, seed=seed)
            
            # Local Windowed Generation Loop
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
                    print(f"    Error: {e}")
                    paragraphs.append("")
            
            # Hierarchical Merge
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
            
            # Triple Extraction & SMGA Evaluation
            extracted = extract_triples_from_text(synthesized_text, model)
            eval_res = smga_evaluator.evaluate(triples, extracted, sos_scores)
            
            record = {
                "scale": scale,
                "mode": mode,
                "sdr": eval_res["semantic_divergence_rate"],
                "coverage": eval_res["coverage"],
                "aps": eval_res["attention_protection_score"],
                "omissions": eval_res["omissions_count"],
                "hallucinations": eval_res["hallucinations_count"],
                "runtime_seconds": round(runtime, 2)
            }
            large_results.append(record)
            print(f"    SDR: {record['sdr']:.4f}, Coverage: {record['coverage']*100:.2f}%, APS: {record['aps']*100:.2f}%, Runtime: {record['runtime_seconds']}s")
            
    # 5. Calculate Paired Statistics
    sdr_a = [r["sdr"] for r in large_results if r["mode"] == "baseline_a"]
    sdr_smart = [r["sdr"] for r in large_results if r["mode"] == "smart_graph"]
    
    t_stat, p_val = ttest_rel(sdr_a, sdr_smart)
    
    # Cohen's d
    diff = np.array(sdr_a) - np.array(sdr_smart)
    std_diff = np.std(diff, ddof=1)
    cohens_d = np.mean(diff) / std_diff if std_diff > 0 else 0.0
    
    print("\n=== Large Graph Study Summary ===")
    print(f"Paired T-test (Baseline A vs SMART SDR): t = {t_stat:.4f}, p = {p_val:.4e}")
    print(f"Cohen's d Effect Size: {cohens_d:.4f}")
    
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "large_graph_study.json"), "w", encoding="utf-8") as f:
        json.dump({
            "paired_stats": {
                "t_statistic": float(t_stat) if not np.isnan(t_stat) else 0.0,
                "p_value": float(p_val) if not np.isnan(p_val) else 1.0,
                "cohens_d": float(cohens_d)
            },
            "results": large_results
        }, f, indent=2, ensure_ascii=False)
        
    print(f"Results exported to {os.path.join(results_dir, 'large_graph_study.json')}")

if __name__ == "__main__":
    run_large_graph_study()
