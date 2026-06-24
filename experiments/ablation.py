import os
import sys
import json
import yaml
import time
import numpy as np
import ollama

# Path injection
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from clustering.acbs_engine import ACBSEngine
from evaluators.strict import StrictEvaluator
from evaluators.smga import SMGAEvaluator
from experiments.threshold_sweep import extract_triples_from_text

def run_ablation_study():
    print("=== Starting SMART-Graph Ablation Study ===")
    
    # Load configs
    with open(os.path.join("configs", "smart_graph.yaml"), "r") as f:
        config = yaml.safe_load(f)
        
    model = config.get("model", "qwen2.5:7b")
    embedding_model = config.get("embedding_model", "nomic-embed-text:latest")
    seed = config.get("seed", 42)
    max_size = config.get("max_cluster_size", 8)
    budget = config.get("vulnerability_budget", 1.5)
    sim_threshold = config.get("similarity_threshold", 0.75)
    
    # Load dataset
    with open(os.path.join("data", "generated_large", "synthetic_large.json"), "r") as f:
        large_data = json.load(f)
    triples = large_data[0]["triples"]
    print(f"Dataset: Synthetic Chained Large ({len(triples)} triples)")
    
    # Engines & Evaluators
    engine = ACBSEngine(max_size=max_size, max_vulnerability_budget=budget, embedding_model=embedding_model)
    smga_evaluator = SMGAEvaluator(embedding_model=embedding_model, similarity_threshold=sim_threshold)
    
    sos_scores = engine.compute_sos(triples)
    
    variants = [
        {"name": "1. Flat + Strict", "mode": "baseline_a", "evaluator": "strict"},
        {"name": "2. Flat + SMGA", "mode": "baseline_a", "evaluator": "smga"},
        {"name": "3. Random + SMGA", "mode": "baseline_b", "evaluator": "smga"},
        {"name": "4. SOS + SMGA", "mode": "baseline_c", "evaluator": "smga"},
        {"name": "5. ACBS + SMGA (SMART)", "mode": "smart_graph", "evaluator": "smga"}
    ]
    
    ablation_results = []
    
    for var in variants:
        var_name = var["name"]
        mode = var["mode"]
        eval_type = var["evaluator"]
        print(f"\nEvaluating: {var_name}...")
        
        # Serialize/Partition
        start_time = time.time()
        clusters = engine.serialize_triples(triples, mode, seed=seed)
        
        # Local Paragraph Generation Loop
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
                print(f"Error generating text block: {e}")
                paragraphs.append("")
                
        # Merge if multiple
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
        
        # Extract triples
        extracted_triples = extract_triples_from_text(synthesized_text, model)
        
        # Evaluate
        if eval_type == "strict":
            eval_res = StrictEvaluator.evaluate(triples, extracted_triples, sos_scores)
        else:
            eval_res = smga_evaluator.evaluate(triples, extracted_triples, sos_scores)
            
        record = {
            "variant": var_name,
            "sdr": eval_res["semantic_divergence_rate"],
            "coverage": eval_res["coverage"],
            "aps": eval_res["attention_protection_score"],
            "omissions": eval_res["omissions_count"],
            "hallucinations": eval_res["hallucinations_count"],
            "runtime_seconds": round(runtime, 2)
        }
        
        ablation_results.append(record)
        print(f"  SDR: {record['sdr']:.4f}, Coverage: {record['coverage']*100:.2f}%, APS: {record['aps']*100:.2f}%, Runtime: {record['runtime_seconds']}s")
        
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "ablation_study.json"), "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2, ensure_ascii=False)
        
    print(f"\nAblation results exported to {os.path.join(results_dir, 'ablation_study.json')}")

if __name__ == "__main__":
    run_ablation_study()
