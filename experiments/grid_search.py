import os
import sys
import json
import yaml
import time
import random
import numpy as np
import ollama

# Path injection
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from clustering.acbs_engine import ACBSEngine
from evaluators.smga import SMGAEvaluator
from experiments.threshold_sweep import extract_triples_from_text

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

def select_test_graphs():
    dataset_path = os.path.join("data", "webnlg", "expanded_dataset.json")
    if not os.path.exists(dataset_path):
        # Fallback to native small if expanded dataset isn't generated/accessible
        print("Warning: expanded_dataset.json not found, using native_small.json")
        dataset_path = os.path.join("data", "webnlg", "native_small.json")
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    # Target scales: close to 7, 10, 15, 20, 30
    targets = [7, 10, 15, 20, 30]
    selected = []
    
    # Sort dataset by scale first to make candidate selection predictable
    dataset_sorted = sorted(dataset, key=lambda g: len(g["triples"]))
    
    for t in targets:
        candidates = sorted(dataset_sorted, key=lambda g: abs(len(g["triples"]) - t))
        for cand in candidates:
            if cand["id"] not in [s["id"] for s in selected]:
                selected.append({
                    "id": cand["id"],
                    "scale": len(cand["triples"]),
                    "triples": [tuple(t) for t in cand["triples"]]
                })
                break
    return selected

def compute_pareto_frontier(results_list):
    """
    Finds the Pareto-optimal configurations.
    Goal: Maximize Coverage, Minimize SDR.
    results_list: list of dicts, each having 'max_size', 'budget', 'avg_coverage', 'avg_sdr'
    """
    pareto_frontier = []
    for candidate in results_list:
        dominated = False
        cand_cov = candidate["avg_coverage"]
        cand_sdr = candidate["avg_sdr"]
        
        for other in results_list:
            other_cov = other["avg_coverage"]
            other_sdr = other["avg_sdr"]
            
            # other dominates candidate if:
            # other_cov >= cand_cov and other_sdr <= cand_sdr and at least one is strict
            if (other_cov >= cand_cov and other_sdr <= cand_sdr) and (other_cov > cand_cov or other_sdr < cand_sdr):
                dominated = True
                break
        
        if not dominated:
            pareto_frontier.append(candidate)
            
    # Sort by SDR ascending for plotting
    pareto_frontier.sort(key=lambda x: x["avg_sdr"])
    return pareto_frontier

def run_grid_search():
    print("=== Starting SMART-Graph Parameter Grid Search ===")
    
    # 1. Load configuration defaults
    with open(os.path.join("configs", "smart_graph.yaml"), "r") as f:
        config = yaml.safe_load(f)
        
    model = config.get("model", "qwen2.5:7b")
    embedding_model = config.get("embedding_model", "nomic-embed-text:latest")
    seed = config.get("seed", 42)
    sim_threshold = config.get("similarity_threshold", 0.75)
    
    set_seeds(seed)
    
    # 2. Select representative graphs
    graphs = select_test_graphs()
    print(f"Selected {len(graphs)} test graphs with scales: {[g['scale'] for g in graphs]}")
    
    # 3. Define sweep parameters
    k_range = [4, 6, 8, 10, 12]
    budget_range = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    smga_evaluator = SMGAEvaluator(embedding_model=embedding_model, similarity_threshold=sim_threshold)
    
    # Cache dictionaries to prevent redundant LLM generations/extractions
    paragraph_cache = {}  # key: tuple of triples -> paragraph text
    synthesis_cache = {}  # key: tuple of paragraphs -> synthesized text
    extraction_cache = {} # key: text -> list of extracted triples
    
    sweep_results = []
    
    # Loop over parameters
    total_runs = len(k_range) * len(budget_range)
    run_idx = 0
    
    for k in k_range:
        for b in budget_range:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] Sweeping Configuration: K={k}, Budget={b:.1f}...")
            
            run_coverages = []
            run_sdrs = []
            run_apss = []
            run_runtimes = []
            
            engine = ACBSEngine(max_size=k, max_vulnerability_budget=b, embedding_model=embedding_model)
            
            for graph in graphs:
                triples = graph["triples"]
                sos_scores = engine.compute_sos(triples)
                
                start_time = time.time()
                clusters = engine.serialize_triples(triples, "smart_graph", seed=seed)
                
                # A. Segment descriptions
                paragraphs = []
                for cluster in clusters:
                    # Sort triples in the cluster to normalize key
                    cluster_key = tuple(sorted(tuple(t) for t in cluster))
                    
                    if cluster_key in paragraph_cache:
                        paragraphs.append(paragraph_cache[cluster_key])
                    else:
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
                            text = response["response"]
                            paragraph_cache[cluster_key] = text
                            paragraphs.append(text)
                        except Exception as e:
                            print(f"      Generation error: {e}")
                            paragraphs.append("")
                            
                # B. Synthesize paragraphs
                if len(paragraphs) > 1:
                    synthesis_key = tuple(paragraphs)
                    if synthesis_key in synthesis_cache:
                        synthesized_text = synthesis_cache[synthesis_key]
                    else:
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
                            synthesis_cache[synthesis_key] = synthesized_text
                        except Exception:
                            synthesized_text = " ".join(paragraphs)
                else:
                    synthesized_text = paragraphs[0] if paragraphs else ""
                    
                # C. Extract & Evaluate
                if synthesized_text in extraction_cache:
                    extracted = extraction_cache[synthesized_text]
                else:
                    extracted = extract_triples_from_text(synthesized_text, model)
                    extraction_cache[synthesized_text] = extracted
                    
                runtime = time.time() - start_time
                
                eval_res = smga_evaluator.evaluate(triples, extracted, sos_scores)
                
                run_coverages.append(eval_res["coverage"])
                run_sdrs.append(eval_res["semantic_divergence_rate"])
                run_apss.append(eval_res["attention_protection_score"])
                run_runtimes.append(runtime)
                
            avg_coverage = float(np.mean(run_coverages))
            avg_sdr = float(np.mean(run_sdrs))
            avg_aps = float(np.mean(run_apss))
            avg_runtime = float(np.mean(run_runtimes))
            
            sweep_results.append({
                "max_size": k,
                "budget": b,
                "avg_coverage": avg_coverage,
                "avg_sdr": avg_sdr,
                "avg_aps": avg_aps,
                "avg_runtime": avg_runtime
            })
            
            print(f"    Avg Coverage: {avg_coverage*100:.2f}%, Avg SDR: {avg_sdr:.4f}, Avg APS: {avg_aps*100:.2f}%, Avg Runtime: {avg_runtime:.2f}s")
            
    # Calculate Pareto frontier
    pareto_frontier = compute_pareto_frontier(sweep_results)
    
    # 4. Save results to JSON
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "grid_search.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "sweep_parameters": {
                "max_cluster_sizes": k_range,
                "vulnerability_budgets": budget_range,
                "embedding_model": embedding_model,
                "synthesis_model": model,
                "similarity_threshold": sim_threshold
            },
            "pareto_frontier": pareto_frontier,
            "all_results": sweep_results
        }, f, indent=2, ensure_ascii=False)
        
    print(f"\nResults saved to {results_path}")
    print("\n=== Pareto-Optimal Configurations ===")
    for p in pareto_frontier:
        print(f"K={p['max_size']}, Budget={p['budget']:.1f} -> Coverage: {p['avg_coverage']*100:.2f}%, SDR: {p['avg_sdr']:.4f}")
        
    # 5. Plot Pareto frontier using matplotlib
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 6))
        
        # Plot all configurations
        all_sdr = [r["avg_sdr"] for r in sweep_results]
        all_cov = [r["avg_coverage"] for r in sweep_results]
        plt.scatter(all_sdr, all_cov, color='royalblue', alpha=0.6, label='Swept Configurations', s=60)
        
        # Plot Pareto frontier
        pareto_sdr = [p["avg_sdr"] for p in pareto_frontier]
        pareto_cov = [p["avg_coverage"] for p in pareto_frontier]
        plt.scatter(pareto_sdr, pareto_cov, color='crimson', edgecolor='black', s=100, zorder=5, label='Pareto Frontier')
        plt.plot(pareto_sdr, pareto_cov, color='crimson', linestyle='--', linewidth=2, zorder=4)
        
        # Annotate Pareto points
        for p in pareto_frontier:
            plt.annotate(
                f"K={p['max_size']}, B={p['budget']:.1f}",
                xy=(p["avg_sdr"], p["avg_coverage"]),
                xytext=(8, -5),
                textcoords='offset points',
                fontsize=9,
                weight='bold',
                color='darkred'
            )
            
        plt.title("SMART-Graph Grid Search: Coverage vs SDR Pareto Frontier", fontsize=12, fontweight='bold')
        plt.xlabel("Semantic Divergence Rate (SDR) [Lower is Better]", fontsize=11)
        plt.ylabel("Fact Coverage [Higher is Better]", fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='best', framealpha=0.9)
        plt.tight_layout()
        
        plot_path = os.path.join(results_dir, "pareto_frontier.svg")
        plt.savefig(plot_path, format="svg")
        plt.close()
        print(f"Pareto frontier plot saved to {plot_path}")
    except Exception as e:
        print(f"Warning: Failed to generate Pareto frontier plot: {e}")

if __name__ == "__main__":
    run_grid_search()
