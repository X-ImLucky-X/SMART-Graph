import os
import csv
import json
import numpy as np

def compute_cohens_d(x, y):
    """Computes paired Cohen's d: mean(x - y) / std(x - y)"""
    diff = np.array(x) - np.array(y)
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0:
        return 0.0
    return np.mean(diff) / std_diff

def generate_effect_size_report():
    print("=== Generating Paired Effect Size (Cohen's d) Report ===")
    csv_path = os.path.join("results", "benchmark.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run the benchmark first.")
        return
        
    # Group by graph name
    runs_by_graph = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            g_name = row["graph_name"]
            if g_name not in runs_by_graph:
                runs_by_graph[g_name] = {}
            runs_by_graph[g_name][row["mode"]] = {
                "coverage": float(row["smga_coverage"]),
                "sdr": float(row["smga_sdr"])
            }
            
    # Align pairs
    flat_covs = []
    random_covs = []
    sos_covs = []
    smart_covs = []
    
    for g_name, modes in runs_by_graph.items():
        if "smart_graph" in modes:
            smart_covs.append(modes["smart_graph"]["coverage"])
            
            # Match baseline modes, fallback if some runs are missing (for legacy logs)
            flat_covs.append(modes.get("baseline_a", {}).get("coverage", 0.0))
            random_covs.append(modes.get("baseline_b", {}).get("coverage", 0.0))
            sos_covs.append(modes.get("baseline_c", {}).get("coverage", 0.0))
            
    # Calculate effect sizes on coverage (SMART - baseline)
    d_flat = compute_cohens_d(smart_covs, flat_covs)
    d_random = compute_cohens_d(smart_covs, random_covs)
    d_sos = compute_cohens_d(smart_covs, sos_covs)
    
    print(f"Effect Sizes (Cohen's d on Fact Coverage):")
    print(f"  SMART-Graph vs. Flat:   d = {d_flat:.4f}")
    print(f"  SMART-Graph vs. Random: d = {d_random:.4f}")
    print(f"  SMART-Graph vs. SOS:    d = {d_sos:.4f}")
    
    # Save results
    results_dir = "results"
    with open(os.path.join(results_dir, "effect_sizes.json"), "w", encoding="utf-8") as f:
        json.dump({
            "smart_vs_flat": d_flat,
            "smart_vs_random": d_random,
            "smart_vs_sos": d_sos
        }, f, indent=2, ensure_ascii=False)
        
    print(f"Effect sizes saved to {os.path.join(results_dir, 'effect_sizes.json')}")

if __name__ == "__main__":
    generate_effect_size_report()
