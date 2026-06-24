import os
import csv
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr

def validate_aps():
    print("=== Validating Attention Protection Score (APS) Metric ===")
    csv_path = os.path.join("results", "benchmark.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run the benchmark first.")
        return
        
    coverages = []
    aps_scores = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            coverages.append(float(row["smga_coverage"]))
            aps_scores.append(float(row["aps"]))
            
    coverages = np.array(coverages)
    aps_scores = np.array(aps_scores)
    
    # Pearson
    p_corr, p_pval = pearsonr(coverages, aps_scores)
    # Spearman
    s_corr, s_pval = spearmanr(coverages, aps_scores)
    
    print(f"Pearson correlation: r = {p_corr:.4f}, p-value = {p_pval:.4e}")
    print(f"Spearman correlation: rho = {s_corr:.4f}, p-value = {s_pval:.4e}")
    
    # Save results
    results_dir = "results"
    with open(os.path.join(results_dir, "aps_validation.json"), "w", encoding="utf-8") as f:
        json.dump({
            "pearson": {"correlation": float(p_corr), "p_value": float(p_pval)},
            "spearman": {"correlation": float(s_corr), "p_value": float(s_pval)}
        }, f, indent=2, ensure_ascii=False)
        
    if abs(p_corr) > 0.6:
        print(f"\nVerdict: APS has a STRONG correlation with Coverage (r = {p_corr:.4f}). The metric is highly meaningful.")
    elif abs(p_corr) > 0.3:
        print(f"\nVerdict: APS has a MODERATE correlation with Coverage (r = {p_corr:.4f}).")
    else:
        print(f"\nVerdict: APS has a WEAK or no correlation with Coverage (r = {p_corr:.4f}). Consider de-emphasizing it.")

if __name__ == "__main__":
    validate_aps()
