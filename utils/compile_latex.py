import os
import json

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def compile_cross_model(data):
    if not data or "model_statistics" not in data:
        return "% Cross-model data missing\n"
        
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Cross-Model Generality Analysis (Baseline vs. SMART-Graph)}")
    latex.append(r"\label{tab:cross_model}")
    latex.append(r"\begin{tabular}{lccccc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Model} & \textbf{Baseline SDR} & \textbf{SMART SDR} & \textbf{$t$-stat} & \textbf{$p$-value} & \textbf{Cohen's $d$} \\")
    latex.append(r"\midrule")
    
    for model, stats in data["model_statistics"].items():
        # Highlight significant p-values or large Cohen's d
        p_val_str = f"{stats['p_value']:.4f}" if stats['p_value'] >= 0.0001 else f"{stats['p_value']:.2e}"
        if stats['p_value'] < 0.05:
            p_val_str = r"\mathbf{" + p_val_str + "}"
            
        cohen_str = f"{stats['cohens_d']:.4f}"
        if abs(stats['cohens_d']) >= 0.8:
            cohen_str = r"\mathbf{" + cohen_str + "}"
            
        latex.append(f"{model} & {stats['sdr_baseline_mean']:.4f} & {stats['sdr_smart_mean']:.4f} & {stats['t_statistic']:.4f} & {p_val_str} & {cohen_str} \\\\")
        
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    return "\n".join(latex)

def compile_large_graph(data):
    if not data or "results" not in data:
        return "% Large graph data missing\n"
        
    # Group results by scale and mode
    scales = sorted(list(set(r["scale"] for r in data["results"])))
    modes = ["baseline_a", "baseline_b", "baseline_d", "smart_graph"]
    mode_names = {
        "baseline_a": "Flat Seq.",
        "baseline_b": "Flat Random",
        "baseline_d": "Modularity",
        "smart_graph": "SMART-Graph"
    }
    
    grouped = {}
    for r in data["results"]:
        scale = r["scale"]
        mode = r["mode"]
        if scale not in grouped:
            grouped[scale] = {}
        grouped[scale][mode] = r
        
    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Large Graph Scaling Benchmarks (Fact Coverage \% / Semantic Divergence Rate)}")
    latex.append(r"\label{tab:large_graph}")
    latex.append(r"\begin{tabular}{ccccc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Scale (Triples)} & \textbf{Flat Seq. (A)} & \textbf{Flat Random (B)} & \textbf{Modularity (D)} & \textbf{SMART-Graph (Ours)} \\")
    latex.append(r"\midrule")
    
    for scale in scales:
        row_parts = [f"{scale}"]
        for mode in modes:
            r = grouped[scale].get(mode)
            if r:
                cov = r["coverage"] * 100
                sdr = r["sdr"]
                
                # Check if this mode is best in coverage or SDR for this scale
                all_covs = [grouped[scale][m]["coverage"] for m in modes if m in grouped[scale]]
                all_sdrs = [grouped[scale][m]["sdr"] for m in modes if m in grouped[scale]]
                
                is_best_cov = (r["coverage"] == max(all_covs))
                is_best_sdr = (r["sdr"] == min(all_sdrs))
                
                cov_str = f"{cov:.2f}\\%"
                sdr_str = f"{sdr:.4f}"
                
                if is_best_cov:
                    cov_str = r"\mathbf{" + cov_str + "}"
                if is_best_sdr:
                    sdr_str = r"\mathbf{" + sdr_str + "}"
                    
                row_parts.append(f"{cov_str} / {sdr_str}")
            else:
                row_parts.append("N/A")
        latex.append(" & ".join(row_parts) + " \\\\")
        
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")
    return "\n".join(latex)

def compile_grid_search(data):
    if not data or "pareto_frontier" not in data:
        return "% Grid search data missing\n"
        
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Pareto-Optimal Hyperparameter Configurations (Coverage vs. SDR)}")
    latex.append(r"\label{tab:pareto_frontier}")
    latex.append(r"\begin{tabular}{ccccc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Config ($K, \mathcal{B}$)} & \textbf{Avg. Coverage} & \textbf{Avg. SDR} & \textbf{Avg. APS} & \textbf{Avg. Runtime (s)} \\")
    latex.append(r"\midrule")
    
    for p in data["pareto_frontier"]:
        latex.append(f"($K={p['max_size']}$, $\\mathcal{{B}}={p['budget']:.1f}$) & {p['avg_coverage']*100:.2f}\\% & {p['avg_sdr']:.4f} & {p['avg_aps']*100:.2f}\\% & {p['avg_runtime']:.2f}s \\\\")
        
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    return "\n".join(latex)

def compile_all():
    print("=== Compiling LaTeX Academic Tables ===")
    
    cross_model_data = load_json(os.path.join("results", "cross_model_study.json"))
    large_graph_data = load_json(os.path.join("results", "large_graph_study.json"))
    grid_search_data = load_json(os.path.join("results", "grid_search.json"))
    
    latex_out = []
    latex_out.append(r"% ==================================================================")
    latex_out.append(r"% AUTO-GENERATED ACADEMIC TABLES FOR SMART-GRAPH MANUSCRIPT")
    latex_out.append(r"% ==================================================================")
    latex_out.append("\n")
    
    latex_out.append(compile_cross_model(cross_model_data))
    latex_out.append("\n\n")
    
    latex_out.append(compile_large_graph(large_graph_data))
    latex_out.append("\n\n")
    
    latex_out.append(compile_grid_search(grid_search_data))
    latex_out.append("\n")
    
    output_path = os.path.join("results", "latex_tables.tex")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_out))
        
    print(f"LaTeX tables compiled and saved to {output_path}")

if __name__ == "__main__":
    compile_all()
