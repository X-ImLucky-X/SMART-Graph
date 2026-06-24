import os
import json

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def compile_table_1(data):
    if not data:
        return "% Ablation data missing\n"
    latex = []
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(r"\caption{Controlled Ablation Matrix on 47-Triple Graph}")
    latex.append(r"\label{tab:ablation_study}")
    latex.append(r"\begin{tabular}{llccccc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Variant} & \textbf{Serialization} & \textbf{Evaluation} & \textbf{Coverage (\%)} & \textbf{SDR} & \textbf{APS (\%)} & \textbf{Runtime (s)} \\")
    latex.append(r"\midrule")
    
    mapping = {
        "1. Flat + Strict": ("Flat", "Strict"),
        "2. Flat + SMGA": ("Flat", "SMGA"),
        "3. Random + SMGA": ("Random", "SMGA"),
        "4. SOS + SMGA": ("SOS", "SMGA"),
        "5. ACBS + SMGA (SMART)": ("ACBS", "SMGA")
    }
    
    # Pre-extract values for bolding (only for SMGA evaluated rows, i.e., indices 1 to 4)
    smga_items = [item for item in data if item["variant"] in mapping and mapping[item["variant"]][1] == "SMGA"]
    max_cov = max(item["coverage"] for item in smga_items) if smga_items else 0.0
    min_sdr = min(item["sdr"] for item in smga_items) if smga_items else 999.0
    max_aps = max(item["aps"] for item in smga_items) if smga_items else 0.0
    
    for item in data:
        name = item["variant"]
        ser, eval_type = mapping.get(name, (name, "SMGA"))
        cov = item["coverage"] * 100
        sdr = item["sdr"]
        aps = item["aps"] * 100
        runtime = item["runtime_seconds"]
        
        cov_str = f"{cov:.2f}\\%"
        sdr_str = f"{sdr:.4f}"
        aps_str = f"{aps:.2f}\\%"
        
        # Apply bolding if it's the best among SMGA variants
        if eval_type == "SMGA":
            if item["coverage"] == max_cov:
                cov_str = r"\mathbf{" + cov_str + "}"
            if item["sdr"] == min_sdr:
                sdr_str = r"\mathbf{" + sdr_str + "}"
            if item["aps"] == max_aps:
                aps_str = r"\mathbf{" + aps_str + "}"
                
        latex.append(f"{name} & {ser} & {eval_type} & {cov_str} & {sdr_str} & {aps_str} & {runtime:.2f}s \\\\")
        
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    return "\n".join(latex)

def compile_table_2(data):
    if not data:
        return "% Rigorous benchmark data missing\n"
        
    modes = ["baseline_a", "baseline_b", "baseline_c", "baseline_d", "smart_graph"]
    
    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Rigorous Benchmark Results: Fact Coverage and Semantic Divergence Rate}")
    latex.append(r"\label{tab:benchmark_rigorous}")
    latex.append(r"\begin{tabular}{llccccc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Category} & \textbf{Metric} & \textbf{Flat Seq.} & \textbf{Flat Random} & \textbf{Flat SOS} & \textbf{Modularity} & \textbf{SMART-Graph (Ours)} \\")
    latex.append(r"\midrule")
    
    # 1. By Graph Size
    latex.append(r"\multicolumn{7}{l}{\textit{By Graph Size}} \\")
    for size_cat in sorted(data["by_size"].keys()):
        cov_row = [f"{size_cat}", "Coverage (\%)"]
        sdr_row = ["", "SDR"]
        
        # Find best values for bolding
        covs = {m: data["by_size"][size_cat].get(m, {}).get("coverage", 0.0) for m in modes}
        sdrs = {m: data["by_size"][size_cat].get(m, {}).get("sdr", 999.0) for m in modes}
        
        max_cov = max(covs.values())
        min_sdr = min(sdrs.values())
        
        for m in modes:
            val = covs[m] * 100
            val_str = f"{val:.2f}\\%"
            if covs[m] == max_cov:
                val_str = r"\mathbf{" + val_str + "}"
            cov_row.append(val_str)
            
            val = sdrs[m]
            val_str = f"{val:.4f}"
            if sdrs[m] == min_sdr:
                val_str = r"\mathbf{" + val_str + "}"
            sdr_row.append(val_str)
            
        latex.append(" & ".join(cov_row) + " \\\\")
        latex.append(" & ".join(sdr_row) + " \\\\")
        
    latex.append(r"\midrule")
    
    # 2. By Dataset Category
    latex.append(r"\multicolumn{7}{l}{\textit{By Dataset Source}} \\")
    for ds_cat in sorted(data["by_dataset"].keys()):
        cov_row = [f"{ds_cat}", "Coverage (\%)"]
        sdr_row = ["", "SDR"]
        
        # Find best values for bolding
        covs = {m: data["by_dataset"][ds_cat].get(m, {}).get("coverage", 0.0) for m in modes}
        sdrs = {m: data["by_dataset"][ds_cat].get(m, {}).get("sdr", 999.0) for m in modes}
        
        max_cov = max(covs.values())
        min_sdr = min(sdrs.values())
        
        for m in modes:
            val = covs[m] * 100
            val_str = f"{val:.2f}\\%"
            if covs[m] == max_cov:
                val_str = r"\mathbf{" + val_str + "}"
            cov_row.append(val_str)
            
            val = sdrs[m]
            val_str = f"{val:.4f}"
            if sdrs[m] == min_sdr:
                val_str = r"\mathbf{" + val_str + "}"
            sdr_row.append(val_str)
            
        latex.append(" & ".join(cov_row) + " \\\\")
        latex.append(" & ".join(sdr_row) + " \\\\")
        
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")
    return "\n".join(latex)

def compile_table_3(data):
    if not data:
        return "% Rigorous benchmark data missing\n"
        
    modes = ["baseline_a", "baseline_b", "baseline_c", "baseline_d", "smart_graph"]
    mode_names = {
        "baseline_a": "Flat Sequential",
        "baseline_b": "Flat Random",
        "baseline_c": "Flat SOS",
        "baseline_d": "Modularity",
        "smart_graph": "SMART-Graph"
    }
    
    latex = []
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(r"\caption{Fact Omissions vs. Hallucinations Count Breakdown}")
    latex.append(r"\label{tab:omissions_hallucinations}")
    latex.append(r"\begin{tabular}{lccccc}")
    latex.append(r"\toprule")
    latex.append(r" & \multicolumn{2}{c}{\textbf{Small Graphs (WebNLG)}} & & \multicolumn{2}{c}{\textbf{Large Graphs (Synthetic)}} \\")
    latex.append(r"\cmidrule{2-3} \cmidrule{5-6}")
    latex.append(r"\textbf{Mode} & \textbf{Omissions} & \textbf{Hallucinations} & & \textbf{Omissions} & \textbf{Hallucinations} \\")
    latex.append(r"\midrule")
    
    webnlg = data["by_dataset"].get("Dataset A (WebNLG)", {})
    synthetic = data["by_dataset"].get("Dataset B (Synthetic)", {})
    
    # Pre-extract best (minimum) values
    webnlg_oms = [webnlg.get(mode, {}).get("omissions", 999.0) for mode in modes]
    webnlg_hals = [webnlg.get(mode, {}).get("hallucinations", 999.0) for mode in modes]
    synth_oms = [synthetic.get(mode, {}).get("omissions", 999.0) for mode in modes]
    synth_hals = [synthetic.get(mode, {}).get("hallucinations", 999.0) for mode in modes]
    
    min_w_om = min(webnlg_oms) if webnlg_oms else 0.0
    min_w_hal = min(webnlg_hals) if webnlg_hals else 0.0
    min_s_om = min(synth_oms) if synth_oms else 0.0
    min_s_hal = min(synth_hals) if synth_hals else 0.0
    
    for m in modes:
        name = mode_names[m]
        
        webnlg_om = webnlg.get(m, {}).get("omissions", 0.0)
        webnlg_hal = webnlg.get(m, {}).get("hallucinations", 0.0)
        
        synth_om = synthetic.get(m, {}).get("omissions", 0.0)
        synth_hal = synthetic.get(m, {}).get("hallucinations", 0.0)
        
        w_om_str = f"{webnlg_om:.2f}"
        if webnlg_om == min_w_om:
            w_om_str = r"\mathbf{" + w_om_str + "}"
            
        w_hal_str = f"{webnlg_hal:.2f}"
        if webnlg_hal == min_w_hal:
            w_hal_str = r"\mathbf{" + w_hal_str + "}"
            
        s_om_str = f"{synth_om:.1f}"
        if synth_om == min_s_om:
            s_om_str = r"\mathbf{" + s_om_str + "}"
            
        s_hal_str = f"{synth_hal:.1f}"
        if synth_hal == min_s_hal:
            s_hal_str = r"\mathbf{" + s_hal_str + "}"
            
        latex.append(f"{name} & {w_om_str} & {w_hal_str} & & {s_om_str} & {s_hal_str} \\\\")
        
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    return "\n".join(latex)

def compile_table_4(data):
    if not data:
        return "% Effect sizes data missing\n"
        
    latex = []
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(r"\caption{Paired Effect Sizes (Cohen's $d$) on Fact Coverage}")
    latex.append(r"\label{tab:effect_sizes}")
    latex.append(r"\begin{tabular}{lc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Comparison} & \textbf{Cohen's $d$} \\")
    latex.append(r"\midrule")
    
    comparisons = [
        ("smart_vs_flat", "SMART-Graph vs. Flat Sequential"),
        ("smart_vs_random", "SMART-Graph vs. Flat Random"),
        ("smart_vs_sos", "SMART-Graph vs. Flat SOS Descending")
    ]
    
    for key, label in comparisons:
        val = data.get(key, 0.0)
        val_str = f"{val:.4f}"
        
        # Annotate effect magnitude based on Cohen's standards
        if abs(val) >= 0.8:
            val_str = r"\mathbf{" + val_str + "} (Large)"
        elif abs(val) >= 0.5:
            val_str = f"{val_str} (Medium)"
        elif abs(val) >= 0.2:
            val_str = f"{val_str} (Small)"
        else:
            val_str = f"{val_str} (Negligible)"
            
        latex.append(f"{label} & {val_str} \\\\")
        
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    return "\n".join(latex)

def compile_table_5(data):
    if not data:
        return "% APS validation data missing\n"
        
    latex = []
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(r"\caption{Attention Protection Score (APS) Validation Against Fact Coverage}")
    latex.append(r"\label{tab:aps_validation}")
    latex.append(r"\begin{tabular}{lcc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Correlation Metric} & \textbf{Coefficient} & \textbf{$p$-value} \\")
    latex.append(r"\midrule")
    
    metrics = [
        ("pearson", "Pearson ($r$)"),
        ("spearman", "Spearman ($\\rho$)")
    ]
    
    for key, label in metrics:
        metric_data = data.get(key, {})
        coef = metric_data.get("correlation", 0.0)
        p_val = metric_data.get("p_value", 1.0)
        
        p_val_str = f"{p_val:.4f}" if p_val >= 0.0001 else f"{p_val:.2e}"
        if p_val < 0.05:
            p_val_str = r"\mathbf{" + p_val_str + "}"
            
        coef_str = f"{coef:.4f}"
        if abs(coef) >= 0.3:
            coef_str = r"\mathbf{" + coef_str + "}"
            
        latex.append(f"{label} & {coef_str} & {p_val_str} \\\\")
        
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    return "\n".join(latex)

def compile_all():
    print("=== Compiling LaTeX Academic Tables ===")
    
    ablation_data = load_json(os.path.join("results", "ablation_study.json"))
    benchmark_data = load_json(os.path.join("results", "benchmark_rigorous.json"))
    effect_sizes_data = load_json(os.path.join("results", "effect_sizes.json"))
    aps_data = load_json(os.path.join("results", "aps_validation.json"))
    
    latex_out = []
    latex_out.append(r"% ==================================================================")
    latex_out.append(r"% AUTO-GENERATED ACADEMIC TABLES FOR SMART-GRAPH MANUSCRIPT")
    latex_out.append(r"% ==================================================================")
    latex_out.append("\n")
    
    print("Generating Table 1 (Controlled Ablation Study)...")
    latex_out.append(compile_table_1(ablation_data))
    latex_out.append("\n\n")
    
    print("Generating Table 2 (Benchmark by Size/Dataset)...")
    latex_out.append(compile_table_2(benchmark_data))
    latex_out.append("\n\n")
    
    print("Generating Table 3 (Omissions vs. Hallucinations breakdown)...")
    latex_out.append(compile_table_3(benchmark_data))
    latex_out.append("\n\n")
    
    print("Generating Table 4 (Paired Cohen's d effect sizes)...")
    latex_out.append(compile_table_4(effect_sizes_data))
    latex_out.append("\n\n")
    
    print("Generating Table 5 (APS Validation)...")
    latex_out.append(compile_table_5(aps_data))
    latex_out.append("\n")
    
    output_path = os.path.join("results", "latex_tables.tex")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_out))
        
    print(f"LaTeX tables compiled and saved to {output_path}")

if __name__ == "__main__":
    compile_all()
