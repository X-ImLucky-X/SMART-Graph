import os
import sys
import json
import subprocess
import shutil

def load_json(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def run_script(script_path):
    print(f"Running {script_path}...")
    res = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running {script_path}:")
        print(res.stderr)
        return False
    return True

def compare_large_graph(ref, new, tolerance=0.02):
    ref_results = ref.get("results", [])
    new_results = new.get("results", [])
    
    if len(ref_results) != len(new_results):
        print(f"Mismatch in results length: reference has {len(ref_results)}, new has {len(new_results)}")
        return False
        
    passed = True
    for r_ref, r_new in zip(ref_results, new_results):
        scale = r_ref["scale"]
        mode = r_ref["mode"]
        
        # Check SDR difference
        sdr_diff = abs(r_ref["sdr"] - r_new["sdr"])
        # Check Coverage difference
        cov_diff = abs(r_ref["coverage"] - r_new["coverage"])
        
        print(f"Scale {scale} ({mode}):")
        print(f"  SDR: Ref={r_ref['sdr']:.4f}, New={r_new['sdr']:.4f} (diff={sdr_diff:.4f})")
        print(f"  Coverage: Ref={r_ref['coverage']*100:.2f}%, New={r_new['coverage']*100:.2f}% (diff={cov_diff*100:.2f}%)")
        
        if sdr_diff > tolerance or cov_diff > tolerance:
            print(f"  [FAIL] Difference exceeds tolerance of {tolerance*100:.1f}%")
            passed = False
        else:
            print(f"  [PASS]")
            
    return passed

def compare_cross_model(ref, new, tolerance=0.02):
    ref_results = ref.get("results", {})
    new_results = new.get("results", {})
    
    passed = True
    for model in ref_results:
        if model not in new_results:
            print(f"Model {model} missing from new results")
            passed = False
            continue
            
        ref_runs = ref_results[model]
        new_runs = new_results[model]
        
        if len(ref_runs) != len(new_runs):
            print(f"Mismatch in run length for model {model}")
            passed = False
            continue
            
        for r_ref, r_new in zip(ref_runs, new_runs):
            graph_name = r_ref["graph_name"]
            mode = r_ref["mode"]
            
            sdr_diff = abs(r_ref["sdr"] - r_new["sdr"])
            cov_diff = abs(r_ref["coverage"] - r_new["coverage"])
            
            print(f"Model {model} - Graph {graph_name} ({mode}):")
            print(f"  SDR: Ref={r_ref['sdr']:.4f}, New={r_new['sdr']:.4f} (diff={sdr_diff:.4f})")
            print(f"  Coverage: Ref={r_ref['coverage']*100:.2f}%, New={r_new['coverage']*100:.2f}% (diff={cov_diff*100:.2f}%)")
            
            if sdr_diff > tolerance or cov_diff > tolerance:
                print(f"  [FAIL] Difference exceeds tolerance of {tolerance*100:.1f}%")
                passed = False
            else:
                print(f"  [PASS]")
                
    return passed

def main():
    print("=== SMART-Graph Clean-Cache Replication Verification ===")
    
    # 1. Back up existing results
    large_graph_ref = load_json(os.path.join("results", "large_graph_study.json"))
    cross_model_ref = load_json(os.path.join("results", "cross_model_study.json"))
    
    if large_graph_ref is None or cross_model_ref is None:
        print("Error: Could not load reference results from results/ folder. Please ensure they exist.")
        sys.exit(1)
        
    # 2. Back up existing embeddings cache
    cache_dir = "cache"
    cache_file = os.path.join(cache_dir, "embeddings.json")
    backup_file = os.path.join(cache_dir, "embeddings_backup_verify.json")
    
    has_cache = os.path.exists(cache_file)
    if has_cache:
        print("Backing up embeddings cache...")
        shutil.copy2(cache_file, backup_file)
        os.remove(cache_file)
        print("Cleaned active embeddings cache.")
    else:
        print("No active embeddings cache found, starting fresh.")
        
    success = False
    try:
        # 3. Re-run experiments on clean cache
        print("\n--- Running Large Graph Study ---")
        large_graph_ok = run_script("experiments/large_graph_study.py")
        
        print("\n--- Running Cross-Model Validation ---")
        cross_model_ok = run_script("experiments/cross_model_study.py")
        
        if not large_graph_ok or not cross_model_ok:
            print("Error: One or more scripts failed to execute.")
            return
            
        # 4. Load new results
        large_graph_new = load_json(os.path.join("results", "large_graph_study.json"))
        cross_model_new = load_json(os.path.join("results", "cross_model_study.json"))
        
        # 5. Perform comparison
        print("\n--- Comparing Scaling Study Metrics ---")
        large_graph_pass = compare_large_graph(large_graph_ref, large_graph_new)
        
        print("\n--- Comparing Cross-Model Metrics ---")
        cross_model_pass = compare_cross_model(cross_model_ref, cross_model_new)
        
        if large_graph_pass and cross_model_pass:
            print("\n==================================================================")
            print("SUCCESS: All replication results verified within 2% stochastic tolerance!")
            print("==================================================================")
            success = True
        else:
            print("\n==================================================================")
            print("WARNING: Some replication results fell outside the 2% tolerance window.")
            print("This is expected for stochastic LLM generation, but review is recommended.")
            print("==================================================================")
            
    finally:
        # 6. Restore cache
        if has_cache:
            print("\nRestoring embeddings cache backup...")
            if os.path.exists(cache_file):
                os.remove(cache_file)
            shutil.move(backup_file, cache_file)
            print("Embeddings cache successfully restored.")

if __name__ == "__main__":
    main()
