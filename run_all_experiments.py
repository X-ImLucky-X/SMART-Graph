import os
import sys
import subprocess
import time

def run_script(script_path):
    print(f"\n==================================================================")
    print(f"RUNNING: {script_path}")
    print(f"==================================================================")
    
    start_time = time.time()
    # Run the script using the current python executable
    res = subprocess.run([sys.executable, script_path], capture_output=False)
    runtime = time.time() - start_time
    
    if res.returncode == 0:
        print(f"SUCCESS: {script_path} completed in {runtime:.2f} seconds.")
        return True
    else:
        print(f"FAILURE: {script_path} failed with exit code {res.returncode}.")
        return False

def main():
    print("=== Starting SMART-Graph Master Benchmarking Pipeline ===")
    start_time = time.time()
    
    pipeline = [
        "experiments/sos_correlation.py",
        "experiments/ablation.py",
        "experiments/large_graph_study.py",
        "experiments/cross_model_study.py",
        "experiments/sentence_isolation_study.py",
        "run_benchmark.py",
        "experiments/validate_aps.py",
        "experiments/effect_size_report.py",
        "experiments/grid_search.py",
        "utils/compile_latex.py"
    ]
    
    failed = []
    for script in pipeline:
        # Check if file exists first
        if not os.path.exists(script):
            print(f"Error: Script {script} does not exist. Skipping.")
            failed.append(script)
            continue
            
        success = run_script(script)
        if not success:
            failed.append(script)
            
    total_time = time.time() - start_time
    print(f"\n==================================================================")
    print(f"Pipeline finished in {total_time/60:.2f} minutes.")
    if failed:
        print(f"The following scripts failed: {failed}")
        sys.exit(1)
    else:
        print("All scripts executed successfully! All results are updated in results/.")
        sys.exit(0)

if __name__ == "__main__":
    main()
