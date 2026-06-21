import os
import sys
import json
import yaml
import numpy as np
import ollama

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluators.smga import SMGAEvaluator
from utils.vector_engine import VectorSpaceProfiler

def extract_triples_from_text(text: str, model_name: str) -> list[tuple[str, str, str]]:
    """
    Prompts Ollama to extract factual triples from generated text in strict JSON format.
    """
    prompt = (
        "Analyze the following text and extract all raw factual facts in the form of "
        "RDF triples [Subject, Predicate, Object]. Output ONLY a JSON object containing "
        "a list of triples under the key 'triples'. Example format:\n"
        "{\n"
        "  \"triples\": [\n"
        "    [\"Subject\", \"Predicate\", \"Object\"]\n"
        "  ]\n"
        "}\n"
        "Do not write any introductory or concluding text.\n\n"
        f"Text:\n{text}"
    )
    
    try:
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            format="json",
            options={"temperature": 0.0, "seed": 42}
        )
        data = json.loads(response["response"])
        extracted = []
        if isinstance(data, dict) and "triples" in data:
            for t in data["triples"]:
                if isinstance(t, list) and len(t) >= 3:
                    extracted.append((str(t[0]), str(t[1]), str(t[2])))
        elif isinstance(data, list):
            for t in data:
                if isinstance(t, list) and len(t) >= 3:
                    extracted.append((str(t[0]), str(t[1]), str(t[2])))
        return extracted
    except Exception as e:
        print(f"Warning: Failed to extract triples via JSON mode: {e}. Trying regex fallback.")
        # Fallback parsing
        return []

def run_threshold_sweep():
    print("=== Starting Similarity Threshold Sweep ===")
    
    # Load config
    with open(os.path.join("configs", "baseline.yaml"), "r") as f:
        config = yaml.safe_load(f)
    model = config.get("model", "qwen2.5:7b")
    embedding_model = config.get("embedding_model", "nomic-embed-text:latest")
    
    # Load some native small data
    with open(os.path.join("data", "webnlg", "native_small.json"), "r") as f:
        small_tasks = json.load(f)
        
    # Take a couple of tasks to form a test sample
    selected_task = small_tasks[0]  # Astronaut triples (7 triples)
    input_triples = selected_task["triples"]
    
    print(f"Task Category: {selected_task['category']}")
    print(f"Input Triple Count: {len(input_triples)}")
    
    # Generate text for this task
    flat_prompt = (
        "Generate a brief, factual description incorporating all of the following facts:\n\n" +
        "\n".join(f"- {s} {p} {o}" for s, p, o in input_triples)
    )
    
    print(f"Generating sample text via Ollama '{model}'...")
    try:
        response = ollama.generate(
            model=model,
            prompt=flat_prompt,
            options={"temperature": 0.2, "seed": 42}
        )
        generated_text = response["response"]
        print(f"\nGenerated Text:\n{generated_text}\n")
    except Exception as e:
        print(f"Error during Ollama generation: {e}")
        return

    # Extract triples
    print("Extracting triples from generated text...")
    extracted_triples = extract_triples_from_text(generated_text, model)
    print(f"Extracted {len(extracted_triples)} triples.")
    for et in extracted_triples:
        print(f"  {et}")

    # Sweep thresholds
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    results = []
    
    print("\nSweeping thresholds...")
    for t in thresholds:
        evaluator = SMGAEvaluator(embedding_model=embedding_model, similarity_threshold=t)
        res = evaluator.evaluate(input_triples, extracted_triples)
        
        results.append({
            "threshold": t,
            "sdr": res["semantic_divergence_rate"],
            "coverage": res["coverage"],
            "omissions_count": res["omissions_count"],
            "hallucinations_count": res["hallucinations_count"]
        })
        
        print(f"Threshold: {t:.2f} -> SDR: {res['semantic_divergence_rate']:.4f}, Coverage: {res['coverage']:.4f}, "
              f"Omissions: {res['omissions_count']}, Hallucinations: {res['hallucinations_count']}")
        
    # Save sweep results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "threshold_sweep.json"), "w", encoding="utf-8") as f:
        json.dump({
            "task_id": selected_task["id"],
            "input_triples": input_triples,
            "extracted_triples": extracted_triples,
            "sweep": results
        }, f, indent=2, ensure_ascii=False)
        
    print(f"\nResults exported to {os.path.join(results_dir, 'threshold_sweep.json')}")

if __name__ == "__main__":
    run_threshold_sweep()
