import os
import sys
import json
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import ollama

# Path injection
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from clustering.acbs_engine import ACBSEngine
from evaluators.smga import SMGAEvaluator
from experiments.threshold_sweep import extract_triples_from_text
from experiments.sos_correlation import run_correlation_study
from experiments.ablation import run_ablation_study
from run_benchmark import run_benchmark

app = FastAPI(title="SMART-Graph Academic Prototype")

# Request Model
class ProcessRequest(BaseModel):
    triples: list[tuple[str, str, str]]
    model: str = "qwen2.5:7b"
    max_cluster_size: int = 8
    vulnerability_budget: float = 1.5
    similarity_threshold: float = 0.75

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_path = os.path.join("static", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Dashboard index.html not found.")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/models")
async def get_models():
    """Queries local Ollama tags to find all available LLM options."""
    try:
        tags = ollama.list()
        models = [m["model"] for m in tags.get("models", [])]
        if not models:
            models = ["qwen2.5:7b"]  # Fallback baseline
        return {"models": models}
    except Exception as e:
        # Fallback if Ollama is not accessible
        return {"models": ["qwen2.5:7b", "llama3:latest"], "warning": str(e)}

@app.post("/api/process")
async def process_graph(req: ProcessRequest):
    """
    Executes the entire SMART-Graph pipeline on an input set of triples.
    """
    try:
        # 1. Initialize engine and evaluators
        engine = ACBSEngine(
            max_size=req.max_cluster_size,
            max_vulnerability_budget=req.vulnerability_budget,
            embedding_model="nomic-embed-text:latest"
        )
        smga_evaluator = SMGAEvaluator(
            embedding_model="nomic-embed-text:latest",
            similarity_threshold=req.similarity_threshold
        )
        
        # 2. Outlier SOS profiling
        sos_scores = engine.compute_sos(req.triples)
        
        # 3. Partitioning
        partitions = engine.serialize_triples(req.triples, "smart_graph", seed=42)
        
        # 4. Paragraph generation
        paragraphs = []
        generation_calls = 0
        for idx, cluster in enumerate(partitions):
            prompt = (
                "Write a brief, fluent, and highly factual paragraph describing the following details:\n\n" +
                "\n".join(f"- {s} {p} {o}" for s, p, o in cluster)
            )
            response = ollama.generate(
                model=req.model,
                prompt=prompt,
                options={"temperature": 0.2, "seed": 42, "num_ctx": 4096}
            )
            paragraphs.append(response["response"])
            generation_calls += 1
            
        # 5. Merging / Synthesizing
        if len(paragraphs) > 1:
            synthesis_prompt = (
                "Synthesize the following paragraphs into a single, cohesive, fluent document. "
                "Keep all the facts and details exactly as they are, but make the transitions smooth:\n\n" +
                "\n\n".join(paragraphs)
            )
            response = ollama.generate(
                model=req.model,
                prompt=synthesis_prompt,
                options={"temperature": 0.2, "seed": 42, "num_ctx": 4096}
            )
            synthesized_text = response["response"]
            generation_calls += 1
        else:
            synthesized_text = paragraphs[0] if paragraphs else ""
            
        # 6. Extract triples
        extracted_triples = extract_triples_from_text(synthesized_text, req.model)
        generation_calls += 1
        
        # 7. Evaluate via SMGA
        metrics = smga_evaluator.evaluate(req.triples, extracted_triples, sos_scores)
        cost_stats = engine.profiler.get_stats()
        
        return {
            "triples": req.triples,
            "sos_scores": sos_scores,
            "partitions": partitions,
            "synthesized_text": synthesized_text,
            "extracted_triples": extracted_triples,
            "alignments": metrics,
            "metrics": {
                "semantic_divergence_rate": metrics["semantic_divergence_rate"],
                "coverage": metrics["coverage"],
                "attention_protection_score": metrics["attention_protection_score"],
                "omissions_count": metrics["omissions_count"],
                "hallucinations_count": metrics["hallucinations_count"]
            },
            "cost": {
                "embedding_calls": cost_stats["embedding_calls"],
                "cache_hits": cost_stats["cache_hits"],
                "cache_hit_rate": cost_stats["cache_hit_rate"],
                "generation_calls": generation_calls
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/correlation")
async def trigger_correlation_study():
    """Runs the SOS Omission Correlation Study."""
    try:
        run_correlation_study()
        results_path = os.path.join("results", "sos_correlation.json")
        if os.path.exists(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"status": "success", "message": "Study completed but results file not found."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ablation")
async def trigger_ablation_study():
    """Runs the Ablation Study."""
    try:
        run_ablation_study()
        results_path = os.path.join("results", "ablation_study.json")
        if os.path.exists(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"status": "success", "message": "Ablation run complete."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/benchmark")
async def trigger_benchmark():
    """Runs the Benchmark Suite."""
    try:
        run_benchmark()
        results_path = os.path.join("results", "benchmark.csv")
        # Find latest run log JSON
        run_dir = os.path.join("experiments", "runs")
        existing_runs = sorted([f for f in os.listdir(run_dir) if f.startswith("run_") and f.endswith(".json")])
        if existing_runs:
            with open(os.path.join(run_dir, existing_runs[-1]), "r", encoding="utf-8") as f:
                return json.load(f)
        return {"status": "success", "message": "Benchmark run complete."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
