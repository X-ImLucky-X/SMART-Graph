import os
import json
import re
import numpy as np
import ollama

class VectorSpaceProfiler:
    def __init__(self, model_name: str = "nomic-embed-text:latest", cache_dir: str = "cache"):
        self.model_name = model_name
        self.cache_path = os.path.join(cache_dir, "embeddings.json")
        self.cache = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        self._load_cache()
        
        # Keep track of calls for reporting
        self.total_embedding_calls = 0
        self.cache_hits = 0

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load embedding cache: {e}. Starting fresh.")
                self.cache = {}

    def _save_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save embedding cache: {e}")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Retrieves the embedding vector for a given string, check the persistent cache first.
        """
        # Normalize text key
        key = " ".join(text.strip().lower().split())
        
        if key in self.cache:
            self.cache_hits += 1
            return np.array(self.cache[key])
        
        self.total_embedding_calls += 1
        try:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            embedding = response["embedding"]
            # Store in cache
            self.cache[key] = embedding
            self._save_cache()
            return np.array(embedding)
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch embedding from local Ollama model '{self.model_name}'. "
                f"Is Ollama running? Error: {e}"
            )

    def calculate_sos(self, triples: list[tuple[str, str, str]]) -> list[float]:
        """
        Computes the Semantic Outlier Score (SOS) for a set of triples.
        Formula: SOS = (1 - CosineSimilarity(t_i, C)) * (1 + delta)
        where C is the centroid, and delta = 0.5 if the triple text has numbers/dates.
        """
        if not triples:
            return []
            
        # Convert triples to unified text representations
        triple_texts = [f"{s} {p} {o}" for s, p, o in triples]
        embeddings = [self.get_embedding(text) for text in triple_texts]
        
        # Calculate global centroid C
        centroid = np.mean(embeddings, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        
        sos_scores = []
        for i, emb in enumerate(embeddings):
            # Compute cosine similarity
            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0 or centroid_norm == 0:
                cos_sim = 0.0
            else:
                cos_sim = np.dot(emb, centroid) / (emb_norm * centroid_norm)
                
            # Cosine distance
            cos_dist = 1.0 - cos_sim
            
            # Numeric/date penalty multiplier (delta)
            has_digits = bool(re.search(r'\d', triple_texts[i]))
            multiplier = 1.5 if has_digits else 1.0
            
            # Calculate final SOS
            sos_scores.append(float(cos_dist * multiplier))
            
        return sos_scores

    def get_stats(self) -> dict:
        """Returns cache stats and API call cost tracking."""
        total = self.total_embedding_calls + self.cache_hits
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        return {
            "embedding_calls": self.total_embedding_calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(hit_rate, 4)
        }

if __name__ == "__main__":
    # Quick self-test
    profiler = VectorSpaceProfiler()
    test_triples = [
        ("Apple", "released", "iPhone 15"),
        ("Apple", "is led by", "Tim Cook"),
        ("The moon", "orbits", "Earth")  # Outlier
    ]
    scores = profiler.calculate_sos(test_triples)
    for t, s in zip(test_triples, scores):
        print(f"Triple: {t} -> SOS: {s:.4f}")
    print(profiler.get_stats())
