import os
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.vector_engine import VectorSpaceProfiler

class SMGAEvaluator:
    def __init__(self, embedding_model="nomic-embed-text:latest", similarity_threshold=0.75):
        self.profiler = VectorSpaceProfiler(model_name=embedding_model)
        self.similarity_threshold = similarity_threshold

    def evaluate(self, input_triples: list[tuple[str, str, str]], 
                 extracted_triples: list[tuple[str, str, str]], 
                 sos_scores: list[float] = None) -> dict:
        """
        Performs Soft-Match Graph Alignment (SMGA) using Hungarian bipartite matching on 
        cosine similarities of triple embeddings.
        """
        if not input_triples:
            return {
                "semantic_divergence_rate": 0.0,
                "coverage": 1.0,
                "attention_protection_score": 1.0,
                "omissions_count": 0,
                "hallucinations_count": len(extracted_triples),
                "omitted_triples": [],
                "hallucinated_triples": [[str(item) for item in t] for t in extracted_triples],
                "matched_pairs": []
            }

        if not extracted_triples:
            # All input triples are omitted
            aps = 0.0 if sos_scores else 1.0
            return {
                "semantic_divergence_rate": 1.0,
                "coverage": 0.0,
                "attention_protection_score": aps,
                "omissions_count": len(input_triples),
                "hallucinations_count": 0,
                "omitted_triples": [[str(item) for item in t] for t in input_triples],
                "hallucinated_triples": [],
                "matched_pairs": []
            }

        # Convert triples to strings and fetch embeddings
        str_in = [f"{s} {p} {o}" for s, p, o in input_triples]
        str_ex = [f"{s} {p} {o}" for s, p, o in extracted_triples]

        emb_in = [self.profiler.get_embedding(text) for text in str_in]
        emb_ex = [self.profiler.get_embedding(text) for text in str_ex]

        # Calculate cosine similarity cost matrix
        cost_matrix = np.zeros((len(input_triples), len(extracted_triples)))
        for i in range(len(input_triples)):
            t_norm = np.linalg.norm(emb_in[i])
            for j in range(len(extracted_triples)):
                ex_norm = np.linalg.norm(emb_ex[j])
                if t_norm > 0 and ex_norm > 0:
                    cost_matrix[i][j] = np.dot(emb_in[i], emb_ex[j]) / (t_norm * ex_norm)
                else:
                    cost_matrix[i][j] = 0.0

        # Apply Hungarian algorithm for optimal 1-to-1 alignments (maximizing similarity)
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)

        matched_inputs = set()
        matched_extractions = set()
        matched_pairs = []

        for r, c in zip(row_ind, col_ind):
            similarity = cost_matrix[r][c]
            if similarity >= self.similarity_threshold:
                matched_inputs.add(r)
                matched_extractions.add(c)
                matched_pairs.append({
                    "input_triple": input_triples[r],
                    "extracted_triple": extracted_triples[c],
                    "similarity": round(float(similarity), 4)
                })

        # Omissions are unmatched inputs, Hallucinations are unmatched extractions
        omitted_indices = set(range(len(input_triples))) - matched_inputs
        hallucinated_indices = set(range(len(extracted_triples))) - matched_extractions

        omitted_triples = [input_triples[i] for i in omitted_indices]
        hallucinated_triples = [extracted_triples[j] for j in hallucinated_indices]

        len_input = len(input_triples)

        # SDR and Coverage metrics
        sdr = (len(omitted_triples) + len(hallucinated_triples)) / len_input
        coverage = len(matched_inputs) / len_input

        # Attention Protection Score (APS) using top 20% highest SOS triples
        aps = 1.0
        if sos_scores:
            paired = list(zip(range(len(input_triples)), sos_scores))
            paired.sort(key=lambda x: x[1], reverse=True)
            
            k_20 = max(1, len(input_triples) // 5)
            high_risk_indices = {idx for idx, _ in paired[:k_20]}
            
            high_risk_total = len(high_risk_indices)
            high_risk_matched = len(high_risk_indices.intersection(matched_inputs))
            aps = high_risk_matched / high_risk_total if high_risk_total > 0 else 1.0

        return {
            "semantic_divergence_rate": round(sdr, 4),
            "coverage": round(coverage, 4),
            "attention_protection_score": round(aps, 4),
            "omissions_count": len(omitted_triples),
            "hallucinations_count": len(hallucinated_triples),
            "omitted_triples": [[str(item) for item in t] for t in omitted_triples],
            "hallucinated_triples": [[str(item) for item in t] for t in hallucinated_triples],
            "matched_pairs": matched_pairs
        }

if __name__ == "__main__":
    # Test SMGA
    evaluator = SMGAEvaluator()
    inputs = [("Apple Inc", "led by", "Tim Cook"), ("Apple", "makes", "iPhones")]
    extractions = [("Apple Company", "is run by", "Mr Tim Cook"), ("Google", "makes", "Android")]
    scores = [0.9, 0.1]
    res = evaluator.evaluate(inputs, extractions, scores)
    print(res)
