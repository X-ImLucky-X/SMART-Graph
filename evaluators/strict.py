import numpy as np

class StrictEvaluator:
    @staticmethod
    def evaluate(input_triples: list[tuple[str, str, str]], 
                 extracted_triples: list[tuple[str, str, str]], 
                 sos_scores: list[float] = None) -> dict:
        """
        Performs exact string matching evaluation on normalized triples.
        """
        # Normalize tuples: strip whitespace, convert to lowercase
        set_input = {tuple(str(item).strip().lower() for item in t) for t in input_triples}
        set_extracted = {tuple(str(item).strip().lower() for item in t) for t in extracted_triples}
        
        omissions = set_input - set_extracted
        hallucinations = set_extracted - set_input
        
        len_input = len(set_input) if len(set_input) > 0 else 1
        
        # Calculate standard metrics
        sdr = (len(omissions) + len(hallucinations)) / len_input
        coverage = len(set_input - omissions) / len_input
        
        # Calculate Attention Protection Score (APS) using top 20% highest SOS triples
        aps = 1.0
        high_risk_omitted = 0
        high_risk_total = 0
        
        if sos_scores and len(input_triples) > 0:
            # Pair input triples with their SOS scores
            paired = list(zip(input_triples, sos_scores))
            # Sort descending by SOS
            paired.sort(key=lambda x: x[1], reverse=True)
            
            # Top 20% are high-risk (at least 1 if input_triples is not empty)
            k_20 = max(1, len(input_triples) // 5)
            high_risk_triples = [t for t, _ in paired[:k_20]]
            high_risk_total = len(high_risk_triples)
            
            # Normalize high-risk triples for matching
            norm_high_risk = {tuple(str(item).strip().lower() for item in t) for t in high_risk_triples}
            
            # Check how many of these normalized high-risk triples are in omissions
            high_risk_omitted = len(norm_high_risk.intersection(omissions))
            aps = (high_risk_total - high_risk_omitted) / high_risk_total if high_risk_total > 0 else 1.0
            
        return {
            "semantic_divergence_rate": round(sdr, 4),
            "coverage": round(coverage, 4),
            "attention_protection_score": round(aps, 4),
            "omissions_count": len(omissions),
            "hallucinations_count": len(hallucinations),
            "omitted_triples": [list(t) for t in omissions],
            "hallucinated_triples": [list(t) for t in hallucinations]
        }

if __name__ == "__main__":
    # Test strict evaluator
    inputs = [("Apple", "foundedBy", "Steve Jobs"), ("Apple", "foundedIn", "1976")]
    extractions = [("apple", "foundedby", "steve jobs")]
    scores = [0.2, 0.8]  # "foundedIn" has higher SOS
    res = StrictEvaluator.evaluate(inputs, extractions, scores)
    print(res)
