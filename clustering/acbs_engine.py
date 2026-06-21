import os
import sys
import random
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.vector_engine import VectorSpaceProfiler

class ACBSEngine:
    def __init__(self, max_size=8, max_vulnerability_budget=1.5, embedding_model="nomic-embed-text:latest"):
        self.max_size = max_size
        self.budget = max_vulnerability_budget
        self.profiler = VectorSpaceProfiler(model_name=embedding_model)

    def compute_sos(self, triples: list[tuple[str, str, str]]) -> list[float]:
        return self.profiler.calculate_sos(triples)

    def serialize_triples(self, triples: list[tuple[str, str, str]], mode: str, seed: int = 42) -> list[list[tuple[str, str, str]]]:
        """
        Unified serialization method. Returns a list of partitioned clusters of triples
        depending on the mode selected.
        Modes: 'baseline_a' (flat), 'baseline_b' (flat random), 'baseline_c' (flat SOS desc),
               'baseline_d' (modularity communities), 'smart_graph' (ACBS).
        """
        if not triples:
            return []

        # Ensure seed is set for deterministic outputs where random is involved
        random.seed(seed)

        if mode == "baseline_a":
            # Flat sequential
            return [triples]

        elif mode == "baseline_b":
            # Flat random shuffling
            shuffled = list(triples)
            random.shuffle(shuffled)
            return [shuffled]

        elif mode == "baseline_c":
            # Flat SOS descending
            sos_scores = self.profiler.calculate_sos(triples)
            scored = list(zip(triples, sos_scores))
            # Sort descending by SOS
            scored.sort(key=lambda x: x[1], reverse=True)
            return [[t for t, _ in scored]]

        elif mode == "baseline_d":
            # Modularity community detection using networkx
            # Build graph where nodes are triple indices, edges share subject/object
            G = nx.Graph()
            for i, t1 in enumerate(triples):
                G.add_node(i, triple=t1)
                for j in range(i + 1, len(triples)):
                    t2 = triples[j]
                    if t1[0] == t2[0] or t1[2] == t2[2] or t1[0] == t2[2] or t1[2] == t2[0]:
                        G.add_edge(i, j)

            # Compute modularity communities
            communities = list(greedy_modularity_communities(G))
            
            clusters = []
            for comm in communities:
                # Sort indices to maintain relative original order
                sorted_idx = sorted(list(comm))
                clusters.append([triples[idx] for idx in sorted_idx])
            return clusters

        elif mode == "smart_graph" or mode == "acbs":
            # Proposed ACBS framework
            sos_scores = self.profiler.calculate_sos(triples)
            
            # Build graph where nodes are triple indices, edges share subject/object
            G = nx.Graph()
            for i, t1 in enumerate(triples):
                G.add_node(i, triple=t1, score=sos_scores[i])
                for j in range(i + 1, len(triples)):
                    t2 = triples[j]
                    if t1[0] == t2[0] or t1[2] == t2[2] or t1[0] == t2[2] or t1[2] == t2[0]:
                        G.add_edge(i, j)

            unassigned = set(G.nodes())
            clusters = []

            while unassigned:
                # Seed with unassigned node that has the highest degree, secondary sort on highest SOS score
                seed_node = max(unassigned, key=lambda n: (G.degree(n), G.nodes[n]['score']))
                current_cluster = [(seed_node, G.nodes[seed_node]['score'])]
                unassigned.remove(seed_node)
                current_sos_sum = G.nodes[seed_node]['score']

                # Expand cluster with direct neighbors
                neighbors = sorted(
                    [n for n in G.neighbors(seed_node) if n in unassigned],
                    key=lambda n: (G.degree(n), G.nodes[n]['score']),
                    reverse=True
                )

                for neighbor in neighbors:
                    if len(current_cluster) >= self.max_size:
                        break
                    neighbor_score = G.nodes[neighbor]['score']
                    if current_sos_sum + neighbor_score > self.budget:
                        break  # Vulnerability budget constraint met
                    
                    current_cluster.append((neighbor, neighbor_score))
                    unassigned.remove(neighbor)
                    current_sos_sum += neighbor_score

                # Fallback: check neighbors of neighbors if cluster is not full and budget allows
                if len(current_cluster) < self.max_size and unassigned:
                    extended_candidates = []
                    for node_idx, _ in current_cluster:
                        extended_candidates.extend([n for n in G.neighbors(node_idx) if n in unassigned])
                    
                    extended_candidates = sorted(
                        list(set(extended_candidates)),
                        key=lambda n: (G.degree(n), G.nodes[n]['score']),
                        reverse=True
                    )

                    for ext_n in extended_candidates:
                        if len(current_cluster) >= self.max_size:
                            break
                        ext_score = G.nodes[ext_n]['score']
                        if current_sos_sum + ext_score > self.budget:
                            break
                        if ext_n in unassigned:
                            current_cluster.append((ext_n, ext_score))
                            unassigned.remove(ext_n)
                            current_sos_sum += ext_score

                # Arrange the cluster in a U-shape context sequence
                u_arranged_triples = self._optimize_u_shape(current_cluster, triples)
                clusters.append(u_arranged_triples)

            return clusters
        else:
            raise ValueError(f"Unknown serialization mode: {mode}")

    def _optimize_u_shape(self, cluster_with_scores: list[tuple[int, float]], triples: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
        """
        Sorts the triples in the cluster such that the highest SOS outliers
        are mapped to the boundaries (primacy/recency) of the prompt window.
        """
        # Sort items by SOS score descending
        sorted_items = sorted(cluster_with_scores, key=lambda x: x[1], reverse=True)
        u_arranged = [None] * len(sorted_items)
        
        left, right = 0, len(sorted_items) - 1
        for idx, (triple_idx, score) in enumerate(sorted_items):
            actual_triple = triples[triple_idx]
            if idx % 2 == 0:
                u_arranged[left] = actual_triple
                left += 1
            else:
                u_arranged[right] = actual_triple
                right -= 1
        return u_arranged

if __name__ == "__main__":
    engine = ACBSEngine(max_size=3, max_vulnerability_budget=1.2)
    test_triples = [
        ("Apple", "released", "iPhone 15"),
        ("Apple", "is led by", "Tim Cook"),
        ("The moon", "orbits", "Earth"),
        ("Tim Cook", "education", "Auburn University"),
        ("Auburn University", "city", "Auburn")
    ]
    
    print("--- ACBS Serialization ---")
    clusters = engine.serialize_triples(test_triples, "smart_graph")
    for idx, c in enumerate(clusters):
        print(f"Cluster {idx+1}: {c}")

    print("\n--- Community Detection ---")
    clusters_comm = engine.serialize_triples(test_triples, "baseline_d")
    for idx, c in enumerate(clusters_comm):
        print(f"Cluster {idx+1}: {c}")
