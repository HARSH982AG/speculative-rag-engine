import numpy as np
from typing import List, Dict, Any

class SIMDRetriever:
    """Vectorized in-memory dense embedding retriever with threshold pruning."""
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.doc_ids: List[str] = []
        self.texts: List[str] = []
        self.embeddings: np.ndarray = np.empty((0, dim), dtype=np.float32)

    def add_documents(self, docs: List[Dict[str, Any]]):
        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        vectors = np.array([d["embedding"] for d in docs], dtype=np.float32)
        # Normalize vectors for fast cosine distance via dot product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = vectors / norms

        self.doc_ids.extend(ids)
        self.texts.extend(texts)
        if len(self.embeddings) == 0:
            self.embeddings = normalized
        else:
            self.embeddings = np.vstack([self.embeddings, normalized])

    def query(self, query_vec: np.ndarray, top_k: int = 3, threshold: float = 0.65) -> List[Dict[str, Any]]:
        if len(self.embeddings) == 0:
            return []
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        scores = np.dot(self.embeddings, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= threshold:
                results.append({
                    "id": self.doc_ids[idx],
                    "text": self.texts[idx],
                    "score": round(score, 4)
                })
        return results
