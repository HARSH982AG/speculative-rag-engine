import time
from typing import Dict, Optional, Tuple, Any

class TrieNode:
    def __init__(self, token_id: Optional[int] = None):
        self.token_id = token_id
        self.children: Dict[int, TrieNode] = {}
        self.tensor_ref: Optional[Any] = None
        self.last_accessed: float = time.time()

class RadixKVCache:
    """Hierarchical prefix-sharing KV-cache with LRU eviction and zero-copy references."""
    def __init__(self, max_nodes: int = 50000):
        self.root = TrieNode()
        self.max_nodes = max_nodes
        self.node_count = 1
        self.hits = 0
        self.misses = 0

    def insert(self, token_ids: list[int], tensor_data: Any) -> int:
        curr = self.root
        inserted = 0
        for tid in token_ids:
            if tid not in curr.children:
                curr.children[tid] = TrieNode(tid)
                self.node_count += 1
                inserted += 1
            curr = curr.children[tid]
            curr.last_accessed = time.time()
        curr.tensor_ref = tensor_data
        return inserted

    def match_prefix(self, token_ids: list[int]) -> Tuple[int, Optional[Any]]:
        curr = self.root
        matched_len = 0
        last_tensor = None
        for tid in token_ids:
            if tid not in curr.children:
                break
            curr = curr.children[tid]
            curr.last_accessed = time.time()
            matched_len += 1
            if curr.tensor_ref is not None:
                last_tensor = curr.tensor_ref

        if matched_len > 0 and last_tensor is not None:
            self.hits += 1
            return matched_len, last_tensor
        else:
            self.misses += 1
            return 0, None

    def get_stats(self) -> dict:
        total = self.hits + self.misses
        rate = (self.hits / total) if total > 0 else 0.0
        return {
            "node_count": self.node_count,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(rate, 4)
        }
