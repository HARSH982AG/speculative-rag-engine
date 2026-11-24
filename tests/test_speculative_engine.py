import pytest
from src.kv_cache import RadixKVCache
from src.retriever import SIMDRetriever
from src.router import SpeculativeRAGRouter
import numpy as np

def test_radix_kv_cache():
    cache = RadixKVCache()
    tokens = [101, 2054, 2003, 1037]
    cache.insert(tokens, {"tensor": "kv_block_01"})
    
    matched, ref = cache.match_prefix(tokens)
    assert matched == 4
    assert ref["tensor"] == "kv_block_01"
    assert cache.hits == 1

def test_simd_retriever():
    retriever = SIMDRetriever(dim=4)
    docs = [
        {"id": "doc1", "text": "Speculative decoding", "embedding": [1.0, 0.0, 0.0, 0.0]},
        {"id": "doc2", "text": "Radix trie cache", "embedding": [0.0, 1.0, 0.0, 0.0]}
    ]
    retriever.add_documents(docs)
    res = retriever.query(np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32), top_k=1)
    assert len(res) == 1
    assert res[0]["id"] == "doc1"

@pytest.mark.asyncio
async def test_router_execution():
    router = SpeculativeRAGRouter()
    res = await router.execute_speculative_turn("What is speculative RAG?", "sess_test")
    assert "telemetry" in res
    assert res["telemetry"]["total_latency_ms"] > 0
    assert res["telemetry"]["acceptance_rate"] > 0
