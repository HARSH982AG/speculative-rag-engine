from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from .router import SpeculativeRAGRouter

app = FastAPI(
    title="Speculative RAG Engine API",
    version="1.0.0",
    description="Sub-30ms speculative RAG inference router with Radix-Trie KV prefix reuse."
)

router = SpeculativeRAGRouter()

class QueryRequest(BaseModel):
    query: str
    session_id: str = "sess_default"
    draft_k: int = 4

@app.get("/health")
def health():
    return {"status": "HEALTHY", "engine": "SpeculativeRAGEngine", "version": "1.0.0"}

@app.post("/v1/speculative/chat")
async def chat(req: QueryRequest):
    return await router.execute_speculative_turn(
        query=req.query,
        session_id=req.session_id,
        draft_k=req.draft_k
    )

@app.get("/v1/cache/stats")
def cache_stats():
    return router.kv_cache.get_stats()

@app.websocket("/v1/speculative/stream")
async def stream_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            res = await router.execute_speculative_turn(data, "ws_session")
            await websocket.send_json(res)
    except Exception:
        await websocket.close()
