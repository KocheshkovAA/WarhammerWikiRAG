# app/core/lightrag_client.py
import httpx
from app.core.config import settings
import logging
from langfuse import observe

logger = logging.getLogger(__name__)

class LightRAGClient:
    def __init__(self):
        self.base_url = settings.LIGHTRAG_BASE_URL  # добавь в settings: "http://lightrag:9621"
        self.timeout = httpx.Timeout(90.0, connect=10.0)

    @observe(name="LightRAG Query")
    async def query(self, question: str, mode: str = "mix") -> dict:
        """
        mode может быть: naive / local / global / hybrid / mix
        """
        payload = {
            "query": question,
            "mode": mode,
            "enable_rerank": True,      # используем твой vLLM reranker
            "top_k": 40,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/query",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
                
                return {
                    "answer": data.get("answer", data.get("response", "Нет ответа")),
                    "sources": data.get("sources", []) or data.get("context", []),
                    "mode": f"lightrag-{mode}",
                }
            except Exception as e:
                logger.error(f"LightRAG error: {e}")
                return {"answer": "LightRAG временно недоступен", "sources": [], "mode": "lightrag-error"}