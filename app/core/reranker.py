import httpx
import asyncio
import math
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.core.config import settings
from langfuse import observe, get_client

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

class Reranker:
    def __init__(self):
        self.enabled = settings.RERANKER_ENABLED
        self.vllm_url = f"{settings.RERANKER_URL.rstrip('/')}/v1/rerank" if settings.RERANKER_URL else ""
        self.top_k = settings.RERANKER_TOP_K or 5
        self.batch_size = settings.RERANKER_BATCH_SIZE or 32
        self.model_name = "bge-reranker-v2-m3"
        self.client = httpx.AsyncClient(timeout=60.0)

    @observe(name="Reranker: Process Documents")
    async def rerank_documents(self, query: str, docs: List[Any]) -> List[Any]:
        """
        Основной метод: принимает LangChain Documents, возвращает топ-N релевантных.
        """
        if not self.enabled or not self.vllm_url or not docs:
            return docs[:self.top_k]

        try:
            texts = [doc.page_content for doc in docs]
            scores = await self._get_scores(query, texts)

            # Объединяем скоры с документами и сортируем
            scored_docs = sorted(
                zip(scores, docs),
                key=lambda x: x[0],
                reverse=True
            )
            
            final_docs = [doc for score, doc in scored_docs[:self.top_k]]

            # Логируем метаданные в текущий trace Langfuse
            get_client().update_current_trace(
                metadata={
                    "rerank_input_count": len(docs),
                    "rerank_output_count": len(final_docs),
                    "rerank_max_score": max(scores) if scores else 0,
                }
            )
            return final_docs

        except Exception as e:
            print(f"Reranker failed: {e}. Falling back to original order.")
            return docs[:self.top_k]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ReadTimeout))
    )
    async def _get_scores(self, query: str, texts: List[str]) -> List[float]:
        """Внутренний метод только для сетевого взаимодействия и получения сырых чисел."""
        all_scores = [0.0] * len(texts)

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            response = await self.client.post(
                self.vllm_url,
                json={
                    "model": self.model_name,
                    "query": query,
                    "documents": batch_texts,
                }
            )
            response.raise_for_status()
            data = response.json()

            # Извлекаем результаты, сохраняя порядок
            results = sorted(data.get("results", []), key=lambda x: x["index"])
            for idx, res in enumerate(results):
                # Применяем sigmoid для нормализации
                all_scores[i + idx] = 1 / (1 + math.exp(-res["relevance_score"]))

        return all_scores

reranker = Reranker()