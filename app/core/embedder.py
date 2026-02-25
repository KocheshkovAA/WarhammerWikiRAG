from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector
import httpx
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.core.config import settings

class HybridEmbedder:
    def __init__(self):
        # Локальная модель, ретраи не нужны
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        self.tei_url = f"{settings.TEI_URL}/embed"
        self.query_instruction = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "

    def get_sparse_embeddings(self, texts: list[str]) -> list[SparseVector]:
        """Генерирует разреженные векторы локально через CPU"""
        embeddings = list(self.sparse_model.embed(texts))
        return [
            SparseVector(
                indices=e.indices.tolist(), 
                values=e.values.tolist()
            ) for e in embeddings
        ]

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectError))
    )
    async def get_dense_embeddings(self, texts: list[str], is_query: bool = True) -> list[list[float]]:
        """Асинхронный запрос к TEI с ретраями"""
        async with httpx.AsyncClient(timeout=300.0) as client:
            processed_texts = [
                f"{self.query_instruction}{t}" if is_query else t 
                for t in texts
            ]

            response = await client.post(self.tei_url, json={"inputs": processed_texts})
            
            if response.status_code == 429:
                # В асинхронном контексте лучше бросать исключение, 
                # чтобы tenacity его обработал, либо делать await sleep
                raise httpx.HTTPStatusError("Rate limit", request=response.request, response=response)
                
            response.raise_for_status()
            return response.json()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectError))
    )
    def get_dense_embeddings_sync(self, texts: list[str], is_query: bool = True) -> list[list[float]]:
        """Синхронный запрос к TEI с ретраями (для скрипта миграции)"""
        with httpx.Client(timeout=300.0) as client:
            processed_texts = [
                f"{self.query_instruction}{t}" if is_query else t 
                for t in texts
            ]

            response = client.post(self.tei_url, json={"inputs": processed_texts})
            
            if response.status_code == 429:
                print("⏳ TEI Rate Limit (429). Ждем...")
                time.sleep(5)
                raise httpx.HTTPStatusError("Rate limit", request=response.request, response=response)
                
            response.raise_for_status()
            return response.json()

# Создаем один экземпляр
embedder = HybridEmbedder()