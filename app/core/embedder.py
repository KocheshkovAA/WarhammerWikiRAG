from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector
import httpx
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.core.config import settings
from langfuse import observe, get_client

class HybridEmbedder:
    def __init__(self):
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        self.tei_url = f"{settings.TEI_URL}/embed"
        self.query_instruction = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "

    @observe(name="Embedder: Sparse")
    def get_sparse_embeddings(self, texts: list[str]) -> list[SparseVector]:
        lf = get_client()
        lf.update_current_trace(input={"num_texts": len(texts)})
        embeddings = list(self.sparse_model.embed(texts))
        return [
            SparseVector(
                indices=e.indices.tolist(), 
                values=e.values.tolist()
            ) for e in embeddings
        ]

    @observe(name="Embedder: Dense (Async)")
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectError))
    )
    async def get_dense_embeddings(self, texts: list[str], is_query: bool = True) -> list[list[float]]:
        lf = get_client()
        lf.update_current_trace(
            input={"num_texts": len(texts), "is_query": is_query},
            metadata={"model_url": self.tei_url}
        )

        async with httpx.AsyncClient(timeout=300.0) as client:
            processed_texts = [
                f"{self.query_instruction}{t}" if is_query else t 
                for t in texts
            ]

            response = await client.post(self.tei_url, json={"inputs": processed_texts})
            
            if response.status_code == 429:
                raise httpx.HTTPStatusError("Rate limit", request=response.request, response=response)
                
            response.raise_for_status()
            return response.json()

    @observe(name="Embedder: Dense (Sync)")
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectError))
    )
    def get_dense_embeddings_sync(self, texts: list[str], is_query: bool = True) -> list[list[float]]:
        lf = get_client()
        lf.update_current_trace(
            input={"num_texts": len(texts), "is_query": is_query},
            metadata={"model_url": self.tei_url}
        )
        
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

embedder = HybridEmbedder()