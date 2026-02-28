from typing import List, Dict, Any
from qdrant_client import QdrantClient, models
from langfuse import observe, get_client, propagate_attributes

from app.core.config import settings
from app.core.embedder import embedder


class RetrievalMetrics:
    def __init__(self):
        self.num_results: int = 0
        self.avg_score: float = 0.0
        self.max_score: float = 0.0
        self.scores: List[float] = []

    def update_from_points(self, points: List[models.ScoredPoint]) -> None:
        if not points:
            return
            
        self.scores = [p.score for p in points if p.score is not None]
        self.num_results = len(self.scores)
        
        if self.scores:
            self.avg_score = sum(self.scores) / len(self.scores)
            self.max_score = max(self.scores)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_results": self.num_results,
            "avg_score": round(self.avg_score, 4),
            "max_score": round(self.max_score, 4),
        }


class VectorStore:
    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.collection_name = settings.COLLECTION_NAME
        self.metrics = RetrievalMetrics()
        self.langfuse = get_client()

    @observe(name="VectorStore → Hybrid Search", capture_input=False, capture_output=False)
    def hybrid_search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        with propagate_attributes(tags=["retrieval", "hybrid", "qdrant"]):
            self.langfuse.update_current_span(
                input={
                    "query_length": len(query),
                    "query_preview": query[:80] + "…" if len(query) > 80 else query,
                    "limit": limit,
                }
            )

            dense_vector = embedder.get_dense_embeddings_sync([query])[0]
            sparse_vector = embedder.get_sparse_embeddings([query])[0]

            search_result = self._perform_hybrid_query(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                limit=limit,
            )

            self.metrics.update_from_points(search_result.points)
            self._log_retrieval_metrics()

            return [point.payload for point in search_result.points if point.payload]

    def _perform_hybrid_query(
        self,
        dense_vector: List[float],
        sparse_vector: models.SparseVector,
        limit: int,
    ) -> models.QueryResponse:
        prefetch = [
            models.Prefetch(
                query=dense_vector,
                using="text-dense",
                limit=limit * 3,
            ),
            models.Prefetch(
                query=sparse_vector,
                using="text-sparse",
                limit=limit * 3,
            ),
        ]

        return self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

    def _log_retrieval_metrics(self) -> None:
        metrics_dict = self.metrics.to_dict()

        self.langfuse.update_current_span(metadata=metrics_dict)

        if self.metrics.max_score > 0:
            self.langfuse.score_current_span(
                name="retrieval_max_score",
                value=round(self.metrics.max_score, 4),
                comment="Макс. релевантность",
            )

        if self.metrics.num_results > 0:
            self.langfuse.score_current_span(
                name="retrieval_avg_score",
                value=round(self.metrics.avg_score, 4),
                comment="Средняя релевантность документов",
            )

vector_store = VectorStore()