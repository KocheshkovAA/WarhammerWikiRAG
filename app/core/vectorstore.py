from qdrant_client import QdrantClient, models
from app.core.config import settings
from app.core.embedder import embedder

class WarhammerVectorStore:
    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.collection_name = settings.COLLECTION_NAME

    def hl_search(self, query: str, limit: int = 5):
        """Гибридный поиск: Dense (смысл) + Sparse (ключевые слова)"""
        
        # 1. Генерируем оба вектора для вопроса
        dense_vector = embedder.get_dense_embeddings_sync([query])[0]
        sparse_vector = embedder.get_sparse_embeddings([query])[0]

        # 2. Делаем запрос с Fusion (RRF)
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="text-dense",
                    limit=limit * 3
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using="text-sparse",
                    limit=limit * 3
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True
        )

        return [hit.payload for hit in results.points]

# Создаем синглтон для использования в цепочках
vector_store = WarhammerVectorStore()