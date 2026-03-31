from typing import List, Dict, Any
from langchain_core.documents import Document


class SourceExtractor:
    """Извлекает источники / метаданные для ответа API"""

    @staticmethod
    def extract(docs: List[Document]) -> List[Dict[str, Any]]:
        return [
            {
                "article_name": doc.metadata.get("article_name", "N/A"),
                "url": doc.metadata.get("url", ""),
                "title": doc.metadata.get("title", ""),
                "score": round(float(doc.metadata.get("hybrid_score", 0.0)), 4),
            }
            for doc in docs
        ]