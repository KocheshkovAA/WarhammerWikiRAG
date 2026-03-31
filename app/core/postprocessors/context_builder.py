from typing import List
from langchain_core.documents import Document


class ContextBuilder:
    """Отвечает за формирование контекста из документов"""

    def __init__(self, separator: str = "\n\n", max_chars: int = 15000):
        self.separator = separator
        self.max_chars = max_chars

    def build(self, docs: List[Document]) -> str:
        """Собирает контекст из документов с учётом лимита символов"""
        context_parts = []
        current_length = 0

        for doc in docs:
            content = doc.page_content.strip()
            if not content:
                continue

            added = content + self.separator
            if current_length + len(added) > self.max_chars:
                break

            context_parts.append(content)
            current_length += len(added)

        return self.separator.join(context_parts)