from app.core.vectorstore import vector_store
from app.core.llm import llm_factory
from app.chains.prompts import QA_SYSTEM_PROMPT
from langfuse.langchain import CallbackHandler
from app.core.config import settings

class WarhammerRAG:
    def __init__(self):
        # Получаем LLM через фабрику
        self.llm = llm_factory.get_llm(temperature=0.15)
        
        # Handler для Langfuse
        self.langfuse_handler = CallbackHandler()

    async def answer(self, question: str):
        # 1. Retrieval
        docs = vector_store.hl_search(question, limit=4)
        context = "\n\n".join([doc["content"] for doc in docs])

        # 2. Собираем сообщения (Chat Format)
        messages = [
            ("system", QA_SYSTEM_PROMPT),
            ("user", f"Контекст:\n{context}\n\nВопрос: {question}")
        ]

        # 3. Invoke с трейсингом
        response = await self.llm.ainvoke(
            messages, 
            config={"callbacks": [self.langfuse_handler]}
        )
        
        return {
            "answer": response.content,
            "sources": [doc.get("metadata", {}).get("source", "Unknown") for doc in docs]
        }

rag_chain = WarhammerRAG()