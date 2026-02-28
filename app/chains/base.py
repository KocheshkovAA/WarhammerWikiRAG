from app.core.vectorstore import vector_store
from app.core.llm import llm_factory
from app.chains.prompts import QA_SYSTEM_PROMPT
from langfuse.langchain import CallbackHandler
from langfuse import observe, get_client, propagate_attributes

class RAG:
    def __init__(self):
        self.llm = llm_factory.get_llm(temperature=0.15)
        self.langfuse_handler = CallbackHandler()
        self.langfuse = get_client()

    @observe(name="RAG → Full Pipeline", capture_input=True, capture_output=True)
    async def answer(self, question: str):
        with propagate_attributes(tags=["rag", "production", "warhammer"]):
            docs = vector_store.hybrid_search(question, limit=3)
            context = "\n\n".join(doc["content"] for doc in docs if "content" in doc)

            self.langfuse.update_current_span( 
                metadata={
                    "num_docs_retrieved": len(docs),
                    "context_length_chars": len(context),
                    "question_length_chars": len(question),
                    "retrieved_sources": [doc.get("metadata", {}).get("source", "N/A") for doc in docs]
                }
            )

            messages = [
                ("system", QA_SYSTEM_PROMPT),
                ("user", f"Контекст:\n{context}\n\nВопрос: {question}")
            ]

            response = await self.llm.ainvoke(
                messages,
                config={
                    "callbacks": [self.langfuse_handler],
                    "run_name": "Answer Generation",
                }
            )

            if hasattr(self.langfuse_handler, "last_trace_id") and self.langfuse_handler.last_trace_id:
                self.langfuse.create_score(
                    trace_id=self.langfuse_handler.last_trace_id,
                    name="answer_length_chars",
                    value=len(response.content),
                    comment="Длина сгенерированного ответа (символы)",
                    data_type="NUMERIC",
                )
                self.langfuse.create_score(
                    trace_id=self.langfuse_handler.last_trace_id,
                    name="retrieval_docs_count",
                    value=len(docs),
                    comment="Количество найденных документов",
                    data_type="NUMERIC",
                )

            sources = [
                {
                    "source": doc.get("metadata", {}).get("source", "Unknown"),
                    "title": doc.get("metadata", {}).get("title", ""),
                    "url": doc.get("metadata", {}).get("url", ""),
                }
                for doc in docs
            ]

            return {
                "answer": response.content,
                "sources": sources,
                "metadata": {
                    "docs_count": len(docs),
                    "context_length": len(context),
                }
            }

rag_chain = RAG()