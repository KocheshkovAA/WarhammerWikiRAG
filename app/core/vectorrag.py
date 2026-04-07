from operator import itemgetter
from typing import List, Dict, Any

from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langfuse import propagate_attributes, get_client
from langfuse.langchain import CallbackHandler

from app.core.query_processor import QueryOptimizer
from langfuse import observe
from app.core.retriever import Retriever
from app.core.llm import llm_factory
from app.chains.prompts import QA_SYSTEM_PROMPT
from app.core.postprocessors.context_builder import ContextBuilder
from app.core.postprocessors.source_extractor import SourceExtractor

import asyncio

from app.core.reranker import reranker  # ← твой класс Reranker
from app.core.config import settings
from typing import List

class RAG:
    def __init__(self):
        self.llm = llm_factory.get_llm(temperature=0.15)
        self.retriever = Retriever.from_collection(k=30)
        self.context_builder = ContextBuilder()
        self.source_extractor = SourceExtractor()
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", QA_SYSTEM_PROMPT),
            ("human", "Контекст:\n{context}\n\nВопрос: {question}")
        ])
        
        self.chain = (
            {
                "context": itemgetter("docs") | RunnableLambda(self.context_builder.build),
                "question": itemgetter("question"),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    @observe(name="Retrieve Only")
    async def get_relevant_documents(self, question: str, handler=None) -> List[Any]:
        """
        Чистый пайплайн поиска: Expansion -> Multi-Retrieval -> Rerank.
        Используется и в answer(), и в скриптах оценки.
        """
        # 1. Оптимизация запроса
        if settings.QUERY_OPTIMIZER_ENABLED:
            optimizer = QueryOptimizer(self.llm)
            expanded_queries = await optimizer.process(question)
        else:
            expanded_queries = [question]

        # 2. Параллельный поиск
        tasks = [
            self.retriever.ainvoke(q, config={"callbacks": [handler] if handler else []}) 
            for q in expanded_queries
        ]
        docs_results = await asyncio.gather(*tasks)

        # 3. Дедупликация
        all_docs = []
        seen_contents = set()
        for sublist in docs_results:
            for doc in sublist:
                if doc.page_content not in seen_contents:
                    all_docs.append(doc)
                    seen_contents.add(doc.page_content)

        if not all_docs:
            return []

        # 4. Реранжирование
        return await reranker.rerank_documents(question, all_docs)

    @observe(name="RAG Pipeline")
    async def answer(self, question: str):
        """Полный цикл с генерацией ответа."""
        with propagate_attributes(tags=["rag", "warhammer", "production"]):
            handler = CallbackHandler()

            # Используем общий метод поиска
            final_docs = await self.get_relevant_documents(question, handler=handler)

            if not final_docs:
                return {"answer": "Данные не найдены.", "sources": []}

            # 5. Генерация финального ответа
            answer = await self.chain.ainvoke(
                {"docs": final_docs, "question": question},
                config={"callbacks": [handler]}
            )

            return {
                "answer": answer,
                "sources": self.source_extractor.extract(final_docs)
            }

rag_chain = RAG()