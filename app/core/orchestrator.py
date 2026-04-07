from enum import Enum
from pydantic import BaseModel, Field
from app.core.llm import llm_factory
from app.core.vectorrag import RAG
from langfuse import observe
from langfuse import propagate_attributes
from app.core.lightrag_client import LightRAGClient

# 1. Определяем типы запросов
class RAGRoute(str, Enum):
    VECTOR = "vector"   # Твой текущий RAG (факты, даты, конкретика)
    GRAPH = "graph"     # LightRAG (связи, таймлайны, "почему", "как повлияло")

class RouteDecision(BaseModel):
    """Схема для классификации входящего вопроса пользователя по вселенной Warhammer 40k.""" # Обязательно добавь описание сюда!
    
    reasoning: str = Field(
        ..., 
        description="Краткое обоснование, почему выбран этот путь (векторный поиск или графовый)"
    )
    route: RAGRoute = Field(
        ..., 
        description="Выбранный маршрут: 'vector' для простых фактов, 'graph' для сложных связей и лора"
    )

class WarhammerOrchestrator:
    def __init__(self, vector_rag: RAG, light_rag: LightRAGClient):
        self.vector_rag = vector_rag
        self.light_rag = light_rag
        self.llm = llm_factory.get_llm(temperature=0) # Температура 0 для стабильной логики
        
        # Инструкция для классификатора
        self.system_prompt = (
            "Ты — логический модуль системы Warhammer 40k Lore Knowledge Base. "
            "Твоя задача — выбрать лучший инструмент для ответа на вопрос.\n\n"
            "Выбирай 'vector' (Vector RAG), если:\n"
            "- Нужен конкретный факт (Кто убил? В каком году? На какой планете?)\n"
            "- Нужно описание конкретного юнита или персонажа.\n\n"
            "Выбирай 'graph' (Graph RAG), если:\n"
            "- Нужно понять связи между сущностями (Как связаны Магнус и Ариман?)\n"
            "- Нужен анализ событий (Как Ересь Хоруса повлияла на текущий Империум?)\n"
            "- Вопрос сложный, охватывает несколько эпох или организаций."
        )

    @observe(name="Router Decision")
    async def classify_route(self, question: str) -> RAGRoute:
        # Используем structured output, чтобы LLM всегда возвращала валидный JSON
        structured_llm = self.llm.with_structured_output(RouteDecision)
        
        messages = [
            ("system", self.system_prompt),
            ("human", f"Вопрос: {question}")
        ]
        
        decision = await structured_llm.ainvoke(messages)
        return decision.route

    @observe(name="Global Orchestrator")
    async def answer(self, question: str):
        with propagate_attributes(tags=["orchestrator", "warhammer"]):
            # Шаг 1: Классификация
            route = await self.classify_route(question)
            
            # Шаг 2: Маршрутизация
            if route == RAGRoute.GRAPH:
                # Вызываем LightRAG (например, в гибридном режиме)
                return await self.light_rag.query(question, mode="hybrid")
            else:
                # Вызываем твой стандартный векторный RAG
                return await self.vector_rag.answer(question)

# Использование
# orchestrator = WarhammerOrchestrator(rag_chain, lightrag_client)
# result = await orchestrator.answer("Как Ересь Хоруса изменила структуру управления Террой?")