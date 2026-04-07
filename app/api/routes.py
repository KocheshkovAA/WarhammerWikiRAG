from fastapi import APIRouter
from pydantic import BaseModel

# Импортируем твои компоненты
from app.core.vectorrag import rag_chain      # Твой существующий векторный RAG
from app.core.lightrag_client import LightRAGClient
from app.core.orchestrator import WarhammerOrchestrator # Новый класс, который мы создали

router = APIRouter()

# Инициализируем компоненты (лучше вынести это в app.core.base)
light_rag = LightRAGClient()
orchestrator = WarhammerOrchestrator(vector_rag=rag_chain, light_rag=light_rag)

class QuestionRequest(BaseModel):
    question: str

@router.post("/ask")
async def ask(request: QuestionRequest):
    # Теперь вызываем оркестратор, который сам решит: 
    # идти в векторную базу или в графовую
    result = await orchestrator.answer(request.question)
    return result