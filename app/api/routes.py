from fastapi import APIRouter, Body
from app.chains.base import rag_chain # Импортируем актуальное имя
from pydantic import BaseModel

router = APIRouter()

# Создаем схему запроса
class QuestionRequest(BaseModel):
    question: str

@router.post("/ask")
async def ask_warhammer(request: QuestionRequest):
    # Вызываем именно тот метод, который мы написали в WarhammerRAG
    result = await rag_chain.answer(request.question)
    
    # Возвращаем результат (в нем уже есть answer и sources)
    return result