from fastapi import APIRouter, Body
from app.chains.base import rag_chain
from pydantic import BaseModel

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

@router.post("/ask")
async def ask(request: QuestionRequest):
    result = await rag_chain.answer(request.question)
    return result