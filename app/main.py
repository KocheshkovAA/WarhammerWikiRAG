from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Warhammer RAG")

app.include_router(router, prefix="/v1")

@app.get("/health")
def health():
    return {"status": "ok"}