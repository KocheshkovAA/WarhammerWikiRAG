# RAG-система по лору Warhammer 40,000
## Что внутри
- **Фреймворк**: LangChain
- **Векторная БД**: Qdrant 
- **Embeddings**: TEI (Text Embeddings Inference) — Qwen3-Embedding-0.6B
- **Reranker**: bge-reranker-v2-m3
- **LLM**: GigaChat Pro / GigaChat Lite (через API Sber)
- **API**: FastAPI (асинхронный, готов к высоким нагрузкам)
- **Инфраструктура**:
  - Всё в Docker + Docker Compose
  - Tracing & Observability — **Langfuse**
- **Оркестрация запросов**:
  - Умный роутинг: простые факты → классический RAG, сложные связи → **LightRAG** (графовый)

## Пайплайн
<img width="376" height="464" alt="RAG-Page-4 drawio" src="https://github.com/user-attachments/assets/9ca31646-69d7-4e65-aa8e-8ef8b716c4ea" />

## Сервисы
<img width="363" height="348" alt="RAG-Page-3 drawio" src="https://github.com/user-attachments/assets/1d7607d0-90d8-41e1-995a-d24dc6141dbe" />

## Метрики и эксперименты
**Эксперименты по выбору би-енкодера:** experiments/test_models.ipynb
