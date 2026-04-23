# RAG-система по лору Warhammer 40,000
## Что внутри
- **Фреймворк**: LangChain
- **Векторная БД**: Qdrant 
- **Embeddings**: TEI (Text Embeddings Inference) — Qwen3-Embedding-0.6B
- **Reranker**: bge-reranker-v2-m3
- **LLM**: GigaChat Pro / GigaChat Lite (через API Sber)
- **API**: FastAPI
- **Инфраструктура**:
  - Всё в Docker + Docker Compose
  - Tracing & Observability — **Langfuse**
- **Оркестрация запросов**:
  - Умный роутинг: простые факты → классический RAG, сложные связи → **LightRAG** (графовый)

## Пайплайн
<img width="376" height="464" alt="RAG-Page-4 drawio" src="https://github.com/user-attachments/assets/9ca31646-69d7-4e65-aa8e-8ef8b716c4ea" />

## Сервисы
<img width="363" height="348" alt="RAG-Page-3 drawio" src="https://github.com/user-attachments/assets/1d7607d0-90d8-41e1-995a-d24dc6141dbe" />

## LightRAG
<img width="1403" height="800" alt="image" src="https://github.com/user-attachments/assets/76cbfeb3-b3fb-4b8b-8040-86892796391a" />

## Метрики и эксперименты
**Эксперименты по выбору би-енкодера:** experiments/test_models.ipynb

### Оценка ретривера
Для оценки качества ретривера мы собрали собственный датасет из 60 вопросов.
Каждый вопрос был размечен вручную:
✅ релевантные статьи (article_title)
✅ ключевые цитаты, которые должны находиться в retrieved чанках
Таким образом, разметка покрывает как документный уровень, так и уровень конкретных текстовых совпадений.
Использование реранкера показало рост метрик.
```
════════════════════════════════════════════════════════════════════════════════════════════════════
СРАВНЕНИЕ @5                 | Base       | Rerank     | Delta
----------------------------------------------------------------------------------------------------
title_hit@5                  | 0.873      | 0.933      |     +0.061
title_mrr@5                  | 0.752      | 0.850      |     +0.098
citation_recall@5            | 0.673      | 0.817      |     +0.144
citation_precision@5         | 0.156      | 0.190      |     +0.034
```
Конечные значение метрик предствалены в таблице.
```
Метрика                      @3         @5         @10        @20       
----------------------------------------------------------------------------------------------------
title_hit                    0.883      0.933      0.933      0.933      
title_recall                 1.725      2.625      2.625      2.625      
title_precision              0.628      0.573      0.287      0.143      
title_mrr                    0.839      0.850      0.850      0.850      
citation_hit                 0.750      0.867      0.867      0.867      
citation_recall              0.692      0.817      0.817      0.817      
citation_precision           0.261      0.190      0.095      0.047      
citation_mrr                 0.633      0.659      0.659      0.659      
```
