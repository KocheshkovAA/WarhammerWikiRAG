import sys
import os
import asyncio
import json
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from langfuse.langchain import CallbackHandler
from langfuse import observe, get_client, propagate_attributes

# Настройка путей
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from app.core.config import settings
from app.chains.base import rag_chain 
from app.core.reranker import reranker

K_VALUES = [3, 5, 10, 20]
langfuse_client = get_client()

def normalize_text(text: str) -> str:
    return text.lower().strip() if text else ""

@observe(name="Retrieval Evaluation Single Case")
async def evaluate_one(question_data, use_rerank: bool):
    q_id = question_data.get("id", "??")
    question = question_data["question"]

    # Подготовка эталонов
    expected_titles = question_data.get("article_title", [])
    if isinstance(expected_titles, str): expected_titles = [expected_titles]
    
    expected_quotes = question_data.get("quote", [])
    if isinstance(expected_quotes, str): expected_quotes = [expected_quotes]
    expected_quotes = [q for q in expected_quotes if q and isinstance(q, str)]

    # Настройки
    reranker.enabled = use_rerank
    settings.QUERY_OPTIMIZER_ENABLED = False # Отключаем для чистоты сравнения Base/Rerank

    try:
        handler = CallbackHandler()
        # Вызов общего метода поиска из твоего RAG класса
        final_docs = await rag_chain.get_relevant_documents(question, handler=handler)

        if not final_docs:
            return {"error": "No documents retrieved"}

        retrieved_titles = [doc.metadata.get("article_name", "UNKNOWN") for doc in final_docs]
        retrieved_contents = [doc.page_content for doc in final_docs]

        metrics = {}
        
        # ── Метрики по заголовкам ──
        for k in K_VALUES:
            top_k_titles = retrieved_titles[:k]
            
            # Hit
            hit = any(any(normalize_text(t) == normalize_text(et) for et in expected_titles) 
                      for t in top_k_titles if t != "UNKNOWN")
            
            # Recall & Precision
            found_count = sum(1 for t in top_k_titles if t != "UNKNOWN" and 
                              any(normalize_text(t) == normalize_text(et) for et in expected_titles))
            
            # MRR
            mrr = next((1.0 / (i + 1) for i, t in enumerate(top_k_titles) if t != "UNKNOWN" and 
                        any(normalize_text(t) == normalize_text(et) for et in expected_titles)), 0.0)

            metrics[f"title_hit@{k}"] = int(hit)
            metrics[f"title_recall@{k}"] = found_count / len(expected_titles) if expected_titles else 0.0
            metrics[f"title_precision@{k}"] = found_count / k
            metrics[f"title_mrr@{k}"] = mrr

        # ── Метрики по цитатам ──
        norm_expected_quotes = [normalize_text(q) for q in expected_quotes]
        norm_retrieved_contents = [normalize_text(c) for c in retrieved_contents]

        for k in K_VALUES:
            top_k_contents = norm_retrieved_contents[:k]
            
            if norm_expected_quotes:
                found_quotes = set()
                for eq in norm_expected_quotes:
                    if any(eq in chunk for chunk in top_k_contents):
                        found_quotes.add(eq)
                
                relevant_chunks = sum(1 for chunk in top_k_contents 
                                      if any(eq in chunk for eq in norm_expected_quotes))
                
                mrr_cit = next((1.0 / (i + 1) for i, chunk in enumerate(top_k_contents) 
                                if any(eq in chunk for eq in norm_expected_quotes)), 0.0)

                metrics[f"citation_recall@{k}"] = len(found_quotes) / len(norm_expected_quotes)
                metrics[f"citation_hit@{k}"] = 1 if found_quotes else 0
                metrics[f"citation_precision@{k}"] = relevant_chunks / k
                metrics[f"citation_mrr@{k}"] = mrr_cit
            else:
                metrics.update({f"citation_recall@{k}": 0.0, f"citation_hit@{k}": 0, 
                                f"citation_precision@{k}": 0.0, f"citation_mrr@{k}": 0.0})
        
        return metrics

    except Exception as e:
        print(f"q{q_id} | Ошибка: {e}")
        return {"error": str(e)}

async def run_evaluation(use_rerank: bool):
    dataset_path = Path(settings.DATASET_PATH)
    with open(dataset_path, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    print(f"\n🚀 Оценка | Rerank = {use_rerank} | Вопросов: {len(questions)}")

    results = await tqdm_asyncio.gather(
        *[evaluate_one(q, use_rerank) for q in questions],
        desc=f"Rerank={use_rerank}"
    )

    valid = [r for r in results if r and "error" not in r]
    if not valid: return {}

    n = len(valid)
    return {key: sum(r.get(key, 0.0) for r in valid) / n for key in valid[0].keys()}

def print_table(agg, use_rerank: bool):
    mode = "WITH RERANKER" if use_rerank else "WITHOUT RERANKER"
    print("\n" + "═" * 100)
    print(f"          SUMMARY — {mode}")
    print("═" * 100)
    print(f"{'Метрика':<28} {'@3':<10} {'@5':<10} {'@10':<10} {'@20':<10}")
    print("-" * 100)
    
    for base_metric in ["title_hit", "title_recall", "title_precision", "title_mrr", 
                        "citation_hit", "citation_recall", "citation_precision", "citation_mrr"]:
        row = f"{base_metric:<28} "
        for k in K_VALUES:
            val = agg.get(f"{base_metric}@{k}", 0)
            row += f"{val:<10.3f} "
        print(row)

async def main():
    # 1. Без реранкера
    agg_no = await run_evaluation(use_rerank=False)
    print_table(agg_no, False)

    # 2. С реранкером
    agg_yes = await run_evaluation(use_rerank=True)
    print_table(agg_yes, True)

    # 3. Сравнение
    print("\n" + "═" * 100)
    print(f"{'СРАВНЕНИЕ @5':<28} | {'Base':<10} | {'Rerank':<10} | {'Delta'}")
    print("-" * 100)
    for m in ["title_hit@5", "title_mrr@5", "citation_recall@5", "citation_precision@5"]:
        v1, v2 = agg_no.get(m, 0), agg_yes.get(m, 0)
        print(f"{m:<28} | {v1:<10.3f} | {v2:<10.3f} | {v2-v1:+10.3f}")

if __name__ == "__main__":
    asyncio.run(main())