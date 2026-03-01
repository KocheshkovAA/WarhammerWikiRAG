# evaluate_retrieval.py — финальная версия с @20 и всеми метриками в таблице
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from app.core.config import settings
from app.core.vectorstore import vector_store

import asyncio
import json
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime

from langfuse import observe, get_client, propagate_attributes

K_VALUES = [3, 5, 10, 20]  # добавили @20
langfuse_client = get_client()


def normalize_text(text: str) -> str:
    return text.lower().strip() if text else ""


@observe(name="Retrieval Evaluation Single", capture_input=True, capture_output=True)
async def evaluate_one(question_data):
    q_id = question_data["id"]
    question = question_data["question"]

    expected_titles = (
        [question_data["article_title"]]
        if isinstance(question_data["article_title"], str)
        else question_data["article_title"] or []
    )

    expected_quotes = (
        question_data["quote"]
        if isinstance(question_data["quote"], list)
        else [question_data["quote"]]
    )
    expected_quotes = [q for q in expected_quotes if q and isinstance(q, str)]

    with propagate_attributes(tags=["eval", "retrieval-only", f"q{q_id}"]):
        try:
            docs = vector_store.hybrid_search(question, limit=max(K_VALUES))
        except Exception as e:
            print(f"q{q_id} | Ошибка поиска: {e}")
            return {"error": str(e)}

        retrieved_titles = [
            doc.get("metadata", {}).get("article_name", "UNKNOWN") for doc in docs
        ]
        retrieved_contents = [doc.get("content", "") for doc in docs]

        metrics = {}

        # ── Метрики по заголовкам (title-based) ────────────────────────────────
        for k in K_VALUES:
            top_k_titles = retrieved_titles[:k]

            hit = any(
                any(t.strip().lower() == et.strip().lower() for et in expected_titles)
                for t in top_k_titles if t != "UNKNOWN"
            )

            found_count = sum(
                1 for t in top_k_titles
                if t != "UNKNOWN" and any(t.strip().lower() == et.strip().lower() for et in expected_titles)
            )

            title_recall = found_count / len(expected_titles) if expected_titles else 0.0
            title_precision = found_count / k if k > 0 else 0.0
            title_mrr = next(
                (1.0 / (i + 1) for i, t in enumerate(top_k_titles)
                 if t != "UNKNOWN" and any(t.strip().lower() == et.strip().lower() for et in expected_titles)),
                0.0
            )

            metrics[f"title_hit@{k}"] = int(hit)
            metrics[f"title_recall@{k}"] = title_recall
            metrics[f"title_precision@{k}"] = title_precision
            metrics[f"title_mrr@{k}"] = title_mrr

        # ── Метрики по цитатам (citation-based) ────────────────────────────────
        norm_expected_quotes = [normalize_text(q) for q in expected_quotes]

        if norm_expected_quotes:
            norm_retrieved_contents = [normalize_text(c) for c in retrieved_contents if c]

            for k in K_VALUES:
                top_k_contents = norm_retrieved_contents[:k]
                found_quotes = set()

                for norm_quote in norm_expected_quotes:
                    if any(norm_quote in chunk for chunk in top_k_contents):
                        found_quotes.add(norm_quote)

                citation_recall = len(found_quotes) / len(norm_expected_quotes)
                metrics[f"citation_recall@{k}"] = citation_recall

                # Добавляем hit для каждого k
                metrics[f"citation_hit@{k}"] = 1 if found_quotes else 0

                # Добавляем precision по цитатам: сколько % чанков содержат хотя бы одну цитату
                relevant_chunks = sum(1 for chunk in top_k_contents if any(norm_quote in chunk for norm_quote in norm_expected_quotes))
                citation_precision = relevant_chunks / k if k > 0 else 0.0
                metrics[f"citation_precision@{k}"] = citation_precision

                # MRR по цитатам (позиция первого чанка с цитатой)
                mrr = 0.0
                for i, chunk in enumerate(top_k_contents, 1):
                    if any(norm_quote in chunk for norm_quote in norm_expected_quotes):
                        mrr = 1.0 / i
                        break
                metrics[f"citation_mrr@{k}"] = mrr
        else:
            for k in K_VALUES:
                metrics[f"citation_recall@{k}"] = 0.0
                metrics[f"citation_hit@{k}"] = 0
                metrics[f"citation_precision@{k}"] = 0.0
                metrics[f"citation_mrr@{k}"] = 0.0

        # Логируем в Langfuse
        langfuse_client.update_current_span(
            metadata={
                "num_docs": len(docs),
                "metrics": metrics,
                "q_id": q_id,
                "expected_titles_cnt": len(expected_titles),
                "expected_quotes_cnt": len(expected_quotes),
            }
        )

        current_trace_id = langfuse_client.get_current_trace_id()
        if current_trace_id:
            for name, val in metrics.items():
                if isinstance(val, (int, float)):
                    langfuse_client.create_score(
                        trace_id=current_trace_id,
                        name=name,
                        value=val,
                        comment=f"{name} q{q_id}",
                        data_type="NUMERIC",
                    )

        # Выводим основные метрики
        print(
            f"q{q_id} | "
            f"title_hit@5: {metrics.get('title_hit@5', 0)} | "
            f"title_recall@5: {metrics.get('title_recall@5', 0):.2f} | "
            f"citation_recall@5: {metrics.get('citation_recall@5', 0):.2f} | "
            f"docs: {len(docs)}"
        )

        return metrics


async def main():
    questions = []
    with open(settings.DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    print(f"Оцениваем {len(questions)} вопросов...")

    @observe(name="Full Retrieval Evaluation Batch", capture_input=True, capture_output=True)
    async def run_batch():
        results = await tqdm_asyncio.gather(
            *[evaluate_one(q) for q in questions],
            desc="Оценка вопросов",
        )

        agg = {}
        for k in K_VALUES:
            agg[f"title_hit@{k}"]         = sum(r.get(f"title_hit@{k}", 0)         for r in results) / len(results)
            agg[f"title_recall@{k}"]      = sum(r.get(f"title_recall@{k}", 0)      for r in results) / len(results)
            agg[f"title_precision@{k}"]   = sum(r.get(f"title_precision@{k}", 0)   for r in results) / len(results)
            agg[f"title_mrr@{k}"]         = sum(r.get(f"title_mrr@{k}", 0)         for r in results) / len(results)
            agg[f"citation_recall@{k}"]   = sum(r.get(f"citation_recall@{k}", 0)   for r in results) / len(results)
            agg[f"citation_hit@{k}"]      = sum(r.get(f"citation_hit@{k}", 0)      for r in results) / len(results)
            agg[f"citation_precision@{k}"] = sum(r.get(f"citation_precision@{k}", 0) for r in results) / len(results)
            agg[f"citation_mrr@{k}"]      = sum(r.get(f"citation_mrr@{k}", 0)      for r in results) / len(results)

        # Красивый вывод — теперь все метрики и все k
        print("\n" + "═" * 100)
        print("                      RETRIEVAL EVALUATION SUMMARY")
        print("═" * 100)
        print(f"{'Метрика':<24} {'@3':<10} {'@5':<10} {'@10':<10} {'@20':<10}")
        print("-" * 100)
        print(f"{'title_hit':<24} {agg['title_hit@3']:<10.1%} {agg['title_hit@5']:<10.1%} {agg['title_hit@10']:<10.1%} {agg['title_hit@20']:<10.1%}")
        print(f"{'title_recall':<24} {agg['title_recall@3']:<10.2f} {agg['title_recall@5']:<10.2f} {agg['title_recall@10']:<10.2f} {agg['title_recall@20']:<10.2f}")
        print(f"{'title_precision':<24} {agg['title_precision@3']:<10.2f} {agg['title_precision@5']:<10.2f} {agg['title_precision@10']:<10.2f} {agg['title_precision@20']:<10.2f}")
        print(f"{'title_mrr':<24} {agg['title_mrr@3']:<10.2f} {agg['title_mrr@5']:<10.2f} {agg['title_mrr@10']:<10.2f} {agg['title_mrr@20']:<10.2f}")
        print("-" * 100)
        print(f"{'citation_recall':<24} {agg['citation_recall@3']:<10.2f} {agg['citation_recall@5']:<10.2f} {agg['citation_recall@10']:<10.2f} {agg['citation_recall@20']:<10.2f}")
        print(f"{'citation_hit':<24} {agg['citation_hit@3']:<10.1%} {agg['citation_hit@5']:<10.1%} {agg['citation_hit@10']:<10.1%} {agg['citation_hit@20']:<10.1%}")
        print(f"{'citation_precision':<24} {agg['citation_precision@3']:<10.2f} {agg['citation_precision@5']:<10.2f} {agg['citation_precision@10']:<10.2f} {agg['citation_precision@20']:<10.2f}")
        print(f"{'citation_mrr':<24} {agg['citation_mrr@3']:<10.2f} {agg['citation_mrr@5']:<10.2f} {agg['citation_mrr@10']:<10.2f} {agg['citation_mrr@20']:<10.2f}")
        print("═" * 100)

        # Худшие вопросы (по citation_recall@5)
        failures = [
            (q, r) for q, r in zip(questions, results)
            if r.get("citation_recall@5", 1.0) < 0.5
        ]
        print(f"\nХудшие вопросы (citation_recall@5 < 0.5): {len(failures)}")
        for q, r in failures[:10]:  # первые 10
            print(f"q{q['id']} | {q['question'][:70]}... | citation_recall@5: {r.get('citation_recall@5', 0):.2f}")

        # Сохранение отчёта
        report = {
            "timestamp": datetime.now().isoformat(),
            "dataset_size": len(questions),
            "aggregated": agg,
            "failures_sample": [{"id": q["id"], "question": q["question"], "metrics": r} for q, r in failures[:5]]
        }
        with open("retrieval_eval_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        with open("retrieval_eval_report.md", "w", encoding="utf-8") as f:
            f.write("# Retrieval Evaluation Report\n\n")
            f.write(f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write("## Агрегированные метрики\n\n")
            f.write("| Метрика            | @3     | @5     | @10    | @20    |\n")
            f.write("|--------------------|--------|--------|--------|--------|\n")
            f.write(f"| title_hit          | {agg['title_hit@3']:.1%} | {agg['title_hit@5']:.1%} | {agg['title_hit@10']:.1%} | {agg['title_hit@20']:.1%} |\n")
            f.write(f"| title_recall       | {agg['title_recall@3']:.2f} | {agg['title_recall@5']:.2f} | {agg['title_recall@10']:.2f} | {agg['title_recall@20']:.2f} |\n")
            f.write(f"| title_precision    | {agg['title_precision@3']:.2f} | {agg['title_precision@5']:.2f} | {agg['title_precision@10']:.2f} | {agg['title_precision@20']:.2f} |\n")
            f.write(f"| title_mrr          | {agg['title_mrr@3']:.2f} | {agg['title_mrr@5']:.2f} | {agg['title_mrr@10']:.2f} | {agg['title_mrr@20']:.2f} |\n")
            f.write(f"| citation_recall    | {agg['citation_recall@3']:.2f} | {agg['citation_recall@5']:.2f} | {agg['citation_recall@10']:.2f} | {agg['citation_recall@20']:.2f} |\n")
            f.write(f"| citation_hit       | -      | {agg['citation_hit@5']:.1%} | -      | {agg.get('citation_hit@20', 0):.1%} |\n\n")

            f.write("## Худшие вопросы (citation_recall@5 < 0.5)\n\n")
            for q, r in failures[:5]:
                f.write(f"- q{q['id']} | {q['question'][:100]}... | citation_recall@5: {r.get('citation_recall@5', 0):.2f}\n")

        return agg

    await run_batch()


if __name__ == "__main__":
    asyncio.run(main())