# diagnose_citation_quality.py
import sys
import os
import json
import asyncio
from pathlib import Path
from collections import defaultdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from app.core.config import settings
from app.core.retriever import vector_store

async def check_citations_in_qdrant():
    print("═" * 80)
    print(" ДИАГНОСТИКА: проверка вхождения цитат из датасета в реальные чанки Qdrant ")
    print("═" * 80)

    dataset_path = Path(settings.DATASET_PATH)
    if not dataset_path.exists():
        print(f"Ошибка: датасет не найден по пути {dataset_path}")
        return

    questions = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    print(f"Всего вопросов в датасете: {len(questions)}")

    total_citations = 0
    matched_citations = 0
    problems = defaultdict(list)

    for q in questions:
        q_id = q["id"]
        title_or_titles = q.get("article_title")
        expected_titles = (
            [title_or_titles] if isinstance(title_or_titles, str)
            else title_or_titles or []
        )

        quotes = q.get("quote", [])
        if isinstance(quotes, str):
            quotes = [quotes]
        quotes = [q.strip() for q in quotes if q and isinstance(q, str)]

        if not quotes or not expected_titles:
            continue

        total_citations += len(quotes)

        # Собираем все чанки по ожидаемым статьям
        all_relevant_chunks = []
        for title in expected_titles:
            # Ищем по названию статьи (BM25/keyword)
            docs = vector_store.hybrid_search(
                title,  # ищем по заголовку статьи
                limit=50,  # берём много, чтобы точно захватить
                # можно добавить фильтр по метаданным, если есть
            )
            chunks = [doc["content"] for doc in docs if doc.get("content")]
            all_relevant_chunks.extend(chunks)

        if not all_relevant_chunks:
            problems["no_chunks"].append(q_id)
            continue

        # Проверяем каждую цитату
        for quote in quotes:
            found = False
            for chunk in all_relevant_chunks:
                if quote.strip() in chunk:
                    found = True
                    matched_citations += 1
                    break

            if not found:
                problems["not_found"].append({
                    "q_id": q_id,
                    "question": q["question"][:80] + "...",
                    "title": expected_titles[0] if expected_titles else "—",
                    "quote_preview": quote[:120] + ("..." if len(quote) > 120 else ""),
                })

    match_rate = matched_citations / total_citations if total_citations else 0

    print(f"\nРезультаты проверки:")
    print(f"  Всего цитат в датасете: {total_citations}")
    print(f"  Из них найдено в чанках: {matched_citations} ({match_rate:.1%})")

    if problems["no_chunks"]:
        print(f"\nВопросы без чанков вообще: {len(problems['no_chunks'])}")

    if problems["not_found"]:
        print(f"\nЦитаты, которых НЕТ в реальных чанках: {len(problems['not_found'])}")
        print("Первые 5 проблемных:")
        for p in problems["not_found"][:5]:
            print(f"q{p['q_id']} | {p['question']}")
            print(f"   Статья: {p['title']}")
            print(f"   Цитата: {p['quote_preview']}")
            print("-" * 60)

    print("\nВывод:")
    if match_rate > 0.85:
        print("→ Цитаты в датасете в целом соответствуют тексту вики → проблема в ранжировании/чанкинге")
    elif match_rate > 0.5:
        print("→ Среднее качество цитат → нужно проверить чанки + добавить fuzzy-поиск")
    else:
        print("→ Цитаты в датасете сильно расходятся с реальным текстом → нужно перегенерировать датасет")


if __name__ == "__main__":
    asyncio.run(check_citations_in_qdrant())