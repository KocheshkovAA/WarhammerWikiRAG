# regenerate_quotes.py — минимальная версия: только проблемные цитаты
import sys
import os
import json
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from app.core.config import settings
from app.core.retriever import vector_store


def normalize_text(text: str) -> str:
    return text.lower().strip() if text else ""


def regenerate_dataset():
    dataset_path = Path(settings.DATASET_PATH)
    if not dataset_path.exists():
        print(f"Ошибка: датасет не найден по пути {dataset_path}")
        return

    output_path = dataset_path.with_name(dataset_path.stem + "_fixed.jsonl")

    fixed_count = 0
    manual_count = 0

    questions = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    with open(output_path, "w", encoding="utf-8") as f_out:
        for q in questions:
            q_id = q["id"]
            expected_titles = (
                [q["article_title"]] if isinstance(q["article_title"], str)
                else q["article_title"] or []
            )

            if not expected_titles:
                f_out.write(json.dumps(q, ensure_ascii=False) + "\n")
                continue

            all_chunks = []
            for title in expected_titles[:1]:
                docs = vector_store.hybrid_search(title, limit=50)
                chunks = [doc["content"] for doc in docs if doc.get("content")]
                all_chunks.extend(chunks)

            full_text = " ".join(all_chunks)

            old_quotes = q.get("quote")
            if not old_quotes:
                f_out.write(json.dumps(q, ensure_ascii=False) + "\n")
                continue

            if isinstance(old_quotes, str):
                old_quotes = [old_quotes]

            new_quotes = []
            for idx, old_quote in enumerate(old_quotes):
                if not old_quote or not isinstance(old_quote, str) or len(old_quote.strip()) < 10:
                    new_quotes.append(old_quote)
                    continue

                norm_old = normalize_text(old_quote)
                old_len = len(norm_old)

                best_match = ""
                best_score = 0

                for i in range(len(full_text) - old_len + 1):
                    candidate = full_text[i:i + old_len]
                    score = sum(a == b for a, b in zip(norm_old, candidate.lower()))
                    if score > best_score:
                        best_score = score
                        best_match = candidate.strip()

                match_ratio = best_score / old_len if old_len else 0

                if match_ratio >= 0.75:
                    new_quotes.append(best_match)
                    fixed_count += 1
                else:
                    manual_count += 1
                    print(f"q{q_id} | {old_quote}")

                    new_quotes.append(old_quote)

            q["quote"] = new_quotes if len(new_quotes) > 1 else new_quotes[0] if new_quotes else None
            f_out.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\nАвто-исправлено: {fixed_count}")
    print(f"Ручная правка нужна: {manual_count}")
    print(f"Всего вопросов: {len(questions)}")
    print(f"Новый файл: {output_path}")


if __name__ == "__main__":
    regenerate_dataset()