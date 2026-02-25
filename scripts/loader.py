import requests
import json
import time

def fetch_all_warhammer_wiki(output_file="raw_warhammer_data.jsonl", delay=1.0):
    base_url = "https://warhammer40k.fandom.com/ru/api.php"
    
    # 1. Получаем список всех названий статей (Pagination)
    titles = []
    list_params = {
        "action": "query",
        "list": "allpages",
        "aplimit": "max",  # Запрашиваем максимально возможный батч названий (500)
        "format": "json"
    }

    print("🔍 Сбор списка названий всех статей...")
    while True:
        try:
            response = requests.get(base_url, params=list_params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            batch = data.get("query", {}).get("allpages", [])
            titles.extend([p["title"] for p in batch])
            
            print(f"--- Собрано {len(titles)} названий...")
            
            if "continue" in data:
                list_params.update(data["continue"])
            else:
                break
        except Exception as e:
            print(f"❌ Ошибка при получении списка: {e}")
            break

    print(f"✅ Всего найдено статей: {len(titles)}")
    print(f"💾 Начинаю выгрузку в файл {output_file} с задержкой {delay} сек...")

    # 2. Выгружаем контент каждой статьи
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for title in titles:
            parse_params = {
                "action": "parse",
                "page": title,
                "prop": "text|categories|sections",
                "redirects": 1,  # Автоматический резолв редиректов
                "format": "json"
            }
            
            try:
                # Делаем паузу ПЕРЕД запросом
                time.sleep(delay)
                
                res = requests.get(base_url, params=parse_params, timeout=15)
                res.raise_for_status()
                article_data = res.json()
                
                if "parse" in article_data:
                    # Записываем статью как одну строку JSONL
                    f.write(json.dumps(article_data, ensure_ascii=False) + "\n")
                    count += 1
                else:
                    print(f"⚠️ Проблема со статьей '{title}': {article_data.get('error')}")

                if count % 10 == 0:
                    print(f"🚀 Прогресс: {count}/{len(titles)} статей сохранено...")

            except Exception as e:
                print(f"❌ Критическая ошибка на статье '{title}': {e}")
                # Если упало соединение, подождем чуть дольше и попробуем следующую
                time.sleep(5)
                continue

    print(f"🎉 Готово! Всего выгружено и сохранено {count} статей.")

if __name__ == "__main__":
    fetch_all_warhammer_wiki()