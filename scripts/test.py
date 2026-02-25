import asyncio
from app.core.vectorstore import vector_store
from app.core.config import settings

async def test_warhammer_search():
    # Список тестовых запросов разного типа
    test_queries = [
        "Кто такой кодианский полк?"                         # Короткое ключевое слово
    ]

    print(f"🔍 Тестирование поиска в коллекции: {settings.COLLECTION_NAME}\n")

    for query in test_queries:
        print(f"❓ Запрос: '{query}'")
        try:
            # Вызываем твой новый метод
            results = vector_store.hl_search(query, limit=3)
            
            if not results:
                print("   ❌ Ничего не найдено.")
            
            for i, res in enumerate(results, 1):
                content = res.get('content', 'Нет текста')
                source = res.get('metadata', {}).get('source', 'unknown')
                
                # Обрезаем текст для вывода в консоль
                preview = (content) if len(content) > 150 else content
                
                print(f"   {i}. [Источник: {source}]")
                print(f"      Текст: {preview}")
            print("-" * 50)
            
        except Exception as e:
            print(f"   🚨 Ошибка при поиске: {e}")

if __name__ == "__main__":
    # Запускаем асинхронно, если твои методы эмбеддинга асинхронные
    # Если get_dense_embeddings_sync реально синхронный, можно просто вызвать функцию
    asyncio.run(test_warhammer_search())