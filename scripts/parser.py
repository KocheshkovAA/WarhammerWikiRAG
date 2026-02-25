import json
import os
import logging
import html2text
from urllib.parse import quote
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Настройка логирования для красивого вывода в консоль
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class WarhammerWikiParser:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        self.html_converter.body_width = 0
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )
        
        self.md_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3"),
                ("####", "Header_4"), ("#####", "Header_5"), ("######", "Header_6")
            ],
            strip_headers=False
        )

    def _clean_html(self, soup):
        """Удаляет мусорные элементы из HTML."""
        for tag in soup(['script', 'style', 'aside']):
            tag.decompose()
        for span in soup.find_all("span", {"class": "mw-editsection"}):
            span.decompose()
        return soup

    def _parse_infobox(self, soup):
        """Извлекает данные из карточки (aside) и формирует ссылки."""
        infobox_data = {}
        links = []
        aside = soup.find("aside", {"class": "portable-infobox"})
        
        if not aside:
            return None, []

        # Базовые поля данных
        for div in aside.select("div.pi-item.pi-data"):
            key_el = div.find("h3", class_="pi-data-label")
            val_el = div.find("div", class_="pi-data-value")
            if key_el and val_el:
                key = key_el.get_text(" ", strip=True)
                val = val_el.get_text(" ", strip=True)
                infobox_data[key] = val
                # Собираем ссылки из значений
                for a in val_el.find_all("a", href=True):
                    if "/ru/wiki/" in a["href"] and ":" not in a["href"]:
                        links.append(a.get("title", ""))
        
        return infobox_data, list(set(links))

    def _generate_url(self, title):
        """Создает валидную ссылку на статью."""
        safe_title = quote(title.replace(' ', '_'))
        return f"https://warhammer40k.fandom.com/ru/wiki/{safe_title}"

    def article_to_chunks(self, html_content, title, categories):
        """Главный метод превращения одной статьи в список Document."""
        soup = BeautifulSoup(html_content, 'html.parser')
        article_url = self._generate_url(title)
        final_chunks = []

        # 1. Сначала обрабатываем инфобокс (пока он еще не удален из soup)
        info_data, _ = self._parse_infobox(soup)
        if info_data:
            content = (
                f"СТАТЬЯ: {title}\n"
                f"ТИП: Сводная информация\n\n"
                f"{json.dumps(info_data, indent=2, ensure_ascii=False)}"
            )
            final_chunks.append(Document(
                page_content=content,
                metadata={"article_name": title, "url": article_url, "type": "infobox", "categories": categories}
            ))

        # 2. Чистим HTML и переводим в Markdown
        clean_soup = self._clean_html(soup)
        markdown_text = self.html_converter.handle(str(clean_soup))

        # 3. Режем по заголовкам, а затем по длине
        header_splits = self.md_header_splitter.split_text(markdown_text)
        text_chunks = self.text_splitter.split_documents(header_splits)

        # 4. Обогащаем текстовые чанки контекстом
        for chunk in text_chunks:
            # Собираем иерархию (Хлебные крошки)
            headers = [v for k, v in sorted(chunk.metadata.items()) if k.startswith("Header")]
            breadcrumb = " > ".join([title] + [h for h in headers if h != title])
            
            chunk.page_content = (
                f"СТАТЬЯ: {title}\n"
                f"КОНТЕКСТ: {breadcrumb}\n\n"
                f"{chunk.page_content}"
            )
            chunk.metadata = {
                "article_name": title,
                "url": article_url,
                "categories": categories,
                "type": "text"
            }
            final_chunks.append(chunk)

        return final_chunks

def main():
    # Конфигурация путей
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "./data/raw/raw_warhammer_data.jsonl")
    output_path = os.path.join(base_dir, "./data/processed/processed_chunks.jsonl")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    parser = WarhammerWikiParser()
    all_chunks = []
    
    # Множество для отслеживания уникальных текстов чанков
    seen_texts = set()
    duplicates_count = 0

    if not os.path.exists(input_path):
        logger.error(f"Входной файл не найден: {input_path}")
        return

    logger.info(f"Начало обработки: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                p_data = data.get("parse", {})
                
                title = p_data.get("title", "Unknown")
                html = p_data.get("text", {}).get("*", "")
                cats = [c["*"].replace("_", " ") for c in p_data.get("categories", [])]

                if html:
                    chunks = parser.article_to_chunks(html, title, cats)
                    
                    # Фильтрация дублей "на лету"
                    for chunk in chunks:
                        content_hash = chunk.page_content.strip()
                        if content_hash not in seen_texts:
                            seen_texts.add(content_hash)
                            all_chunks.append(chunk)
                        else:
                            duplicates_count += 1

                if (i + 1) % 50 == 0:
                    logger.info(f"Обработано {i + 1} статей. Найдено дублей: {duplicates_count}")
            except Exception as e:
                logger.error(f"Ошибка в статье на строке {i}: {e}")

    # Сохранение результатов
    logger.info(f"Итого: {len(all_chunks)} уникальных чанков. Удалено дублей: {duplicates_count}")
    logger.info(f"Сохранение в {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps({
                "text": chunk.page_content,
                "meta": chunk.metadata
            }, ensure_ascii=False) + "\n")

    logger.info("✨ Готово!")

if __name__ == "__main__":
    main()