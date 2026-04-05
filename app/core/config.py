from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    PROJECT_NAME: str = "RAG API"
    
    # --- Внутренние URL для Docker-сети ---
    # API обращается к сервисам по их именам в docker-compose
    QDRANT_URL: str = Field(default="http://qdrant:6333")
    TEI_URL: str = Field(default="http://tei:80")
    REDIS_URL: str = Field(default="redis://redis:6379/0")
    
    # --- RERANKER ---
    RERANKER_ENABLED: bool = Field(default=True)
    RERANKER_TOP_K: int = 5
    # В docker-compose сервис называется vllm-reranker, порт внутри 8000
    RERANKER_URL: str = Field(default="http://vllm-reranker:8000")
    RERANKER_MODEL: str = Field(default="bge-reranker-v2-m3")
    RERANKER_BATCH_SIZE: int = 2 # vLLM хорошо держит батчи побольше

    QUERY_OPTIMIZER_ENABLED: bool = False
    
    # --- Данные и Коллекции ---
    COLLECTION_NAME: str = "warhammer_wiki"
    DATA_PATH: str = "data/processed/processed_chunks.jsonl"
    VECTOR_SIZE: int = 1024  # Для Qwen3-Embedding
    
    # --- Langfuse (для трекинга) ---
    LANGFUSE_PUBLIC_KEY: str = Field(default="")
    LANGFUSE_SECRET_KEY: str = Field(default="")
    LANGFUSE_HOST: str = Field(default="http://langfuse:3000")
    LLM_PROVIDER: str = "gigachat"  # или "openrouter"

    DATASET_PATH: str = "data/eval/warhammer40k_eval_60q.jsonl"
    
    # --- GigaChat ---
    GIGACHAT_CREDENTIALS: str = Field(default="")
    GIGACHAT_MODEL_NAME: str = Field(default="Gigachat")
    
    # --- OpenRouter (если нужно) ---
    OPENROUTER_API_KEY: str = Field(default="")
    LLM_MODEL_NAME: str = "qwen/qwen-2.5-72b-instruct"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore" # Игнорируем лишние переменные (типа паролей БД), которые не нужны в Python
    )

    QDRANT_COLLECTION: str = "warhammer_wiki"

settings = Settings()