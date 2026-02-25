from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    PROJECT_NAME: str = "Warhammer RAG API"
    
    # --- Внутренние URL для Docker-сети ---
    # API обращается к сервисам по их именам в docker-compose
    QDRANT_URL: str = Field(default="http://qdrant:6333")
    TEI_URL: str = Field(default="http://tei:80")
    REDIS_URL: str = Field(default="redis://redis:6379/0")
    
    # --- Данные и Коллекции ---
    COLLECTION_NAME: str = "warhammer_wiki"
    DATA_PATH: str = "data/processed/processed_chunks.jsonl"
    VECTOR_SIZE: int = 1024  # Для Qwen3-Embedding
    
    # --- Langfuse (для трекинга) ---
    LANGFUSE_PUBLIC_KEY: str = Field(default="")
    LANGFUSE_SECRET_KEY: str = Field(default="")
    LANGFUSE_HOST: str = Field(default="http://langfuse:3000")

    LLM_PROVIDER: str = "gigachat"  # или "openrouter"
    
    # --- GigaChat ---
    GIGACHAT_CREDENTIALS: str = Field(default="")
    
    # --- OpenRouter (если нужно) ---
    OPENROUTER_API_KEY: str = Field(default="")
    LLM_MODEL_NAME: str = "qwen/qwen-2.5-72b-instruct"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore" # Игнорируем лишние переменные (типа паролей БД), которые не нужны в Python
    )

settings = Settings()