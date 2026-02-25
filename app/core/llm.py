from abc import ABC, abstractmethod
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_gigachat.chat_models import GigaChat
from langchain_openai import ChatOpenAI
from app.core.config import settings

class LLMFactory:
    """Фабрика для создания инстансов LLM"""
    
    @staticmethod
    def get_llm(temperature: float = 0.12) -> BaseChatModel:
        if settings.LLM_PROVIDER == "gigachat":
            return GigaChat(
                credentials=settings.GIGACHAT_CREDENTIALS,
                verify_ssl_certs=False,
                temperature=temperature,
            )
        
        elif settings.LLM_PROVIDER == "openrouter":
            return ChatOpenAI(
                model=settings.LLM_MODEL_NAME,
                openai_api_key=settings.OPENROUTER_API_KEY,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=temperature,
            )
        
        # Если захочешь вернуть Ollama, просто добавишь ветку сюда
        raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")

# Создаем объект для использования
llm_factory = LLMFactory()