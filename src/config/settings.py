"""
Centralized configuration using Pydantic Settings.
Automatically loads from environment variables and .env file.
"""

from enum import Enum
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class LLMProvider(str, Enum):
    """Available LLM providers."""
    LM_STUDIO = "lm_studio"
    DEEPSEEK = "deepseek"
    OPENAI = "openai"


class Settings(BaseSettings):
    """Global application configuration."""

    # === LLM Provider ===
    llm_provider: LLMProvider = Field(
        default=LLMProvider.LM_STUDIO,
        description="Active LLM provider"
    )

    # === LM Studio (Local) ===
    lm_studio_base_url: str = "http://127.0.0.1/v1"
    lm_studio_model: str = "gpt-oss-20b"
    lm_studio_embedding_model: str = "text-embedding-nomic-embed-text-v1.5"

    # === DeepSeek (Cheap Fallback) ===
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"

    # === OpenAI (Optional) ===
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # === News APIs ===
    newsapi_key: str = ""
    gnews_api_key: str = ""

    # === Database ===
    database_url: str = "postgresql://daniel:@localhost:5432/tfm_db"

    # === Agent Configuration ===
    default_language: str = "es"
    default_region: str = "co"
    max_articles_per_fetch: int = 100
    similarity_threshold: float = 0.65
    duplicate_threshold: float = 0.95

    # === Embeddings ===
    embedding_model: str = "text-embedding-nomic-embed-text-v1.5"
    embedding_dimension: int = 768  # nomic-embed-text-v1.5 uses 768 dimensions
    use_lm_studio_embeddings: bool = True  # Use LM Studio API for embeddings

    # === LLM Generation ===
    llm_temperature: float = 0.5
    llm_max_tokens: int = 1000

    # === Telegram Bot ===
    telegram_bot_token: str = ""
    telegram_api_base_url: str = "http://localhost:8000"  # FastAPI URL for bot

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Usage:
        from src.config.settings import get_settings
        settings = get_settings()
        print(settings.database_url)
    """
    return Settings()


# Global instance for convenience
settings = get_settings()
