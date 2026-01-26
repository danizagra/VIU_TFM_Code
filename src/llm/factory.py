"""
Factory for creating LLM clients based on configuration.
"""

from typing import Optional
import structlog

from src.config.settings import settings, LLMProvider
from src.llm.base import LLMClient
from src.llm.lm_studio import LMStudioClient


logger = structlog.get_logger()


def get_llm_client(provider: Optional[LLMProvider] = None) -> LLMClient:
    """
    Get an LLM client based on the configured provider.

    Args:
        provider: Override the default provider from settings.

    Returns:
        An LLMClient instance ready to use.

    Raises:
        ValueError: If the provider is not supported or not configured.

    Usage:
        from src.llm.factory import get_llm_client

        client = get_llm_client()
        response = client.generate("Summarize this article...")
        print(response.content)
    """
    provider = provider or settings.llm_provider

    if provider == LLMProvider.LM_STUDIO:
        logger.info("Initializing LM Studio client", model=settings.lm_studio_model)
        return LMStudioClient()

    elif provider == LLMProvider.DEEPSEEK:
        # Import here to avoid initialization errors if not configured
        from src.llm.deepseek import DeepSeekClient
        logger.info("Initializing DeepSeek client", model=settings.deepseek_model)
        return DeepSeekClient()

    elif provider == LLMProvider.OPENAI:
        # Future: implement OpenAI client if needed
        raise NotImplementedError(
            "OpenAI provider not yet implemented. "
            "Use 'lm_studio' or 'deepseek' instead."
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_available_client() -> Optional[LLMClient]:
    """
    Get the first available LLM client.

    Tries providers in order of preference:
    1. LM Studio (local, free)
    2. DeepSeek (API, cheap)

    Returns:
        An available LLMClient, or None if none are available.
    """
    # Try LM Studio first (local, free)
    try:
        client = LMStudioClient()
        if client.is_available():
            logger.info("Using LM Studio (local)")
            return client
    except Exception as e:
        logger.debug("LM Studio not available", error=str(e))

    # Try DeepSeek as fallback
    try:
        from src.llm.deepseek import DeepSeekClient
        client = DeepSeekClient()
        if client.is_available():
            logger.info("Using DeepSeek API (fallback)")
            return client
    except Exception as e:
        logger.debug("DeepSeek not available", error=str(e))

    logger.warning("No LLM provider available")
    return None
