"""
LLM provider abstraction layer.

Usage:
    from src.llm import get_llm_client, LLMResponse

    client = get_llm_client()
    response = client.generate("Summarize this article...")
    print(response.content)
"""

from src.llm.base import LLMClient, LLMResponse
from src.llm.factory import get_llm_client, get_available_client
from src.llm.lm_studio import LMStudioClient

__all__ = [
    "LLMClient",
    "LLMResponse",
    "LMStudioClient",
    "get_llm_client",
    "get_available_client",
]
