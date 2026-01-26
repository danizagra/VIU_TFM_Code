"""
DeepSeek API client (prepared for future use).

DeepSeek provides an OpenAI-compatible API, making integration straightforward.
Cost: ~$0.14/1M input tokens, ~$0.28/1M output tokens (very cheap fallback).
"""

from typing import Optional

from openai import OpenAI
from openai import APIConnectionError, APIError, AuthenticationError

from src.config.settings import settings
from src.llm.base import LLMClient, LLMResponse


class DeepSeekClient(LLMClient):
    """
    Client for DeepSeek API.

    DeepSeek offers an OpenAI-compatible API at https://api.deepseek.com
    This is a cost-effective fallback when local LLM is not available.

    Usage:
        client = DeepSeekClient()
        response = client.generate("Summarize this news article...")
        print(response.content)
    """

    API_BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize DeepSeek client.

        Args:
            api_key: DeepSeek API key (default from settings).
            model: Model name to use (default from settings).
        """
        self._api_key = api_key or settings.deepseek_api_key
        self._model = model or settings.deepseek_model

        if not self._api_key or self._api_key.startswith("sk-xxx"):
            raise ValueError(
                "DeepSeek API key not configured. "
                "Set DEEPSEEK_API_KEY in your .env file."
            )

        self._client = OpenAI(
            base_url=self.API_BASE_URL,
            api_key=self._api_key
        )

    @property
    def model_name(self) -> str:
        """Return the model name being used."""
        return self._model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 1000
    ) -> LLMResponse:
        """
        Generate text from a prompt.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system instructions.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with generated content.

        Raises:
            ValueError: If API key is not configured.
            ConnectionError: If API is unreachable.
            RuntimeError: If generation fails.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self.chat(messages, temperature, max_tokens)

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.5,
        max_tokens: int = 1000
    ) -> LLMResponse:
        """
        Chat with message history.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with generated content.

        Raises:
            AuthenticationError: If API key is invalid.
            ConnectionError: If API is unreachable.
            RuntimeError: If generation fails.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            choice = response.choices[0]
            usage = response.usage

            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None
            )

        except AuthenticationError as e:
            raise ValueError(
                "Invalid DeepSeek API key. Check your DEEPSEEK_API_KEY."
            ) from e
        except APIConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to DeepSeek API at {self.API_BASE_URL}."
            ) from e
        except APIError as e:
            raise RuntimeError(f"DeepSeek API error: {e}") from e

    def is_available(self) -> bool:
        """Check if DeepSeek API is reachable and key is valid."""
        try:
            # Small test generation
            self.generate("Hi", max_tokens=5)
            return True
        except Exception:
            return False
