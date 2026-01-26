"""
LM Studio client for local LLM inference.

LM Studio provides an OpenAI-compatible API, so we use the OpenAI client
to communicate with it.
"""

from typing import Optional

from openai import OpenAI
from openai import APIConnectionError, APIError

from src.config.settings import settings
from src.llm.base import LLMClient, LLMResponse


class LMStudioClient(LLMClient):
    """
    Client for LM Studio running locally.

    LM Studio exposes an OpenAI-compatible API at http://127.0.0.1/v1
    This client uses the OpenAI SDK to communicate with it.

    Usage:
        client = LMStudioClient()
        response = client.generate("Summarize this news article...")
        print(response.content)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize LM Studio client.

        Args:
            base_url: LM Studio server URL (default from settings).
            model: Model name to use (default from settings).
        """
        self._base_url = base_url or settings.lm_studio_base_url
        self._model = model or settings.lm_studio_model

        self._client = OpenAI(
            base_url=self._base_url,
            api_key="lm-studio"  # LM Studio doesn't require a real API key
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
            ConnectionError: If LM Studio is not running.
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
            ConnectionError: If LM Studio is not running.
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

        except APIConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to LM Studio at {self._base_url}. "
                "Make sure LM Studio is running and the server is started."
            ) from e
        except APIError as e:
            raise RuntimeError(f"LM Studio API error: {e}") from e

    def is_available(self) -> bool:
        """Check if LM Studio is running and responsive."""
        try:
            # Try to list models as a health check
            self._client.models.list()
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """
        List available models in LM Studio.

        Returns:
            List of model IDs.
        """
        try:
            response = self._client.models.list()
            return [model.id for model in response.data]
        except Exception:
            return []
