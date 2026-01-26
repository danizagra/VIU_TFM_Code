"""
Abstract base class for LLM providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class LLMClient(ABC):
    """
    Abstract interface for LLM providers.

    All LLM clients must implement this interface to ensure
    consistent usage across the application.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name being used."""
        pass

    @abstractmethod
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
            prompt: The user prompt to send to the model.
            system_prompt: Optional system instructions.
            temperature: Sampling temperature (0.0 - 1.0).
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with generated content and metadata.
        """
        pass

    @abstractmethod
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
            temperature: Sampling temperature (0.0 - 1.0).
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with generated content and metadata.
        """
        pass

    def is_available(self) -> bool:
        """
        Check if the LLM service is available.

        Returns:
            True if the service is reachable and working.
        """
        try:
            response = self.generate("Hello", max_tokens=5)
            return bool(response.content)
        except Exception:
            return False
