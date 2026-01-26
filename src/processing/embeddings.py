"""
Embedding generation using SentenceTransformers.

Supports both local SentenceTransformer models and LM Studio embeddings API.
"""

from typing import Optional
import numpy as np
import structlog

from src.config.settings import settings

logger = structlog.get_logger()


class EmbeddingGenerator:
    """
    Generate embeddings for text using SentenceTransformers.

    Usage:
        generator = EmbeddingGenerator()

        # Single text
        embedding = generator.embed_single("Hello world")

        # Batch of texts
        embeddings = generator.embed_texts(["Hello", "World"])
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize embedding generator.

        Args:
            model_name: SentenceTransformer model name (default from settings).
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detected if None.
        """
        self._model_name = model_name or settings.embedding_model
        self._device = device
        self._model = None  # Lazy loading

        logger.info(
            "Initialized EmbeddingGenerator",
            model=self._model_name
        )

    @property
    def model(self):
        """Lazy load the model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading SentenceTransformer model", model=self._model_name)
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device
            )
            logger.info(
                "Model loaded",
                model=self._model_name,
                embedding_dim=self._model.get_sentence_embedding_dimension()
            )
        return self._model

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed.

        Returns:
            Numpy array of shape (embedding_dim,).
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embedding

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            batch_size: Batch size for encoding.
            show_progress: Show progress bar.

        Returns:
            Numpy array of shape (n_texts, embedding_dim).
        """
        if not texts:
            return np.array([])

        logger.info("Generating embeddings", count=len(texts))

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )

        logger.info(
            "Embeddings generated",
            count=len(texts),
            shape=embeddings.shape
        )

        return embeddings

    def embed_articles(
        self,
        articles: list,
        text_field: str = "get_text_for_embedding",
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for a list of article objects.

        Args:
            articles: List of RawArticle or similar objects.
            text_field: Method or attribute name to get text from article.
            batch_size: Batch size for encoding.

        Returns:
            Numpy array of shape (n_articles, embedding_dim).
        """
        texts = []
        for article in articles:
            if callable(getattr(article, text_field, None)):
                text = getattr(article, text_field)()
            else:
                text = getattr(article, text_field, str(article))
            texts.append(text)

        return self.embed_texts(texts, batch_size=batch_size)


class LMStudioEmbeddingGenerator:
    """
    Generate embeddings using LM Studio's embeddings API.

    This is an alternative to SentenceTransformers when LM Studio
    has an embedding model loaded.

    Usage:
        generator = LMStudioEmbeddingGenerator()
        embedding = generator.embed_single("Hello world")
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize LM Studio embedding generator.

        Args:
            base_url: LM Studio server URL (default from settings).
            model: Embedding model name loaded in LM Studio (default from settings).
        """
        from openai import OpenAI

        self._base_url = base_url or settings.lm_studio_base_url
        self._model = model or settings.lm_studio_embedding_model
        self._client = OpenAI(
            base_url=self._base_url,
            api_key="lm-studio"
        )

        logger.info(
            "Initialized LMStudioEmbeddingGenerator",
            base_url=self._base_url,
            model=self._model
        )

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return settings.embedding_dimension

    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed.

        Returns:
            Numpy array of shape (embedding_dim,).
        """
        response = self._client.embeddings.create(
            model=self._model,
            input=text
        )
        return np.array(response.data[0].embedding)

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            batch_size: Batch size for encoding.
            show_progress: Show progress bar (ignored for API).

        Returns:
            Numpy array of shape (n_texts, embedding_dim).
        """
        if not texts:
            return np.array([])

        logger.info("Generating embeddings via LM Studio", count=len(texts))

        # LM Studio supports batch embedding
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._client.embeddings.create(
                model=self._model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        embeddings = np.array(all_embeddings)

        logger.info(
            "Embeddings generated via LM Studio",
            count=len(texts),
            shape=embeddings.shape
        )

        return embeddings

    def embed_articles(
        self,
        articles: list,
        text_field: str = "get_text_for_embedding",
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for a list of article objects.

        Args:
            articles: List of RawArticle or similar objects.
            text_field: Method or attribute name to get text from article.
            batch_size: Batch size for encoding.

        Returns:
            Numpy array of shape (n_articles, embedding_dim).
        """
        texts = []
        for article in articles:
            if callable(getattr(article, text_field, None)):
                text = getattr(article, text_field)()
            else:
                text = getattr(article, text_field, str(article))
            texts.append(text)

        return self.embed_texts(texts, batch_size=batch_size)


def get_embedding_generator(
    use_lm_studio: bool | None = None
) -> EmbeddingGenerator | LMStudioEmbeddingGenerator:
    """
    Factory function to get an embedding generator.

    Args:
        use_lm_studio: If True, use LM Studio embeddings API.
                       If False, use SentenceTransformers.
                       If None, use setting from config (default).

    Returns:
        An embedding generator instance.
    """
    if use_lm_studio is None:
        use_lm_studio = settings.use_lm_studio_embeddings

    if use_lm_studio:
        return LMStudioEmbeddingGenerator()
    return EmbeddingGenerator()
