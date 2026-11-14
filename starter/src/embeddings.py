"""Local multilingual embedding adapter using Sentence Transformers."""
import math
from typing import Protocol


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Load a local Sentence Transformers model for document and query vectors."""

    def __init__(self, model_name: str) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Sentence Transformers is not installed; install starter/requirements.txt"
            ) from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        dimensions = self._model.get_embedding_dimension()
        if not dimensions:
            raise RuntimeError(f"Could not determine embedding size for {model_name}")
        self.dimensions = int(dimensions)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if any(not text.strip() for text in texts):
            raise ValueError("Texts must not be empty")
        if not texts:
            return []

        encoded = self._model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = encoded.tolist()
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding model returned {len(vectors)} vectors for {len(texts)} texts"
            )
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise RuntimeError(
                    f"Embedding model returned {len(vector)} dimensions; "
                    f"expected {self.dimensions}"
                )
            if not all(math.isfinite(float(value)) for value in vector):
                raise RuntimeError("Embedding model returned a non-finite value")
        return [[float(value) for value in vector] for vector in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, question: str) -> list[float]:
        vectors = self._embed([question])
        return vectors[0]

    def embed_queries(self, questions: list[str]) -> list[list[float]]:
        return self._embed(questions)
