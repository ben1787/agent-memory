from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from agent_memory.config import MemoryConfig


TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


class Embedder(Protocol):
    dimensions: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_text(self, text: str) -> list[float]:
        ...


@dataclass(slots=True)
class HashEmbedder:
    dimensions: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def embed_text(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in TOKEN_PATTERN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimensions
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            magnitude = 1.0 + min(len(token), 12) / 12.0
            vector[index] += sign * magnitude
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector.tolist()
        return (vector / norm).tolist()


class FastEmbedder:
    def __init__(self, model_name: str, dimensions: int) -> None:
        from fastembed import TextEmbedding

        self.dimensions = dimensions
        self._model = TextEmbedding(model_name=model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for embedding in self._model.embed(texts):
            vector = np.asarray(embedding, dtype=np.float32)
            if vector.shape[0] != self.dimensions:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.dimensions}, "
                    f"received {vector.shape[0]}"
                )
            embeddings.append(vector.tolist())
        return embeddings

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


def build_embedder(config: MemoryConfig) -> Embedder:
    backend = config.embedding_backend.lower()
    if backend == "hash":
        return HashEmbedder(dimensions=config.embedding_dimensions)
    if backend == "fastembed":
        return FastEmbedder(
            model_name=config.embedding_model,
            dimensions=config.embedding_dimensions,
        )
    raise ValueError(
        f"Unknown embedding backend `{config.embedding_backend}`. "
        "Use `fastembed` or `hash`."
    )


def cosine_similarity(left: list[float], right: list[float]) -> float:
    left_arr = np.asarray(left, dtype=np.float32)
    right_arr = np.asarray(right, dtype=np.float32)
    left_norm = float(np.linalg.norm(left_arr))
    right_norm = float(np.linalg.norm(right_arr))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left_arr, right_arr) / (left_norm * right_norm))
