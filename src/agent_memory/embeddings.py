from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
import shutil
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
class CachedModelEntry:
    model_name: str | None
    hf_source: str | None
    path: Path
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "hf_source": self.hf_source,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
        }


@dataclass(slots=True)
class FastembedCachePruneResult:
    cache_dir: Path
    kept_models: list[str]
    pruned: list[CachedModelEntry]

    @property
    def freed_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.pruned)

    def to_dict(self) -> dict[str, object]:
        return {
            "cache_dir": str(self.cache_dir),
            "kept_models": self.kept_models,
            "pruned": [entry.to_dict() for entry in self.pruned],
            "freed_bytes": self.freed_bytes,
        }


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

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)

    def embed_document(self, text: str) -> list[float]:
        return self.embed_text(text)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text)


def stable_fastembed_cache_dir() -> Path:
    env_override = os.environ.get("FASTEMBED_CACHE_PATH")
    if env_override:
        return Path(env_override)
    return Path.home() / ".cache" / "agent-memory" / "fastembed"


class FastEmbedder:
    def __init__(
        self,
        model_name: str,
        dimensions: int,
        cache_dir: str | Path | None = None,
    ) -> None:
        from fastembed import TextEmbedding

        self.dimensions = dimensions
        resolved_cache = Path(cache_dir) if cache_dir else stable_fastembed_cache_dir()
        resolved_cache.mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(resolved_cache))

    def _collect(self, embeddings) -> list[list[float]]:
        vectors = []
        for embedding in embeddings:
            vector = np.asarray(embedding, dtype=np.float32)
            if vector.shape[0] != self.dimensions:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.dimensions}, "
                    f"received {vector.shape[0]}"
                )
            vectors.append(vector.tolist())
        return vectors

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._collect(self._model.embed(texts))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._collect(self._model.passage_embed(texts))

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._collect(self._model.query_embed(texts))

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_document(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0]


def embed_documents(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    if hasattr(embedder, "embed_documents"):
        return embedder.embed_documents(texts)
    return embedder.embed_texts(texts)


def embed_document(embedder: Embedder, text: str) -> list[float]:
    if hasattr(embedder, "embed_document"):
        return embedder.embed_document(text)
    return embedder.embed_text(text)


def embed_queries(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    if hasattr(embedder, "embed_queries"):
        return embedder.embed_queries(texts)
    return embedder.embed_texts(texts)


def embed_query(embedder: Embedder, text: str) -> list[float]:
    if hasattr(embedder, "embed_query"):
        return embedder.embed_query(text)
    return embedder.embed_text(text)


def fastembed_cache_dir(cache_dir: str | None = None) -> Path:
    if cache_dir:
        from fastembed.common.utils import define_cache_dir

        return define_cache_dir(cache_dir)
    return stable_fastembed_cache_dir()


def _supported_fastembed_hf_sources() -> dict[str, str]:
    from fastembed import TextEmbedding

    supported: dict[str, str] = {}
    for payload in TextEmbedding.list_supported_models():
        model_name = payload.get("model")
        if not isinstance(model_name, str):
            continue
        sources = payload.get("sources")
        if not isinstance(sources, dict):
            continue
        hf_source = sources.get("hf")
        if not isinstance(hf_source, str) or not hf_source:
            continue
        supported[model_name] = hf_source
    return supported


def _hf_cache_dir_name(hf_source: str) -> str:
    return f"models--{hf_source.replace('/', '--')}"


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for candidate in path.rglob("*"):
        if candidate.is_file():
            total += candidate.stat().st_size
    return total


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def prune_fastembed_model_cache(
    keep_model_names: list[str],
    *,
    cache_dir: str | None = None,
) -> FastembedCachePruneResult:
    cache_root = fastembed_cache_dir(cache_dir)
    supported = _supported_fastembed_hf_sources()

    keep_dir_names = {
        _hf_cache_dir_name(hf_source)
        for model_name, hf_source in supported.items()
        if model_name in keep_model_names
    }
    if keep_model_names and not keep_dir_names:
        raise ValueError(
            "None of the requested keep-model names are recognized by fastembed: "
            + ", ".join(keep_model_names)
        )

    model_name_by_dir = {
        _hf_cache_dir_name(hf_source): model_name
        for model_name, hf_source in supported.items()
    }
    hf_source_by_dir = {
        _hf_cache_dir_name(hf_source): hf_source
        for model_name, hf_source in supported.items()
    }

    pruned: list[CachedModelEntry] = []
    for candidate in sorted(cache_root.glob("models--*")):
        if not candidate.is_dir() or candidate.name in keep_dir_names:
            continue
        size_bytes = _dir_size_bytes(candidate)
        lock_path = cache_root / ".locks" / candidate.name
        model_name = model_name_by_dir.get(candidate.name)
        hf_source = hf_source_by_dir.get(candidate.name)
        _remove_path(candidate)
        _remove_path(lock_path)
        pruned.append(
            CachedModelEntry(
                model_name=model_name,
                hf_source=hf_source,
                path=candidate,
                size_bytes=size_bytes,
            )
        )

    return FastembedCachePruneResult(
        cache_dir=cache_root,
        kept_models=keep_model_names,
        pruned=pruned,
    )


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
