from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from agent_memory.benchmark import (
    BenchmarkCase,
    extract_excerpt,
    parse_title,
    score_case,
)
from agent_memory.embeddings import Embedder, cosine_similarity
from agent_memory.engine import AgentMemory


TOKEN_PATTERN = re.compile(r"\S+")
CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")
DEFAULT_ANSWER_MODEL_ID = "google/flan-t5-base"


@dataclass(slots=True)
class ContextItem:
    reference_id: str
    title: str
    text: str
    score: float
    locator: str | None = None

    @property
    def excerpt(self) -> str:
        return extract_excerpt(self.text, 220)

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_id": self.reference_id,
            "title": self.title,
            "locator": self.locator,
            "score": round(self.score, 4),
            "excerpt": self.excerpt,
        }


class RawCorpusRetriever:
    def __init__(self, paragraphs: list[dict[str, object]], embedder: Embedder) -> None:
        self.paragraphs = paragraphs
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int) -> list[ContextItem]:
        query_embedding = self.embedder.embed_text(query)
        scored: list[tuple[float, dict[str, object]]] = []
        for paragraph in self.paragraphs:
            similarity = cosine_similarity(query_embedding, paragraph["embedding"])
            scored.append((similarity, paragraph))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            ContextItem(
                reference_id=item[1]["reference_id"],
                title=item[1]["title"],
                text=item[1]["text"],
                score=round(item[0], 4),
                locator=item[1]["locator"],
            )
            for item in scored[:top_k]
        ]


class LocalSeq2SeqAnswerer:
    def __init__(
        self,
        model_id: str = DEFAULT_ANSWER_MODEL_ID,
        max_input_tokens: int = 512,
        max_new_tokens: int = 96,
    ) -> None:
        self.model_id = model_id
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def warmup(self) -> None:
        self.generate(
            "What is graph theory?",
            [
                ContextItem(
                    reference_id="mem_demo",
                    title="Graph theory",
                    text="Source: Graph theory\nGraph theory is the study of graphs in mathematics and computer science.",
                    score=1.0,
                    locator=None,
                )
            ],
        )

    def build_prompt(self, query: str, contexts: list[ContextItem]) -> str:
        lines = [
            "Answer the question using only the provided context.",
            "Synthesize across snippets when useful.",
            "If the context is insufficient, say so briefly.",
            "After the answer, add a line that begins with `References:` and list only the source references you actually used.",
            "Use the exact reference labels from the context, for example `[mem_123abc]` or `[Graph theory ¶1]`.",
            f"Question: {query}",
            "Context:",
        ]
        for index, item in enumerate(contexts, start=1):
            lines.append(f"[{item.reference_id}] {item.title}: {item.text}")
        lines.append("Answer format:")
        lines.append("Answer: <your answer>")
        lines.append("References: [ref1], [ref2]")
        return "\n".join(lines)

    def generate(self, query: str, contexts: list[ContextItem]) -> dict[str, object]:
        prompt = self.build_prompt(query, contexts)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        prompt_tokens = int(encoded["input_ids"].shape[1])
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        start = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generation_ms = (time.perf_counter() - start) * 1000
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        answer_tokens = len(TOKEN_PATTERN.findall(answer))
        return {
            "answer": answer,
            "prompt_tokens": prompt_tokens,
            "answer_tokens": answer_tokens,
            "generation_ms": round(generation_ms, 3),
        }


def flatten_paragraphs(corpus: dict[str, list[str]], embedder: Embedder) -> list[dict[str, object]]:
    paragraphs: list[dict[str, object]] = []
    texts: list[str] = []
    for title, items in corpus.items():
        for paragraph_index, paragraph in enumerate(items, start=1):
            texts.append(paragraph)
            paragraphs.append(
                {
                    "title": title,
                    "text": paragraph,
                    "reference_id": f"{title} ¶{paragraph_index}",
                    "locator": f"¶{paragraph_index}",
                }
            )
    embeddings = embedder.embed_texts(texts)
    for paragraph, embedding in zip(paragraphs, embeddings):
        paragraph["embedding"] = embedding
    return paragraphs


def load_cached_corpus(cache_file: Path) -> dict[str, list[str]]:
    return json.loads(cache_file.read_text(encoding='utf-8'))


def build_graph_context(
    memory: AgentMemory,
    query: str,
    *,
    max_clusters: int,
    max_hits_per_cluster: int,
) -> tuple[list[ContextItem], dict[str, object]]:
    retrieval_started = time.perf_counter()
    recall = memory.recall(query, max_clusters=max_clusters)
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
    items: list[ContextItem] = []
    for cluster in recall.clusters[:max_clusters]:
        seen_titles: set[str] = set()
        for hit in cluster.hits:
            title = parse_title(hit.text)
            if title in seen_titles:
                continue
            seen_titles.add(title)
            items.append(
                ContextItem(
                    reference_id=hit.memory_id,
                    title=title,
                    text=hit.text,
                    score=hit.query_similarity,
                    locator=None,
                )
            )
            if len(seen_titles) >= max_hits_per_cluster:
                break
    return items, {
        "retrieval_ms": round(retrieval_ms, 3),
        "cluster_count": len(recall.clusters),
        "cluster_sizes": [len(cluster.memory_ids) for cluster in recall.clusters],
    }


def build_raw_context(
    retriever: RawCorpusRetriever,
    query: str,
    *,
    top_k: int,
) -> tuple[list[ContextItem], dict[str, object]]:
    retrieval_started = time.perf_counter()
    items = retriever.retrieve(query, top_k=top_k)
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
    return items, {
        "retrieval_ms": round(retrieval_ms, 3),
        "retrieved_titles": [item.title for item in items],
    }


def context_token_count(items: list[ContextItem]) -> int:
    return sum(len(TOKEN_PATTERN.findall(item.text)) for item in items)


def extract_cited_references(answer: str, contexts: list[ContextItem]) -> list[str]:
    available = {item.reference_id for item in contexts}
    cited: list[str] = []
    for candidate in CITATION_PATTERN.findall(answer):
        if candidate in available and candidate not in cited:
            cited.append(candidate)
    return cited


def infer_supporting_references(answer: str, contexts: list[ContextItem], limit: int = 4) -> list[str]:
    answer_tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]+", answer)
        if len(token) >= 4
    }
    if not answer_tokens:
        return [item.reference_id for item in contexts[:limit]]

    scored: list[tuple[int, float, str]] = []
    for item in contexts:
        context_tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]+", item.text)
            if len(token) >= 4
        }
        overlap = len(answer_tokens & context_tokens)
        scored.append((overlap, item.score, item.reference_id))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    inferred = [reference_id for overlap, _, reference_id in scored if overlap > 0][:limit]
    if inferred:
        return inferred
    return [item.reference_id for item in contexts[:limit]]


def context_score(case: BenchmarkCase, items: list[ContextItem]) -> dict[str, object]:
    titles = {item.title for item in items}
    matched_expected = [title for title in case.expected_titles if title in titles]
    matched_forbidden = [title for title in case.forbidden_titles if title in titles]
    combined_text = "\n".join(item.text.lower() for item in items)
    matched_required = [
        term for term in case.required_terms if term.lower() in combined_text
    ]
    top_cluster_recall = len(matched_expected) / len(case.expected_titles) if case.expected_titles else 1.0
    overall_recall = top_cluster_recall
    forbidden_clean_score = (
        1.0 - (len(matched_forbidden) / len(case.forbidden_titles))
        if case.forbidden_titles
        else 1.0
    )
    cluster_requirement_score = 1.0
    required_term_recall = (
        len(matched_required) / len(case.required_terms)
        if case.required_terms
        else 1.0
    )
    aggregate_score = score_case(
        kind=case.kind,
        top_cluster_recall=top_cluster_recall,
        overall_recall=overall_recall,
        forbidden_clean_score=forbidden_clean_score,
        cluster_requirement_score=cluster_requirement_score,
        required_term_recall=required_term_recall,
    )
    return {
        "matched_expected_titles": matched_expected,
        "matched_forbidden_titles": matched_forbidden,
        "matched_required_terms": matched_required,
        "required_term_recall": round(required_term_recall, 4),
        "context_score": round(aggregate_score, 4),
    }


def evaluate_path(
    *,
    label: str,
    case: BenchmarkCase,
    contexts: list[ContextItem],
    retrieval_meta: dict[str, object],
    answerer: LocalSeq2SeqAnswerer,
) -> dict[str, object]:
    answer_payload = answerer.generate(case.query, contexts)
    retrieval_ms = float(retrieval_meta["retrieval_ms"])
    generation_ms = float(answer_payload["generation_ms"])
    support = context_score(case, contexts)
    cited_references = extract_cited_references(answer_payload["answer"], contexts)
    inferred_references = infer_supporting_references(answer_payload["answer"], contexts)
    return {
        "system": label,
        "case_id": case.case_id,
        "query": case.query,
        "expected_count": len(case.expected_titles),
        "retrieval_ms": round(retrieval_ms, 3),
        "generation_ms": round(generation_ms, 3),
        "total_ms": round(retrieval_ms + generation_ms, 3),
        "context_tokens": context_token_count(contexts),
        "context_titles": [item.title for item in contexts],
        "context_excerpts": [f"{item.title}: {item.excerpt}" for item in contexts],
        "context_references": [item.to_dict() for item in contexts],
        "prompt_tokens": answer_payload["prompt_tokens"],
        "answer_tokens": answer_payload["answer_tokens"],
        "answer": answer_payload["answer"],
        "cited_references": cited_references,
        "inferred_references": inferred_references,
        "display_references": cited_references or inferred_references,
        **support,
        **retrieval_meta,
    }


def summarize_results(results: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {
        "average_retrieval_ms": round(
            sum(float(item["retrieval_ms"]) for item in results) / len(results), 3
        ),
        "average_generation_ms": round(
            sum(float(item["generation_ms"]) for item in results) / len(results), 3
        ),
        "average_total_ms": round(
            sum(float(item["total_ms"]) for item in results) / len(results), 3
        ),
        "average_context_tokens": round(
            sum(int(item["context_tokens"]) for item in results) / len(results), 1
        ),
        "average_prompt_tokens": round(
            sum(int(item["prompt_tokens"]) for item in results) / len(results), 1
        ),
        "average_context_score": round(
            sum(float(item["context_score"]) for item in results) / len(results), 4
        ),
    }
    by_level: dict[int, list[dict[str, object]]] = defaultdict(list)
    for item in results:
        level = int(item["case_id"].split("-")[-1])
        by_level[level].append(item)
    summary["levels"] = {
        str(level): {
            "average_total_ms": round(
                sum(float(item["total_ms"]) for item in items) / len(items), 3
            ),
            "average_generation_ms": round(
                sum(float(item["generation_ms"]) for item in items) / len(items), 3
            ),
            "average_context_tokens": round(
                sum(int(item["context_tokens"]) for item in items) / len(items), 1
            ),
            "average_context_score": round(
                sum(float(item["context_score"]) for item in items) / len(items), 4
            ),
        }
        for level, items in sorted(by_level.items())
    }
    return summary
