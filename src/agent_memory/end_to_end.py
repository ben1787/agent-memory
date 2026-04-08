from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable

# Reason: the answerer is now an OpenAI Agents SDK loop, not a local
# seq2seq model. Embeddings still run on the local fastembed model
# (see Embedder), since per the project design those happen continuously
# in the background on both write and query and need to stay cheap and
# offline. Only the answering agent is remote.
from agents import Agent, ModelSettings, RunConfig, Runner, function_tool
from pydantic import BaseModel, Field

from agent_memory.benchmark import (
    BenchmarkCase,
    parse_title,
    score_case,
)
from agent_memory.embeddings import Embedder, cosine_similarity
from agent_memory.engine import AgentMemory


TOKEN_PATTERN = re.compile(r"\S+")
CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")
# Reason: default OpenAI model for the answering agent. The user
# explicitly chose gpt-5.4 — do not silently substitute a different
# model if it errors; surface the error so we know.
DEFAULT_ANSWER_MODEL_ID = "gpt-5.4"


@dataclass
class ContextItem:
    reference_id: str
    title: str
    text: str
    score: float
    locator: str | None = None

    def to_dict(self) -> dict[str, object]:
        # Reason: report the full paragraph body, not a 220-char excerpt.
        # The excerpt was a per-row display string used by the HTML report;
        # for analysis we need the full text the agent actually saw so we
        # can verify retrieval coverage of required terms that live deep
        # in a paragraph (not just in the first sentence).
        return {
            "reference_id": self.reference_id,
            "title": self.title,
            "locator": self.locator,
            "score": round(self.score, 4),
            "text": self.text,
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


class JudgeScore(BaseModel):
    # Reason: 4-criterion rubric synthesized from MT-Bench (Zheng et al. 2023)
    # and G-Eval (Liu et al. 2023). Each score is 1-5 where 5 is excellent.
    completeness: int = Field(ge=1, le=5, description="Does it address every part of the question?")
    accuracy: int = Field(ge=1, le=5, description="Are the factual claims correct?")
    depth: int = Field(ge=1, le=5, description="Concrete, specific detail vs vague generalities?")
    clarity: int = Field(ge=1, le=5, description="Well-organized and easy to follow?")


class JudgeVerdict(BaseModel):
    # Reason: forces the judge model to score each candidate independently
    # before declaring a winner, so position bias and "halo effect" rationale
    # are mitigated. The structured-output schema also gives us deterministic
    # parsing instead of regex over free-form text.
    score_a: JudgeScore
    score_b: JudgeScore
    score_c: JudgeScore
    overall_winner: str = Field(description="One of: A, B, C, or tie")
    reasoning: str = Field(description="Brief justification for the winner")


JUDGE_INSTRUCTIONS = (
    "You are an expert judge evaluating answers to research questions. "
    "You will see a question followed by three candidate answers labeled A, B, and C. "
    "The answers were produced by different retrieval systems but you do NOT know which "
    "is which, and the order is randomized. Judge them only on their content.\n\n"
    "Score each answer on four criteria using a 1 to 5 scale "
    "(1 = poor, 3 = adequate, 5 = excellent):\n"
    "  completeness: does it address every part of the question?\n"
    "  accuracy: are its factual claims correct based on general knowledge?\n"
    "  depth: does it provide concrete specific detail rather than vague generalities?\n"
    "  clarity: is it well-organized and easy to follow?\n\n"
    "Score each answer independently first, then declare an overall winner "
    "(A, B, C, or 'tie' if two or more are clearly equivalent). "
    "Keep the reasoning to 2-3 sentences."
)


def build_judge_agent(model: str = DEFAULT_ANSWER_MODEL_ID) -> Agent[Any]:
    # Reason: judge agent has no tools — it just reads the prompt and emits
    # structured output. Same Agents-SDK plumbing as the answering agents so
    # we can reuse the run_async + RunConfig path.
    return Agent(
        name="Answer Judge",
        instructions=JUDGE_INSTRUCTIONS,
        model=model,
        output_type=JudgeVerdict,
    )


class AnswerFormat(BaseModel):
    # Reason: structured output so we can reliably extract the plain-text
    # answer and the list of reference IDs the agent claims to have used.
    # Used as `output_type` on every benchmark Agent so all three modes
    # (graph / cosine / raw) emit the same shape.
    answer: str
    references: list[str] = Field(default_factory=list)


@dataclass
class HitRecord:
    # Reason: uniform in-process record of every hit an agent retrieved
    # across all tool calls for a given case. The tool closure appends to
    # a per-case list; after the Runner finishes we dedupe and convert to
    # ContextItem so the existing context_score pipeline keeps working.
    reference_id: str
    title: str
    text: str
    score: float
    locator: str | None = None


@dataclass
class AgentRunResult:
    answer: str
    hits: list[HitRecord]
    generation_ms: float
    input_tokens: int
    output_tokens: int
    tool_call_count: int
    model_turns: int
    references_from_model: list[str] = field(default_factory=list)


class OpenAIAgentAnswerer:
    # Reason: thin wrapper around `agents.Runner.run_sync` so each
    # retrieval mode (graph / cosine / raw) can build its own Agent with
    # its own tool(s) but share the same run/collection/timing/trace
    # logic. The caller is responsible for building the Agent (because
    # the tool closure captures per-case state) and passing it in.

    def __init__(self, *, model: str = DEFAULT_ANSWER_MODEL_ID, max_turns: int = 15) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY must be set before running the OpenAI agent answerer."
            )
        self.model = model
        self.max_turns = max_turns
        # Reason: `store=False` keeps these benchmark runs out of the
        # OpenAI dashboard retention, `tracing_disabled=True` avoids a
        # dependency on the hosted trace viewer.
        self.run_config = RunConfig(
            model=model,
            tracing_disabled=True,
            model_settings=ModelSettings(store=False),
        )

    def run(self, agent: Agent[Any], query: str, hits: list[HitRecord]) -> AgentRunResult:
        started = time.perf_counter()
        result = Runner.run_sync(agent, query, run_config=self.run_config, max_turns=self.max_turns)
        generation_ms = (time.perf_counter() - started) * 1000
        return self._build_result(result, hits, generation_ms)

    async def run_async(self, agent: Agent[Any], query: str, hits: list[HitRecord]) -> AgentRunResult:
        # Reason: async sibling of `run()` so the benchmark orchestrator can
        # `asyncio.gather` over many (agent, case) pairs and get all OpenAI
        # round trips in flight at once. The work is identical otherwise —
        # same RunConfig, same result-shape extraction.
        started = time.perf_counter()
        result = await Runner.run(agent, query, run_config=self.run_config, max_turns=self.max_turns)
        generation_ms = (time.perf_counter() - started) * 1000
        return self._build_result(result, hits, generation_ms)

    def _build_result(self, result: Any, hits: list[HitRecord], generation_ms: float) -> AgentRunResult:
        final = result.final_output
        if isinstance(final, AnswerFormat):
            answer_text = final.answer
            model_refs = list(final.references)
        else:
            # Reason: if the agent returned a bare string (some models
            # skip structured output on short answers) fall back to it
            # rather than crashing — the retrieval-side metrics are still
            # valid even if the structured output is missing.
            answer_text = str(final) if final is not None else ""
            model_refs = []

        # Reason: pull per-turn usage and tool-call counts off the Runner
        # result for trace reporting. None-safe because different
        # Runner/response shapes may omit usage on tool-only turns.
        tool_call_count = 0
        for item in result.new_items:
            if type(item).__name__ == "ToolCallItem":
                tool_call_count += 1
        input_tokens = 0
        output_tokens = 0
        for response in result.raw_responses:
            usage = getattr(response, "usage", None)
            if usage is None:
                continue
            input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

        return AgentRunResult(
            answer=answer_text,
            hits=hits,
            generation_ms=generation_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_call_count=tool_call_count,
            model_turns=len(result.raw_responses),
            references_from_model=model_refs,
        )


def hits_to_context_items(hits: list[HitRecord]) -> list[ContextItem]:
    # Reason: the agent may call its retrieval tool several times and
    # return duplicate hits across calls. Dedupe by reference_id while
    # keeping the best (highest) score seen for that ID, then project
    # into ContextItem so the downstream `context_score` / `evaluate_path`
    # code keeps working unchanged.
    best: dict[str, HitRecord] = {}
    order: list[str] = []
    for hit in hits:
        existing = best.get(hit.reference_id)
        if existing is None:
            best[hit.reference_id] = hit
            order.append(hit.reference_id)
        elif hit.score > existing.score:
            best[hit.reference_id] = hit
    return [
        ContextItem(
            reference_id=best[rid].reference_id,
            title=best[rid].title,
            text=best[rid].text,
            score=best[rid].score,
            locator=best[rid].locator,
        )
        for rid in order
    ]


ANSWER_INSTRUCTIONS_SUFFIX = (
    "Return a structured result with two fields: `answer` (a short plain-English "
    "response that directly answers the question) and `references` (the list of "
    "reference labels you actually relied on — use the exact IDs returned by your "
    "tool calls, e.g. `mem_123abc` or `Graph theory ¶1`)."
)


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
# Reason: AgentMemory.save rejects text >1000 words. Wikipedia paragraphs
# occasionally exceed that (one in the cached corpus is 1243 words). Per
# user instruction we don't lower or bypass the cap — instead we split
# oversized paragraphs into roughly equal chunks at sentence boundaries
# closest to target_words = ceil(words/1000) target.
_MAX_PARAGRAPH_WORDS = 1000


def _split_paragraph_at_sentences(paragraph: str) -> list[str]:
    word_count = len(paragraph.split())
    if word_count <= _MAX_PARAGRAPH_WORDS:
        return [paragraph]
    sentences = _SENTENCE_END_RE.split(paragraph.strip())
    if len(sentences) < 2:
        # Reason: nothing to split on — return as-is and let the engine
        # raise so the corpus author can fix the source rather than us
        # silently producing a chunk that still trips the cap.
        return [paragraph]
    n_chunks = math.ceil(word_count / _MAX_PARAGRAPH_WORDS)
    target = word_count / n_chunks
    sentence_word_counts = [len(s.split()) for s in sentences]
    cumulative: list[int] = []
    running = 0
    for n in sentence_word_counts:
        running += n
        cumulative.append(running)
    # Reason: pick the sentence boundary whose cumulative word count is
    # closest to target*1, target*2, ... target*(n-1). Force monotonicity
    # so we never pick the same boundary twice for adjacent chunks.
    split_indices: list[int] = []
    last = 0
    for k in range(1, n_chunks):
        goal = target * k
        best_idx = last
        best_diff = abs(cumulative[last] - goal)
        for i in range(last + 1, len(cumulative) - 1):
            diff = abs(cumulative[i] - goal)
            if diff < best_diff:
                best_idx = i
                best_diff = diff
        split_indices.append(best_idx)
        last = best_idx + 1
    chunks: list[str] = []
    start = 0
    for idx in split_indices:
        chunks.append(" ".join(sentences[start : idx + 1]))
        start = idx + 1
    chunks.append(" ".join(sentences[start:]))
    return chunks


def flatten_paragraphs(corpus: dict[str, list[str]], embedder: Embedder) -> list[dict[str, object]]:
    paragraphs: list[dict[str, object]] = []
    texts: list[str] = []
    for title, items in corpus.items():
        for paragraph_index, paragraph in enumerate(items, start=1):
            chunks = _split_paragraph_at_sentences(paragraph)
            for chunk_index, chunk in enumerate(chunks, start=1):
                # Reason: keep the original paragraph locator stable for
                # single-chunk paragraphs (the common case) and only
                # disambiguate split chunks with a sub-index, so existing
                # benchmark cases that reference "¶N" still match.
                if len(chunks) == 1:
                    locator = f"¶{paragraph_index}"
                    reference_id = f"{title} ¶{paragraph_index}"
                else:
                    locator = f"¶{paragraph_index}.{chunk_index}"
                    reference_id = f"{title} ¶{paragraph_index}.{chunk_index}"
                texts.append(chunk)
                paragraphs.append(
                    {
                        "title": title,
                        "text": chunk,
                        "reference_id": reference_id,
                        "locator": locator,
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


def build_cosine_context(
    memory: AgentMemory,
    query: str,
    *,
    top_k: int,
) -> tuple[list[ContextItem], dict[str, object]]:
    # Reason: third retrieval path that bypasses the graph entirely and
    # ranks memories purely by cosine similarity to the query. Exists so
    # the isolated-agents benchmark can compare PPV graph spreading
    # against flat top-N nearest-neighbor retrieval on the same store
    # and embedder, with the same downstream answerer.
    retrieval_started = time.perf_counter()
    recall = memory.recall_cosine(query, limit=top_k)
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
    items: list[ContextItem] = []
    seen_titles: set[str] = set()
    for hit in recall.hits:
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
    return items, {
        "retrieval_ms": round(retrieval_ms, 3),
        "retrieved_titles": [item.title for item in items],
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
    agent_result: AgentRunResult,
) -> dict[str, object]:
    # Reason: retrieval and generation timings are tracked separately
    # even in the agent-loop world. retrieval_meta["retrieval_ms"] is
    # the sum of wall-clock time spent inside tool calls; agent_result
    # generation_ms is the full Runner wall clock (includes tool time).
    # We report generation_ms - retrieval_ms as "pure" generation time.
    retrieval_ms = float(retrieval_meta["retrieval_ms"])
    total_ms = float(agent_result.generation_ms)
    generation_ms = max(total_ms - retrieval_ms, 0.0)
    support = context_score(case, contexts)
    cited_references = extract_cited_references(agent_result.answer, contexts)
    inferred_references = infer_supporting_references(agent_result.answer, contexts)
    return {
        "system": label,
        "case_id": case.case_id,
        "query": case.query,
        "expected_count": len(case.expected_titles),
        "retrieval_ms": round(retrieval_ms, 3),
        "generation_ms": round(generation_ms, 3),
        "total_ms": round(total_ms, 3),
        "context_tokens": context_token_count(contexts),
        "context_titles": [item.title for item in contexts],
        "context_excerpts": [f"{item.title}: {item.text}" for item in contexts],
        "context_references": [item.to_dict() for item in contexts],
        "prompt_tokens": agent_result.input_tokens,
        "answer_tokens": agent_result.output_tokens,
        "answer": agent_result.answer,
        "cited_references": cited_references,
        "inferred_references": inferred_references,
        "display_references": cited_references or inferred_references,
        "agent_references": agent_result.references_from_model,
        "tool_call_count": agent_result.tool_call_count,
        "model_turns": agent_result.model_turns,
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
        # Reason: surface tool-call iteration as a first-class metric in
        # the summary so the comparison table can show it next to scores.
        "average_tool_calls": round(
            sum(int(item["tool_call_count"]) for item in results) / len(results), 2
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
