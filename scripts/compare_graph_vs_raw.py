from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random
import tempfile
import time
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agent_memory.benchmark import (
    ARTICLE_GROUPS,
    BENCHMARK_CASES,
    TITLE_PREFIX,
    BenchmarkCase,
    build_benchmark_config,
    extract_excerpt,
    extract_paragraphs,
    fetch_article_extract,
    fetch_article_extracts,
    load_wikipedia_corpus,
    parse_title,
    score_case,
)
from agent_memory.embeddings import Embedder, build_embedder, cosine_similarity
from agent_memory.engine import AgentMemory


WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "agent-memory-compare/0.1"


class RawCorpusRetriever:
    def __init__(self, paragraphs: list[dict[str, object]], embedder: Embedder) -> None:
        self.paragraphs = paragraphs
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 12) -> dict[str, object]:
        query_embedding = self.embedder.embed_text(query)
        scored = []
        for paragraph in self.paragraphs:
            similarity = cosine_similarity(query_embedding, paragraph["embedding"])
            scored.append((similarity, paragraph))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[:top_k]
        title_counts = Counter(item[1]["title"] for item in selected)
        top_hits = [
            {
                "title": item[1]["title"],
                "query_similarity": round(item[0], 4),
                "preview": item[1]["preview"],
                "excerpt": item[1]["excerpt"],
            }
            for item in selected[:6]
        ]
        draft_answer_parts = []
        seen_titles: set[str] = set()
        for hit in top_hits:
            if hit["title"] in seen_titles:
                continue
            seen_titles.add(hit["title"])
            draft_answer_parts.append(f"{hit['title']}: {hit['excerpt']}")
            if len(draft_answer_parts) >= 3:
                break
        return {
            "title_counts": dict(title_counts),
            "top_hits": top_hits,
            "texts": [item[1]["text"] for item in selected],
            "draft_answer": " ".join(draft_answer_parts),
        }


def fetch_random_titles(count: int, exclude_titles: set[str]) -> list[str]:
    titles: set[str] = set()
    while len(titles) < count:
        remaining = count - len(titles)
        batch = min(50, remaining)
        params = {
            "action": "query",
            "list": "random",
            "rnnamespace": 0,
            "rnlimit": batch,
            "format": "json",
        }
        request = Request(
            f"{WIKIPEDIA_API_URL}?{urlencode(params)}",
            headers={"User-Agent": USER_AGENT},
        )
        payload = request_json(request)
        for item in payload["query"]["random"]:
            title = item["title"]
            if title in exclude_titles:
                continue
            titles.add(title)
    return sorted(titles)


def request_json(request: Request, attempts: int = 6) -> dict[str, object]:
    delay = 1.0
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            with urlopen(request) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                raise
        except URLError as exc:
            last_error = exc
        time.sleep(delay)
        delay *= 2
    if last_error is not None:
        raise last_error
    raise RuntimeError("Wikipedia request failed without an error.")


def build_or_load_corpus(
    total_articles: int,
    paragraphs_per_article: int,
    cache_file: Path | None,
) -> dict[str, list[str]]:
    corpus: dict[str, list[str]] = {}
    if cache_file and cache_file.exists():
        corpus = json.loads(cache_file.read_text())
        if len(corpus) >= total_articles:
            return corpus

    target_titles = [title for titles in ARTICLE_GROUPS.values() for title in titles]
    if len(target_titles) > total_articles:
        raise ValueError("`total_articles` must be at least the number of benchmark target titles.")
    if not corpus:
        corpus = load_wikipedia_corpus(paragraphs_per_article)

    attempts = 0
    while len(corpus) < total_articles and attempts < 12:
        attempts += 1
        remaining = total_articles - len(corpus)
        random_titles = fetch_random_titles(min(max(remaining * 2, 50), 200), set(corpus))
        for start in range(0, len(random_titles), 20):
            if len(corpus) >= total_articles:
                break
            batch = random_titles[start : start + 20]
            try:
                extracts = fetch_article_extracts(batch)
            except Exception:
                continue
            for title, extract in extracts.items():
                if len(corpus) >= total_articles:
                    break
                if title in corpus:
                    continue
                paragraphs = extract_paragraphs(title, extract, paragraphs_per_article)
                if paragraphs:
                    corpus[title] = paragraphs
                    if cache_file:
                        cache_file.parent.mkdir(parents=True, exist_ok=True)
                        cache_file.write_text(json.dumps(corpus, indent=2) + "\n")

    if len(corpus) < total_articles:
        raise RuntimeError(
            f"Unable to build the requested corpus. Collected {len(corpus)} articles "
            f"out of {total_articles}."
        )

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(corpus, indent=2) + "\n")
    return corpus


def flatten_paragraphs(corpus: dict[str, list[str]], embedder: Embedder) -> list[dict[str, object]]:
    paragraphs = []
    texts = [paragraph for items in corpus.values() for paragraph in items]
    embeddings = embedder.embed_texts(texts)
    for paragraph, embedding in zip(texts, embeddings):
        title = parse_title(paragraph)
        paragraphs.append(
            {
                "title": title,
                "text": paragraph,
                "embedding": embedding,
                "preview": " ".join(paragraph.split())[:180],
                "excerpt": extract_excerpt(paragraph),
            }
        )
    return paragraphs


def evaluate_raw_case(
    retriever: RawCorpusRetriever,
    case: BenchmarkCase,
) -> dict[str, object]:
    start = time.perf_counter()
    result = retriever.retrieve(case.query)

    titles = set(result["title_counts"])
    matched_expected = [title for title in case.expected_titles if title in titles]
    matched_forbidden = [title for title in case.forbidden_titles if title in titles]
    all_text = "\n".join(result["texts"]).lower()
    matched_required = [
        term for term in case.required_terms if term.lower() in all_text
    ]
    top_cluster_recall = len(matched_expected) / len(case.expected_titles) if case.expected_titles else 1.0
    overall_recall = top_cluster_recall
    forbidden_clean_score = (
        1.0 - (len(matched_forbidden) / len(case.forbidden_titles))
        if case.forbidden_titles
        else 1.0
    )
    cluster_requirement_score = min(1.0 / case.min_clusters, 1.0)
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
    elapsed = time.perf_counter() - start
    return {
        "case_id": case.case_id,
        "query": case.query,
        "elapsed_ms": round(elapsed * 1000, 3),
        "matched_expected_titles_any": matched_expected,
        "matched_forbidden_titles_any": matched_forbidden,
        "matched_required_terms_any": matched_required,
        "required_term_recall": round(required_term_recall, 4),
        "aggregate_score": round(aggregate_score, 4),
        "draft_answer": result["draft_answer"],
        "title_counts": result["title_counts"],
        "top_hits": result["top_hits"],
    }


def evaluate_graph_case(memory: AgentMemory, case: BenchmarkCase) -> dict[str, object]:
    start = time.perf_counter()
    recall = memory.recall(case.query, max_clusters=5)
    clusters = recall.clusters
    top_cluster = clusters[0] if clusters else None
    top_titles = Counter(parse_title(hit.text) for hit in top_cluster.hits) if top_cluster else Counter()
    all_titles: set[str] = set()
    for cluster in clusters:
        all_titles.update(parse_title(hit.text) for hit in cluster.hits)
    matched_expected = [title for title in case.expected_titles if title in all_titles]
    matched_forbidden = [title for title in case.forbidden_titles if title in all_titles]
    all_text = "\n".join(hit.text.lower() for cluster in clusters for hit in cluster.hits)
    matched_required = [
        term for term in case.required_terms if term.lower() in all_text
    ]
    top_cluster_recall = len([title for title in case.expected_titles if title in top_titles]) / len(case.expected_titles) if case.expected_titles else 1.0
    overall_recall = len(matched_expected) / len(case.expected_titles) if case.expected_titles else 1.0
    forbidden_clean_score = (
        1.0 - (len(matched_forbidden) / len(case.forbidden_titles))
        if case.forbidden_titles
        else 1.0
    )
    cluster_requirement_score = min(len(clusters) / case.min_clusters, 1.0)
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
    draft_answer_parts = []
    for cluster in clusters[:2]:
        seen_titles: set[str] = set()
        excerpts = []
        for hit in cluster.hits:
            title = parse_title(hit.text)
            if title in seen_titles:
                continue
            seen_titles.add(title)
            excerpts.append(f"{title}: {extract_excerpt(hit.text)}")
            if len(excerpts) >= 2:
                break
        if excerpts:
            draft_answer_parts.append(" ".join(excerpts))
    elapsed = time.perf_counter() - start
    return {
        "case_id": case.case_id,
        "query": case.query,
        "elapsed_ms": round(elapsed * 1000, 3),
        "matched_expected_titles_any": matched_expected,
        "matched_forbidden_titles_any": matched_forbidden,
        "matched_required_terms_any": matched_required,
        "required_term_recall": round(required_term_recall, 4),
        "aggregate_score": round(aggregate_score, 4),
        "draft_answer": " ".join(draft_answer_parts),
        "cluster_count": len(clusters),
        "cluster_sizes": [len(cluster.memory_ids) for cluster in clusters],
    }


def compare_systems(
    total_articles: int,
    paragraphs_per_article: int,
    embedding_backend: str,
    cache_file: Path | None,
    keep_workspace: bool,
) -> dict[str, object]:
    corpus_started = time.perf_counter()
    config = build_benchmark_config(embedding_backend)
    embedder = build_embedder(config)
    corpus = build_or_load_corpus(total_articles, paragraphs_per_article, cache_file)
    corpus_elapsed = time.perf_counter() - corpus_started

    raw_started = time.perf_counter()
    paragraph_records = flatten_paragraphs(corpus, embedder)
    raw_retriever = RawCorpusRetriever(paragraph_records, embedder)
    raw_prepare_elapsed = time.perf_counter() - raw_started

    workspace = Path(tempfile.mkdtemp(prefix="agent-memory-compare-"))
    memory = AgentMemory.initialize(workspace, config=config, force=True, embedder=embedder)
    try:
        ingest_started = time.perf_counter()
        for paragraph in paragraph_records:
            memory.save(paragraph["text"], embedding=paragraph["embedding"])
        graph_ingest_elapsed = time.perf_counter() - ingest_started

        # Warm both paths once before timing.
        memory.recall(BENCHMARK_CASES[0].query, max_clusters=5)
        raw_retriever.retrieve(BENCHMARK_CASES[0].query)

        graph_results = [evaluate_graph_case(memory, case) for case in BENCHMARK_CASES]
        raw_results = [evaluate_raw_case(raw_retriever, case) for case in BENCHMARK_CASES]
        payload = {
            "embedding_backend": embedding_backend,
            "total_articles": len(corpus),
            "paragraphs_per_article": paragraphs_per_article,
            "corpus_load_ms": round(corpus_elapsed * 1000, 3),
            "raw_prepare_ms": round(raw_prepare_elapsed * 1000, 3),
            "graph_ingest_ms": round(graph_ingest_elapsed * 1000, 3),
            "graph_stats": memory.stats().to_dict(),
            "graph": {
                "average_elapsed_ms": round(
                    sum(item["elapsed_ms"] for item in graph_results) / len(graph_results), 3
                ),
                "average_score": round(
                    sum(item["aggregate_score"] for item in graph_results) / len(graph_results), 4
                ),
                "results": graph_results,
            },
            "raw": {
                "average_elapsed_ms": round(
                    sum(item["elapsed_ms"] for item in raw_results) / len(raw_results), 3
                ),
                "average_score": round(
                    sum(item["aggregate_score"] for item in raw_results) / len(raw_results), 4
                ),
                "results": raw_results,
            },
            "workspace": str(workspace),
        }
        return payload
    finally:
        memory.close()
        if not keep_workspace:
            import shutil

            shutil.rmtree(workspace, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare graph retrieval vs raw paragraph retrieval.")
    parser.add_argument("--total-articles", type=int, default=100)
    parser.add_argument("--paragraphs-per-article", type=int, default=0)
    parser.add_argument("--embedding-backend", choices=["fastembed", "hash"], default="fastembed")
    parser.add_argument("--cache-file", type=Path, default=None)
    parser.add_argument("--keep-workspace", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = compare_systems(
        total_articles=args.total_articles,
        paragraphs_per_article=args.paragraphs_per_article,
        embedding_backend=args.embedding_backend,
        cache_file=args.cache_file,
        keep_workspace=args.keep_workspace,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    print(
        f"Corpus: articles={payload['total_articles']} paragraphs/article={payload['paragraphs_per_article']}"
    )
    print(
        f"Graph avg: {payload['graph']['average_elapsed_ms']} ms, score={payload['graph']['average_score']}"
    )
    print(
        f"Raw avg:   {payload['raw']['average_elapsed_ms']} ms, score={payload['raw']['average_score']}"
    )
    print("\nPer-case deltas")
    raw_by_id = {item["case_id"]: item for item in payload["raw"]["results"]}
    for graph_case in payload["graph"]["results"]:
        raw_case = raw_by_id[graph_case["case_id"]]
        print(
            f"{graph_case['case_id']}: "
            f"graph={graph_case['elapsed_ms']}ms/{graph_case['aggregate_score']} "
            f"raw={raw_case['elapsed_ms']}ms/{raw_case['aggregate_score']}"
        )
    if args.keep_workspace:
        print(f"\nWorkspace: {payload['workspace']}")


if __name__ == "__main__":
    main()
