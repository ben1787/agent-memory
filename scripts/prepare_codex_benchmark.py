from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from agent_memory.benchmark import BENCHMARK_CASES, build_benchmark_config
from agent_memory.end_to_end import flatten_paragraphs
from agent_memory.embeddings import build_embedder
from agent_memory.engine import AgentMemory
from agent_memory.raw_corpus import load_raw_article_corpus, write_raw_article_corpus


DEFAULT_RAW_ARTICLES_DIR = Path(__file__).resolve().parents[1] / "benchmark_corpus" / "raw_articles"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a local benchmark workspace for Codex subagent evaluation."
    )
    parser.add_argument(
        "--raw-articles-dir",
        type=Path,
        default=DEFAULT_RAW_ARTICLES_DIR,
        help="Canonical raw article folder. Defaults to benchmark_corpus/raw_articles inside the repo.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--total-articles", type=int, default=100)
    parser.add_argument(
        "--paragraphs-per-article",
        type=int,
        default=0,
        help="Maximum number of paragraphs per article. Use 0 for all available paragraphs.",
    )
    parser.add_argument("--embedding-backend", choices=["fastembed", "hash"], default="fastembed")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.force:
            raise SystemExit(
                f"{output_root} already exists. Re-run with --force to overwrite it."
            )
        shutil.rmtree(output_root)

    raw_articles_dir = output_root / "raw_articles"
    graph_root = output_root / "graph_project"
    output_root.mkdir(parents=True, exist_ok=True)
    raw_articles_dir.mkdir(parents=True, exist_ok=True)

    source_raw_articles_dir = args.raw_articles_dir.resolve()
    if not source_raw_articles_dir.exists():
        raise SystemExit(
            "No raw article corpus found. Expected a folder at "
            f"{source_raw_articles_dir}. Rebuild it with "
            "`uv run python scripts/build_benchmark_corpus.py --force`."
        )
    corpus = load_raw_article_corpus(source_raw_articles_dir)

    if not corpus:
        raise SystemExit(f"No article data found in {source_raw_articles_dir}.")

    required_titles: list[str] = []
    seen_titles: set[str] = set()
    for case in BENCHMARK_CASES:
        for title in case.expected_titles:
            if title in seen_titles or title not in corpus:
                continue
            seen_titles.add(title)
            required_titles.append(title)
    for title in corpus:
        if len(required_titles) >= args.total_articles:
            break
        if title in seen_titles:
            continue
        seen_titles.add(title)
        required_titles.append(title)

    if args.paragraphs_per_article > 0:
        limited_corpus = {
            title: corpus[title][:args.paragraphs_per_article] for title in required_titles
        }
    else:
        limited_corpus = {title: corpus[title] for title in required_titles}

    article_manifest = write_raw_article_corpus(limited_corpus, raw_articles_dir)

    config = build_benchmark_config(args.embedding_backend)
    embedder = build_embedder(config)
    paragraph_records = flatten_paragraphs(limited_corpus, embedder)
    memory = AgentMemory.initialize(graph_root, config=config, force=True, embedder=embedder)
    try:
        graph_stats = memory.import_memories(paragraph_records).to_dict()
    finally:
        memory.close()

    cases_payload = [
        {
            "case_id": case.case_id,
            "query": case.query,
            "kind": case.kind,
            "expected_titles": case.expected_titles,
            "forbidden_titles": case.forbidden_titles,
            "required_terms": case.required_terms,
        }
        for case in BENCHMARK_CASES
    ]
    (output_root / "benchmark_cases.json").write_text(json.dumps(cases_payload, indent=2) + "\n")

    manifest = {
        "source_raw_articles_dir": str(source_raw_articles_dir),
        "output_root": str(output_root),
        "raw_articles_dir": str(raw_articles_dir),
        "graph_project_root": str(graph_root),
        "total_articles": args.total_articles,
        "paragraphs_per_article": args.paragraphs_per_article,
        "embedding_backend": args.embedding_backend,
        "graph_stats": graph_stats,
        "article_manifest": article_manifest,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Prepared benchmark workspace at {output_root}")
    print(f"Raw articles: {raw_articles_dir}")
    print(f"Graph project: {graph_root}")
    print(f"Cases: {output_root / 'benchmark_cases.json'}")


if __name__ == "__main__":
    main()
