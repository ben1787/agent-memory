from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_graph_vs_raw import build_or_load_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or refresh a cached Wikipedia corpus for benchmarking."
    )
    parser.add_argument("--total-articles", type=int, default=100)
    parser.add_argument(
        "--paragraphs-per-article",
        type=int,
        default=0,
        help="Maximum paragraphs per article. Use 0 for all available paragraphs.",
    )
    parser.add_argument("--cache-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = build_or_load_corpus(
        total_articles=args.total_articles,
        paragraphs_per_article=args.paragraphs_per_article,
        cache_file=args.cache_file,
    )
    args.cache_file.parent.mkdir(parents=True, exist_ok=True)
    args.cache_file.write_text(json.dumps(corpus, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cache_file": str(args.cache_file.resolve()),
                "total_articles": len(corpus),
                "paragraphs_per_article": args.paragraphs_per_article,
                "sample_titles": list(corpus)[:10],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
