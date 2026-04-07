from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from compare_graph_vs_raw import build_or_load_corpus

from agent_memory.raw_corpus import write_raw_article_corpus


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "benchmark_corpus" / "raw_articles"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the canonical local benchmark corpus as raw article files."
    )
    parser.add_argument("--total-articles", type=int, default=100)
    parser.add_argument(
        "--paragraphs-per-article",
        type=int,
        default=0,
        help="Maximum paragraphs per article. Use 0 for full articles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where the canonical raw article corpus should be written.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.force:
            raise SystemExit(
                f"{output_dir} already exists. Re-run with --force to overwrite it."
            )
        shutil.rmtree(output_dir)

    corpus = build_or_load_corpus(
        total_articles=args.total_articles,
        paragraphs_per_article=args.paragraphs_per_article,
        cache_file=None,
    )
    article_manifest = write_raw_article_corpus(corpus, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "total_articles": len(article_manifest),
                "paragraphs_per_article": args.paragraphs_per_article,
                "sample_titles": [item["title"] for item in article_manifest[:10]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
