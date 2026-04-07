# Benchmark Corpus

This directory is the canonical local benchmark corpus for `agent-memory`.

- Raw Wikipedia article files live in `benchmark_corpus/raw_articles/`
- Each article is stored as a standalone Markdown file
- `INDEX.md` inside that folder lists the available articles and paragraph counts

The raw article files are intentionally gitignored. They are local benchmark fixtures, not product code.

To rebuild the corpus locally:

```bash
uv run python scripts/build_benchmark_corpus.py --force
```
