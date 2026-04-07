---
name: raw-file-benchmark-answer
description: Use when a benchmark subagent must answer a question using only a folder of raw article files. The agent must decide which files to inspect with content search, read only the most relevant files, and cite the article paragraph labels actually used.
---

# Raw File Benchmark Answer

Use this skill only for the raw-document side of the benchmark.

## Inputs

Expect:
- a benchmark question
- a directory of raw article files

Each article file is separate. Paragraphs are labeled like:

```text
[Graph theory ¶1]
```

## Retrieval workflow

1. Start with content search to decide which files to inspect:

```bash
rg -n "<keyword1>|<keyword2>|<keyword3>" "<RAW_ARTICLES_DIR>"
```

2. Use the search results to shortlist a small set of candidate article files.
3. Open only the files that look relevant.
4. Read the labeled paragraphs you need.
5. Answer only from those files.

## Output rules

- Do not inspect the graph project or use `agent-memory recall`.
- Do not rely on outside knowledge unless the files are clearly insufficient.
- Return JSON only, with this shape:

```json
{
  "answer": "short answer text",
  "references": ["Graph theory ¶1", "Graph database ¶1"],
  "inspected_files": ["Graph-theory.md", "Graph-database.md"]
}
```

- `references` must contain only the paragraph labels you actually used.
- `inspected_files` should list only the files you actually opened.
- If the files are insufficient, say so briefly in `answer` and still include the paragraphs you checked.

## Notes

- The benchmark is testing whether you can decide which raw files to inspect, so do not read the whole corpus by default.
- Prefer the smallest set of files that supports the answer.
