from __future__ import annotations

import re
from pathlib import Path


SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
PARAGRAPH_LABEL_PATTERN = re.compile(r"^\[(.+?) ¶(\d+)\]$")


def safe_slug(title: str) -> str:
    slug = SAFE_FILENAME_PATTERN.sub("-", title.strip()).strip("-._")
    return slug or "article"


def render_article_file(title: str, paragraphs: list[str]) -> str:
    lines = [f"# {title}", ""]
    for index, paragraph in enumerate(paragraphs, start=1):
        body = "\n".join(paragraph.splitlines()[1:]).strip()
        lines.append(f"[{title} ¶{index}]")
        lines.append(body)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_index_file(article_manifest: list[dict[str, object]]) -> str:
    lines = [
        "# Raw Article Index",
        "",
        "Use this index to decide which article files to open.",
        "Each entry lists the article title, filename, and paragraph count.",
        "",
    ]
    for item in article_manifest:
        lines.append(
            f"- {item['title']} | file={Path(str(item['file'])).name} | paragraphs={item['paragraph_count']}"
        )
    lines.append("")
    return "\n".join(lines)


def write_raw_article_corpus(corpus: dict[str, list[str]], output_dir: Path) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    article_manifest: list[dict[str, object]] = []
    for title, paragraphs in corpus.items():
        filename = f"{safe_slug(title)}.md"
        article_path = output_dir / filename
        article_path.write_text(render_article_file(title, paragraphs), encoding='utf-8')
        article_manifest.append(
            {
                "title": title,
                "file": str(article_path),
                "paragraph_count": len(paragraphs),
                "references": [f"{title} ¶{index}" for index in range(1, len(paragraphs) + 1)],
            }
        )
    (output_dir / "INDEX.md").write_text(render_index_file(article_manifest), encoding='utf-8')
    return article_manifest


def load_raw_article_corpus(raw_articles_dir: Path) -> dict[str, list[str]]:
    corpus: dict[str, list[str]] = {}
    for article_path in sorted(raw_articles_dir.glob("*.md")):
        if article_path.name == "INDEX.md":
            continue
        title, paragraphs = parse_article_file(article_path)
        if paragraphs:
            corpus[title] = paragraphs
    return corpus


def parse_article_file(article_path: Path) -> tuple[str, list[str]]:
    text = article_path.read_text(encoding='utf-8')
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError(f"Unexpected article file format: {article_path}")
    title = lines[0].removeprefix("# ").strip()
    paragraphs: list[str] = []
    current: list[str] = []
    current_label: str | None = None

    for line in lines[1:]:
        match = PARAGRAPH_LABEL_PATTERN.match(line.strip())
        if match:
            if current_label is not None:
                body = "\n".join(current).strip()
                paragraphs.append(f"Source: {title}\n{body}")
            current_label = match.group(1)
            current = []
            continue
        if current_label is None:
            continue
        current.append(line)

    if current_label is not None:
        body = "\n".join(current).strip()
        paragraphs.append(f"Source: {title}\n{body}")

    return title, paragraphs
