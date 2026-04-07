from __future__ import annotations

from html import escape


TRACK_LABELS = {
    "technical": "Technical Track",
    "science": "Science Track",
}


def _format_ms(value: object) -> str:
    return f"{float(value):,.3f} ms"


def _winner_class(left: float, right: float, current: float) -> str:
    if abs(left - right) < 0.001:
        return "neutral"
    if current <= min(left, right):
        return "winner"
    return "loser"


def _render_reference_list(result: dict[str, object]) -> str:
    selected = list(result.get("display_references", []))
    contexts = {
        item["reference_id"]: item
        for item in result.get("context_references", [])
    }
    if not selected:
        return "<p class='empty'>No cited references were extracted.</p>"

    parts = ["<ul class='references'>"]
    for reference_id in selected:
        item = contexts.get(reference_id)
        if item is None:
            parts.append(f"<li><code>{escape(reference_id)}</code></li>")
            continue
        locator = f" <span class='locator'>{escape(item['locator'])}</span>" if item.get("locator") else ""
        parts.append(
            "<li>"
            f"<code>{escape(item['reference_id'])}</code> "
            f"<strong>{escape(item['title'])}</strong>{locator}"
            f"<div class='excerpt'>{escape(item['excerpt'])}</div>"
            "</li>"
        )
    parts.append("</ul>")
    return "".join(parts)


def _render_context_details(result: dict[str, object]) -> str:
    contexts = result.get("context_references", [])
    if not contexts:
        return "<p class='empty'>No retrieved context captured.</p>"

    parts = ["<details><summary>Retrieved context</summary><ul class='contexts'>"]
    for item in contexts:
        locator = f" <span class='locator'>{escape(item['locator'])}</span>" if item.get("locator") else ""
        parts.append(
            "<li>"
            f"<code>{escape(item['reference_id'])}</code> "
            f"<strong>{escape(item['title'])}</strong>{locator}"
            f"<span class='score'>score {float(item['score']):.4f}</span>"
            f"<div class='excerpt'>{escape(item['excerpt'])}</div>"
            "</li>"
        )
    parts.append("</ul></details>")
    return "".join(parts)


def _render_system_panel(
    label: str,
    result: dict[str, object],
    graph_total_ms: float,
    raw_total_ms: float,
) -> str:
    total_ms = float(result["total_ms"])
    winner_class = _winner_class(graph_total_ms, raw_total_ms, total_ms)
    chip = "Faster" if winner_class == "winner" else "Slower" if winner_class == "loser" else "Tie"
    cited = ", ".join(f"[{ref}]" for ref in result.get("cited_references", []))
    inferred = ", ".join(f"[{ref}]" for ref in result.get("inferred_references", []))
    return (
        f"<section class='system-card {winner_class}'>"
        f"<div class='system-head'><h4>{escape(label)}</h4><span class='chip {winner_class}'>{chip}</span></div>"
        "<dl class='metrics'>"
        f"<div><dt>Total</dt><dd>{escape(_format_ms(result['total_ms']))}</dd></div>"
        f"<div><dt>Retrieval</dt><dd>{escape(_format_ms(result['retrieval_ms']))}</dd></div>"
        f"<div><dt>Generation</dt><dd>{escape(_format_ms(result['generation_ms']))}</dd></div>"
        f"<div><dt>Context Tokens</dt><dd>{int(result['context_tokens'])}</dd></div>"
        f"<div><dt>Context Score</dt><dd>{float(result['context_score']):.4f}</dd></div>"
        "</dl>"
        "<div class='answer-block'>"
        "<h5>Answer</h5>"
        f"<pre>{escape(str(result['answer']))}</pre>"
        "</div>"
        "<div class='answer-block'>"
        "<h5>References Used</h5>"
        f"<p class='meta'><strong>Cited:</strong> {escape(cited) if cited else 'none'}</p>"
        f"<p class='meta'><strong>Inferred:</strong> {escape(inferred) if inferred else 'none'}</p>"
        f"{_render_reference_list(result)}"
        "</div>"
        f"{_render_context_details(result)}"
        "</section>"
    )


def _render_case(graph_result: dict[str, object], raw_result: dict[str, object]) -> str:
    case_id = str(graph_result["case_id"])
    _, level = case_id.rsplit("-level-", 1)
    track = case_id.split("-")[0]
    graph_total_ms = float(graph_result["total_ms"])
    raw_total_ms = float(raw_result["total_ms"])
    faster = "Graph" if graph_total_ms < raw_total_ms else "Raw" if raw_total_ms < graph_total_ms else "Tie"
    return (
        "<article class='question-card'>"
        f"<div class='question-meta'><span class='track'>{escape(TRACK_LABELS.get(track, track.title()))}</span>"
        f"<span class='level'>Level {escape(level)}</span>"
        f"<span class='winner-banner'>{escape(faster)} faster</span></div>"
        f"<h3>{escape(str(graph_result['query']))}</h3>"
        "<div class='panels'>"
        f"{_render_system_panel('Graph Agent', graph_result, graph_total_ms, raw_total_ms)}"
        f"{_render_system_panel('Raw Agent', raw_result, graph_total_ms, raw_total_ms)}"
        "</div>"
        "</article>"
    )


def render_isolated_benchmark_report(payload: dict[str, object]) -> str:
    graph_results = {item["case_id"]: item for item in payload["graph"]["results"]}
    raw_results = {item["case_id"]: item for item in payload["raw"]["results"]}

    technical_cases = []
    science_cases = []
    for case_id in graph_results:
        rendered = _render_case(graph_results[case_id], raw_results[case_id])
        if case_id.startswith("technical-"):
            technical_cases.append(rendered)
        else:
            science_cases.append(rendered)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent Memory Benchmark Report</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: rgba(255,255,255,0.82);
      --panel-strong: #fffdf8;
      --ink: #1e1b18;
      --muted: #6c6258;
      --line: rgba(48, 35, 23, 0.12);
      --graph: #1d6b57;
      --raw: #8c4f1f;
      --winner: #dff3ea;
      --loser: #f8e5d6;
      --neutral: #ece7de;
      --shadow: 0 18px 48px rgba(74, 53, 30, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(198, 154, 109, 0.24), transparent 36%),
        radial-gradient(circle at top right, rgba(88, 150, 131, 0.18), transparent 28%),
        linear-gradient(180deg, #f9f4eb 0%, #f1eadf 100%);
    }}
    main {{
      width: min(1400px, calc(100vw - 48px));
      margin: 0 auto;
      padding: 40px 0 80px;
    }}
    header {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 32px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    h1, h2, h3, h4, h5 {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      font-weight: 700;
      letter-spacing: -0.02em;
    }}
    h1 {{ font-size: clamp(2.2rem, 4vw, 3.7rem); margin-bottom: 12px; }}
    .lede {{ color: var(--muted); font-size: 1.04rem; max-width: 70ch; line-height: 1.6; }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-top: 24px;
    }}
    .summary-card {{
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px 20px;
    }}
    .summary-card dt {{ color: var(--muted); font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.08em; }}
    .summary-card dd {{ margin: 8px 0 0; font-size: 1.5rem; font-weight: 700; }}
    .track-section {{ margin-top: 34px; }}
    .track-header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      margin-bottom: 16px;
      gap: 16px;
    }}
    .track-header p {{ margin: 0; color: var(--muted); }}
    .question-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 24px;
      box-shadow: var(--shadow);
      margin-bottom: 18px;
    }}
    .question-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 14px;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .question-meta span {{
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.66);
      border-radius: 999px;
      padding: 6px 10px;
    }}
    .winner-banner {{
      color: var(--ink);
      font-weight: 700;
    }}
    .question-card h3 {{
      font-size: 1.45rem;
      line-height: 1.25;
      margin-bottom: 20px;
    }}
    .panels {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .system-card {{
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 20px;
      background: var(--panel-strong);
    }}
    .system-card.winner {{ background: linear-gradient(180deg, #f7fffb 0%, var(--winner) 100%); }}
    .system-card.loser {{ background: linear-gradient(180deg, #fff9f4 0%, var(--loser) 100%); }}
    .system-card.neutral {{ background: linear-gradient(180deg, #ffffff 0%, var(--neutral) 100%); }}
    .system-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }}
    .chip {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 0.82rem;
      font-weight: 700;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.72);
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin: 0 0 16px;
    }}
    .metrics div {{
      padding: 10px 12px;
      border-radius: 16px;
      background: rgba(255,255,255,0.55);
      border: 1px solid var(--line);
    }}
    .metrics dt {{
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 6px;
    }}
    .metrics dd {{
      margin: 0;
      font-size: 1rem;
      font-weight: 700;
    }}
    .answer-block {{
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px solid var(--line);
    }}
    .answer-block h5 {{ font-size: 1rem; margin-bottom: 10px; }}
    pre {{
      white-space: pre-wrap;
      margin: 0;
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 0.92rem;
      line-height: 1.55;
      background: rgba(255,255,255,0.78);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
    }}
    .meta {{ margin: 0 0 8px; color: var(--muted); }}
    ul.references, ul.contexts {{
      margin: 10px 0 0;
      padding-left: 18px;
    }}
    ul.references li, ul.contexts li {{
      margin-bottom: 12px;
      line-height: 1.45;
    }}
    code {{
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 0.9em;
      background: rgba(30,27,24,0.06);
      border-radius: 8px;
      padding: 2px 6px;
    }}
    .locator, .score {{
      color: var(--muted);
      margin-left: 8px;
      font-size: 0.88rem;
    }}
    .excerpt {{
      margin-top: 6px;
      color: var(--ink);
    }}
    details {{
      margin-top: 14px;
      border-top: 1px solid var(--line);
      padding-top: 14px;
    }}
    details summary {{
      cursor: pointer;
      font-weight: 700;
    }}
    .empty {{ color: var(--muted); margin: 0; }}
    @media (max-width: 980px) {{
      main {{ width: min(100vw - 24px, 1400px); padding-top: 24px; }}
      .panels {{ grid-template-columns: 1fr; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Agent Memory Isolated Benchmark</h1>
      <p class="lede">This report compares two isolated subprocess agents on the same 20-question ladder. The graph agent only sees the memory graph. The raw agent only sees raw paragraph files. Each answer includes the answer text, the references it cited or most likely used, and the retrieved context that was available to that agent.</p>
      <section class="summary-grid">
        <dl class="summary-card">
          <dt>Model</dt>
          <dd>{escape(str(payload['model_id']))}</dd>
        </dl>
        <dl class="summary-card">
          <dt>Corpus</dt>
          <dd>{int(payload['total_articles'])} articles</dd>
        </dl>
        <dl class="summary-card">
          <dt>Graph Avg</dt>
          <dd>{escape(_format_ms(payload['graph']['summary']['average_total_ms']))}</dd>
        </dl>
        <dl class="summary-card">
          <dt>Raw Avg</dt>
          <dd>{escape(_format_ms(payload['raw']['summary']['average_total_ms']))}</dd>
        </dl>
        <dl class="summary-card">
          <dt>Graph Context</dt>
          <dd>{payload['graph']['summary']['average_context_tokens']} tok</dd>
        </dl>
        <dl class="summary-card">
          <dt>Raw Context</dt>
          <dd>{payload['raw']['summary']['average_context_tokens']} tok</dd>
        </dl>
      </section>
    </header>

    <section class="track-section">
      <div class="track-header">
        <h2>Technical Track</h2>
        <p>Levels 1-10 scale from single-article lookup to multi-article architectural synthesis.</p>
      </div>
      {"".join(technical_cases)}
    </section>

    <section class="track-section">
      <div class="track-header">
        <h2>Science Track</h2>
        <p>Levels 1-10 scale from direct astronomy/biology lookup to broad cross-domain synthesis.</p>
      </div>
      {"".join(science_cases)}
    </section>
  </main>
</body>
</html>
"""
