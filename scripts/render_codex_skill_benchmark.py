from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Codex skill benchmark JSON file to HTML."
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    return parser.parse_args()


def format_ms(value: object) -> str:
    return f"{float(value):,.1f} ms"


def render_reference_list(items: list[str], empty: str) -> str:
    if not items:
        return f"<p class='empty'>{escape(empty)}</p>"
    return "<ul>" + "".join(f"<li><code>{escape(item)}</code></li>" for item in items) + "</ul>"


def render_trace(trace: dict[str, object] | None) -> str:
    if not trace:
        return "<p class='empty'>No trace captured.</p>"

    responses = list(trace.get("responses") or [])
    tool_calls = list(trace.get("tool_calls") or [])
    summary = f"""
    <div class="trace-summary">
      <div><strong>Model turns</strong><span>{escape(str(trace.get('model_turns', 0)))}</span></div>
      <div><strong>Tool calls</strong><span>{escape(str(trace.get('tool_call_count', 0)))}</span></div>
      <div><strong>Input tokens</strong><span>{escape(str(trace.get('total_input_tokens', 0)))}</span></div>
      <div><strong>Output tokens</strong><span>{escape(str(trace.get('total_output_tokens', 0)))}</span></div>
      <div><strong>Cached tokens</strong><span>{escape(str(trace.get('total_cached_tokens', 0)))}</span></div>
    </div>
    """

    tool_blocks = []
    for index, call in enumerate(tool_calls, start=1):
        tool_blocks.append(
            f"""
            <div class="trace-block">
              <div class="trace-label">Tool {index}</div>
              <p><strong>Name:</strong> <code>{escape(str(call.get('name') or 'shell'))}</code></p>
              <p><strong>Arguments:</strong></p>
              <pre>{escape(str(call.get('arguments') or ''))}</pre>
              <p><strong>Output preview:</strong></p>
              <pre>{escape(str(call.get('output_preview') or ''))}</pre>
            </div>
            """
        )

    response_blocks = []
    for response in responses:
        response_blocks.append(
            f"""
            <div class="trace-block">
              <div class="trace-label">Response {escape(str(response.get('index', 0)))}</div>
              <pre>{escape(json.dumps(response, indent=2))}</pre>
            </div>
            """
        )

    return (
        summary
        + "<details><summary>Tool Calls</summary>"
        + ("".join(tool_blocks) if tool_blocks else "<p class='empty'>No tool calls</p>")
        + "</details>"
        + "<details><summary>Model Responses</summary>"
        + ("".join(response_blocks) if response_blocks else "<p class='empty'>No response metadata</p>")
        + "</details>"
    )


def render_case(case: dict[str, object]) -> str:
    graph = case["graph"]
    raw = case["raw"]
    graph_ms = float(graph.get("elapsed_ms") or 0.0)
    raw_ms = float(raw.get("elapsed_ms") or 0.0)
    if abs(graph_ms - raw_ms) < 0.001:
        graph_class = raw_class = "neutral"
        faster = "Tie"
    elif graph_ms < raw_ms:
        graph_class, raw_class, faster = "winner", "loser", "Graph faster"
    else:
        graph_class, raw_class, faster = "loser", "winner", "Raw faster"
    return f"""
    <article class="case-card">
      <div class="case-meta">
        <span>{escape(str(case['case_id']))}</span>
        <span>{escape(faster)}</span>
      </div>
      <h2>{escape(str(case['query']))}</h2>
      <div class="columns">
        <section class="panel {graph_class}">
          <div class="panel-head"><h3>Graph Skill</h3><span>{escape(format_ms(graph_ms))}</span></div>
          <h4>Answer</h4>
          <pre>{escape(str(graph.get('answer', '')))}</pre>
          <h4>References</h4>
          {render_reference_list(list(graph.get('references') or []), "No references")}
          <h4>Checked Memory IDs</h4>
          {render_reference_list(list(graph.get('checked_memory_ids') or []), "No checked memory IDs")}
          <h4>Trace</h4>
          {render_trace(graph.get('trace'))}
          <details>
            <summary>Raw completion</summary>
            <pre>{escape(str(graph.get('raw_completion', '')))}</pre>
          </details>
        </section>
        <section class="panel {raw_class}">
          <div class="panel-head"><h3>Raw Skill</h3><span>{escape(format_ms(raw_ms))}</span></div>
          <h4>Answer</h4>
          <pre>{escape(str(raw.get('answer', '')))}</pre>
          <h4>References</h4>
          {render_reference_list(list(raw.get('references') or []), "No references")}
          <h4>Inspected Files</h4>
          {render_reference_list(list(raw.get('inspected_files') or []), "No inspected files")}
          <h4>Trace</h4>
          {render_trace(raw.get('trace'))}
          <details>
            <summary>Raw completion</summary>
            <pre>{escape(str(raw.get('raw_completion', '')))}</pre>
          </details>
        </section>
      </div>
    </article>
    """


def render_html(payload: dict[str, object]) -> str:
    cases = payload["cases"]
    graph_avg = sum(float(case["graph"].get("elapsed_ms") or 0.0) for case in cases) / len(cases)
    raw_avg = sum(float(case["raw"].get("elapsed_ms") or 0.0) for case in cases) / len(cases)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Codex Skill Benchmark</title>
  <style>
    :root {{
      --bg: #f5efe3;
      --ink: #1f1a16;
      --muted: #6e645b;
      --line: rgba(40, 28, 16, 0.14);
      --card: rgba(255,255,255,0.82);
      --winner: #e1f3ea;
      --loser: #fae7dc;
      --neutral: #f0ebe2;
      --shadow: 0 18px 44px rgba(73, 52, 27, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(197, 161, 114, 0.24), transparent 34%),
        radial-gradient(circle at top right, rgba(79, 124, 111, 0.18), transparent 30%),
        linear-gradient(180deg, #f8f3ea, #efe6d8);
    }}
    main {{
      width: min(1360px, calc(100vw - 40px));
      margin: 0 auto;
      padding: 34px 0 72px;
    }}
    header, .case-card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 26px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
    }}
    header {{
      padding: 30px;
      margin-bottom: 24px;
    }}
    h1, h2, h3, h4 {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
    }}
    h1 {{ font-size: clamp(2.1rem, 4vw, 3.4rem); margin-bottom: 10px; }}
    .lede {{ color: var(--muted); max-width: 75ch; line-height: 1.6; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-top: 22px;
    }}
    .summary div {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px 16px;
      background: rgba(255,255,255,0.6);
    }}
    .summary strong {{ display: block; color: var(--muted); font-size: 0.84rem; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 6px; }}
    .summary span {{ font-size: 1.35rem; font-weight: 700; }}
    .case-card {{
      padding: 22px;
      margin-bottom: 18px;
    }}
    .case-meta {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      color: var(--muted);
      margin-bottom: 14px;
    }}
    .case-meta span {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      background: rgba(255,255,255,0.58);
      font-size: 0.9rem;
    }}
    .columns {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-top: 18px;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
      background: rgba(255,255,255,0.7);
    }}
    .panel.winner {{ background: linear-gradient(180deg, #fbfffd 0%, var(--winner) 100%); }}
    .panel.loser {{ background: linear-gradient(180deg, #fffaf7 0%, var(--loser) 100%); }}
    .panel.neutral {{ background: linear-gradient(180deg, #ffffff 0%, var(--neutral) 100%); }}
    .panel-head {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 14px;
    }}
    h4 {{ margin: 14px 0 8px; font-size: 1rem; }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 0.9rem;
      line-height: 1.55;
      background: rgba(255,255,255,0.78);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
    }}
    ul {{
      margin: 0;
      padding-left: 20px;
    }}
    li {{ margin-bottom: 8px; }}
    .trace-summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
      margin-top: 6px;
      margin-bottom: 12px;
    }}
    .trace-summary div {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.62);
    }}
    .trace-summary strong {{
      display: block;
      color: var(--muted);
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 4px;
    }}
    .trace-summary span {{
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 0.92rem;
    }}
    .trace-block {{
      margin-top: 12px;
    }}
    .trace-label {{
      font-weight: 700;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    code {{
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 0.92em;
      background: rgba(31, 26, 22, 0.06);
      border-radius: 8px;
      padding: 2px 6px;
    }}
    details {{ margin-top: 12px; }}
    .empty {{ color: var(--muted); margin: 0; }}
    @media (max-width: 940px) {{
      main {{ width: min(100vw - 20px, 1360px); }}
      .columns {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Codex Skill Benchmark</h1>
      <p class="lede">This report compares two fresh Codex skill runs per question. The graph skill may only use the local Agent Memory graph. The raw skill may only use the folder of raw article files and must decide which files to inspect itself.</p>
      <section class="summary">
        <div><strong>Model</strong><span>{escape(str(payload.get('model', '')))}</span></div>
        <div><strong>Cases</strong><span>{len(cases)}</span></div>
        <div><strong>Graph Avg</strong><span>{escape(format_ms(graph_avg))}</span></div>
        <div><strong>Raw Avg</strong><span>{escape(format_ms(raw_avg))}</span></div>
      </section>
    </header>
    {''.join(render_case(case) for case in cases)}
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input_file.read_text())
    args.output_file.write_text(render_html(payload))


if __name__ == "__main__":
    main()
