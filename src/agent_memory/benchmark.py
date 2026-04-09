from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import shutil
import tempfile
import time
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agent_memory.config import MemoryConfig
from agent_memory.embeddings import Embedder, build_embedder
from agent_memory.engine import AgentMemory


WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "agent-memory-eval/0.2 (benchmark harness)"
TITLE_PREFIX = "Source"
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


ARTICLE_GROUPS = {
    "neuroscience": [
        "Hippocampus",
        "Neuron",
        "Synapse",
        "Engram (neuropsychology)",
    ],
    "graphs": [
        "Graph database",
        "Graph theory",
        "Knowledge graph",
        "Vector database",
    ],
    "llms": [
        "Transformer (deep learning)",
        "Attention (machine learning)",
        "Large language model",
        "Retrieval-augmented generation",
    ],
    "astronomy": [
        "Black hole",
        "Exoplanet",
        "Dark matter",
        "Galaxy",
    ],
    "biology": [
        "DNA",
        "Gene expression",
        "Protein folding",
        "CRISPR",
    ],
}


@dataclass(slots=True)
class BenchmarkCase:
    case_id: str
    query: str
    kind: str
    expected_titles: list[str]
    forbidden_titles: list[str] = field(default_factory=list)
    required_terms: list[str] = field(default_factory=list)
    min_clusters: int = 1


TECHNICAL_TRACK = [
    "Graph theory",
    "Graph database",
    "Knowledge graph",
    "Vector database",
    "Retrieval-augmented generation",
    "Large language model",
    "Transformer (deep learning)",
    "Attention (machine learning)",
    "Hippocampus",
    "Engram (neuropsychology)",
]

SCIENCE_TRACK = [
    "Exoplanet",
    "Black hole",
    "Galaxy",
    "Dark matter",
    "DNA",
    "Gene expression",
    "Protein folding",
    "CRISPR",
    "Neuron",
    "Synapse",
]

TECHNICAL_CASE_SPECS = [
    {
        "query": "How did the Seven Bridges of Konigsberg and Euler's characteristic push graph theory toward abstract connectivity rather than geometric layout?",
        "required_terms": ["Seven Bridges of Konigsberg", "Euler", "connectivity"],
    },
    {
        "query": "How does a graph database operationalize graph theory's abstract vertices and edges, and why does index-free adjacency matter for traversal-heavy queries compared with relational joins?",
        "required_terms": ["index-free adjacency", "traversal", "joins"],
    },
    {
        "query": "If a graph database already stores nodes and edges, what extra work does a knowledge graph's schema or ontology layer do for semantics, inference, and entity meaning?",
        "required_terms": ["schema", "ontology", "inference"],
    },
    {
        "query": "Why are vector databases built around embeddings and approximate nearest-neighbor methods such as HNSW not interchangeable with graph databases or knowledge graphs?",
        "required_terms": ["embeddings", "approximate nearest neighbor", "HNSW"],
    },
    {
        "query": "In a retrieval stack, how should vector search, graph structure, and knowledge-graph semantics complement one another in RAG, and why doesn't retrieval by itself eliminate hallucination or source-conflict problems?",
        "required_terms": ["hallucination", "conflicting information", "vector search"],
    },
    {
        "query": "Why can RAG reduce the need to retrain large language models for fresh domain knowledge while still leaving unresolved issues such as hallucination, fine-tuning behavior, and alignment?",
        "required_terms": ["retrain", "fine-tuning", "alignment", "hallucination"],
    },
    {
        "query": "Why did transformer architectures overtake recurrent seq2seq models for large language models, and what role did self-attention and parallelization play in that shift?",
        "required_terms": ["recurrent", "self-attention", "parallelization"],
    },
    {
        "query": "How does attention evolve from source-target alignment into causally masked self-attention in transformers, and why is that change crucial for autoregressive generation?",
        "required_terms": ["alignment", "causal masking", "self-attention", "autoregressive"],
    },
    {
        "query": "If you compare this retrieval stack to the hippocampus, which pieces resemble spatial indexing, replay during sharp waves and ripples, or consolidation, and where does the analogy break?",
        "required_terms": ["spatial memory", "sharp waves and ripples", "consolidation"],
    },
    {
        "query": "Extend the analogy with engrams: how would a brain-inspired memory design combine explicit graphs, vector retrieval, hippocampal replay, and distributed engram reactivation without pretending one memory lives in one node?",
        "required_terms": ["engram", "reactivation", "optogenetic", "distributed"],
    },
]

SCIENCE_CASE_SPECS = [
    {
        "query": "How does the exoplanet article distinguish planets from brown dwarfs using the deuterium-fusion threshold, and why do rogue planets complicate a simple orbit-based definition?",
        "required_terms": ["deuterium", "brown dwarf", "rogue planets"],
    },
    {
        "query": "Compare the indirect evidence used for exoplanets and black holes: why do exoplanets rely on transits and Doppler shifts while black holes are inferred from accretion, stellar motions, and gravitational waves?",
        "required_terms": ["transit photometry", "Doppler spectroscopy", "accretion", "gravitational waves"],
    },
    {
        "query": "How do galaxies supply the larger setting for both exoplanets and black holes, and why are supermassive black holes galactic-scale structures while exoplanets remain local to individual stars?",
        "required_terms": ["supermassive black holes", "galactic", "stars"],
    },
    {
        "query": "How do dark matter and black holes both become visible mainly through gravity, yet play very different explanatory roles in galaxy structure and cosmology?",
        "required_terms": ["gravity", "gravitational lensing", "galaxy structure"],
    },
    {
        "query": "What changes when you compare hidden astronomical mass with hidden biological information: how do dark matter and black holes differ from DNA as unseen causes inferred from observable effects?",
        "required_terms": ["double helix", "nucleotides", "observable effects"],
    },
    {
        "query": "Why is DNA alone not enough without gene expression: how do transcription, translation, and regulation turn stored sequence into functional behavior, analogous to how astronomy infers hidden causes from indirect signals?",
        "required_terms": ["transcription", "translation", "regulation"],
    },
    {
        "query": "Why is reading sequence still not enough without protein folding: how do native state, chaperones, and misfolding show that biological function depends on structure as well as code?",
        "required_terms": ["native state", "chaperones", "misfolding"],
    },
    {
        "query": "How does CRISPR move biology from merely reading hidden information to rewriting it, and what roles do guide RNA, Cas9, and DNA repair play in that shift?",
        "required_terms": ["guide RNA", "Cas9", "DNA repair"],
    },
    {
        "query": "How do neurons add an active information-processing layer beyond DNA editing, with dendrites, axons, and action potentials turning stored biological instructions into signals and computation?",
        "required_terms": ["dendrites", "axons", "action potentials"],
    },
    {
        "query": "Pull the full analogy together: how do exoplanets, black holes, galaxies, dark matter, DNA, gene expression, protein folding, CRISPR, neurons, and synapses trace a progression from indirect detection to mechanistic explanation to plastic intervention?",
        "required_terms": ["indirect detection", "gene expression", "synaptic plasticity", "CRISPR"],
    },
]


def build_level_case(
    prefix: str,
    level: int,
    spec: dict[str, object],
    track: list[str],
    forbidden_titles: list[str],
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=f"{prefix}-level-{level}",
        query=str(spec["query"]),
        kind="direct" if level == 1 else "synthesis",
        expected_titles=track[:level],
        forbidden_titles=forbidden_titles,
        required_terms=list(spec.get("required_terms") or []),
    )


BENCHMARK_CASES = [
    *[
        build_level_case(
            prefix="technical",
            level=index,
            spec=TECHNICAL_CASE_SPECS[index - 1],
            track=TECHNICAL_TRACK,
            forbidden_titles=["Exoplanet", "DNA", "Synapse"],
        )
        for index in range(1, 11)
    ],
    *[
        build_level_case(
            prefix="science",
            level=index,
            spec=SCIENCE_CASE_SPECS[index - 1],
            track=SCIENCE_TRACK,
            forbidden_titles=["Graph database", "Vector database", "Large language model"],
        )
        for index in range(1, 11)
    ],
]


@dataclass(slots=True)
class ClusterSnapshot:
    cluster_id: str
    score: float
    size: int
    titles: dict[str, int]
    top_hits: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class CaseReport:
    case_id: str
    query: str
    kind: str
    expected_titles: list[str]
    forbidden_titles: list[str]
    required_terms: list[str]
    cluster_count: int
    cluster_sizes: list[int]
    top_cluster_titles: dict[str, int]
    matched_expected_titles_top: list[str]
    matched_expected_titles_any: list[str]
    matched_forbidden_titles_top: list[str]
    matched_forbidden_titles_any: list[str]
    matched_required_terms_any: list[str]
    top_cluster_recall: float
    overall_recall: float
    forbidden_clean_score: float
    cluster_requirement_score: float
    required_term_recall: float
    aggregate_score: float
    strict_pass: bool
    draft_answer: str
    clusters: list[ClusterSnapshot]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["clusters"] = [cluster.to_dict() for cluster in self.clusters]
        return payload


def fetch_article_extract(title: str) -> str:
    extracts = fetch_article_extracts([title])
    if title in extracts:
        return extracts[title]
    if extracts:
        return next(iter(extracts.values()))
    raise RuntimeError(f"No extract returned for article `{title}`.")


def fetch_article_extracts(titles: list[str]) -> dict[str, str]:
    if not titles:
        return {}
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": "|".join(titles),
        "format": "json",
    }
    request = Request(
        f"{WIKIPEDIA_API_URL}?{urlencode(params)}",
        headers={"User-Agent": USER_AGENT},
    )
    payload = _request_json(request)
    pages = payload["query"]["pages"]
    extracts: dict[str, str] = {}
    for page in pages.values():
        extract = page.get("extract", "")
        title = page.get("title", "")
        if not title or not extract:
            continue
        extracts[title] = extract
    return extracts


def _request_json(request: Request, attempts: int = 6) -> dict[str, object]:
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


def extract_paragraphs(title: str, text: str, limit: int) -> list[str]:
    paragraphs = []
    for raw in text.split("\n\n"):
        paragraph = " ".join(raw.split())
        if len(paragraph.split()) < 40:
            continue
        paragraphs.append(f"{TITLE_PREFIX}: {title}\n{paragraph}")
        if limit > 0 and len(paragraphs) >= limit:
            break
    return paragraphs


def parse_title(memory_text: str) -> str:
    first_line = memory_text.splitlines()[0].strip()
    if first_line.startswith(f"{TITLE_PREFIX}: "):
        return first_line.replace(f"{TITLE_PREFIX}: ", "", 1)
    return "(unknown)"


def extract_excerpt(memory_text: str, limit: int = 220) -> str:
    lines = memory_text.splitlines()
    body = " ".join(line.strip() for line in lines[1:] if line.strip())
    if not body:
        body = lines[0].strip()
    sentence = SENTENCE_SPLIT_PATTERN.split(body)[0].strip()
    if len(sentence) <= limit:
        return sentence
    return f"{sentence[: limit - 1].rstrip()}…"


def build_benchmark_config(embedding_backend: str) -> MemoryConfig:
    if embedding_backend == "hash":
        return MemoryConfig(
            embedding_backend="hash",
        )
    return MemoryConfig(
        embedding_backend="fastembed",
        embedding_model="snowflake/snowflake-arctic-embed-m",
        embedding_dimensions=768,
        duplicate_threshold=0.985,
        overlap_threshold=0.93,
    )


def load_wikipedia_corpus(
    article_limit_per_title: int,
    article_groups: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    groups = article_groups or ARTICLE_GROUPS
    corpus: dict[str, list[str]] = {}
    for title in [title for titles in groups.values() for title in titles]:
        extract = fetch_article_extract(title)
        paragraphs = extract_paragraphs(title, extract, article_limit_per_title)
        if paragraphs:
            corpus[title] = paragraphs
    return corpus


def run_benchmark_on_corpus(
    corpus: dict[str, list[str]],
    config: MemoryConfig,
    *,
    cases: list[BenchmarkCase] | None = None,
    keep_workspace: bool = False,
    embedder: Embedder | None = None,
) -> dict[str, object]:
    benchmark_cases = cases or BENCHMARK_CASES
    temp_root = Path(tempfile.mkdtemp(prefix="agent-memory-eval-"))
    memory = AgentMemory.initialize(temp_root, config=config, force=True, embedder=embedder)
    report_path = temp_root / "evaluation_report.json"
    workspace_removed = False
    try:
        paragraph_count = 0
        titles = []
        for title, paragraphs in corpus.items():
            titles.append(title)
            for paragraph in paragraphs:
                memory.save(paragraph)
                paragraph_count += 1

        stats = memory.stats().to_dict()
        case_reports = [evaluate_case(memory, case) for case in benchmark_cases]
        summary = summarize_benchmark(case_reports)

        payload = {
            "workspace": str(temp_root),
            "embedding_backend": config.embedding_backend,
            "config": config.to_dict(),
            "titles": titles,
            "paragraph_count": paragraph_count,
            "stats": stats,
            "summary": summary,
            "queries": [report.to_dict() for report in case_reports],
        }
        report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding='utf-8')
        payload["report_path"] = str(report_path)
        return payload
    finally:
        memory.close()
        if not keep_workspace:
            shutil.rmtree(temp_root, ignore_errors=True)
            workspace_removed = True
        # Keep the returned payload consistent even when cleanup happens in finally.
        if report_path.exists():
            pass


def run_wikipedia_benchmark(
    article_limit_per_title: int,
    embedding_backend: str,
    *,
    keep_workspace: bool = False,
    config: MemoryConfig | None = None,
    corpus: dict[str, list[str]] | None = None,
    cases: list[BenchmarkCase] | None = None,
    embedder: Embedder | None = None,
) -> dict[str, object]:
    resolved_config = config or build_benchmark_config(embedding_backend)
    resolved_corpus = corpus or load_wikipedia_corpus(article_limit_per_title)
    payload = run_benchmark_on_corpus(
        resolved_corpus,
        resolved_config,
        cases=cases,
        keep_workspace=keep_workspace,
        embedder=embedder,
    )
    if not keep_workspace:
        payload["workspace_removed"] = True
    return payload


def evaluate_case(memory: AgentMemory, case: BenchmarkCase) -> CaseReport:
    recall = memory.recall(case.query, max_clusters=5)
    cluster_sizes = [len(cluster.memory_ids) for cluster in recall.clusters]
    cluster_snapshots = [snapshot_cluster(cluster) for cluster in recall.clusters]
    top_cluster = cluster_snapshots[0] if cluster_snapshots else None
    top_titles = top_cluster.titles if top_cluster else {}
    all_titles = set()
    for cluster in cluster_snapshots:
        all_titles.update(cluster.titles.keys())

    matched_expected_top = [title for title in case.expected_titles if title in top_titles]
    matched_expected_any = [title for title in case.expected_titles if title in all_titles]
    matched_forbidden_top = [title for title in case.forbidden_titles if title in top_titles]
    matched_forbidden_any = [title for title in case.forbidden_titles if title in all_titles]
    all_cluster_text = "\n".join(
        hit.text.lower()
        for cluster in recall.clusters
        for hit in cluster.hits
    )
    matched_required_any = [
        term for term in case.required_terms if term.lower() in all_cluster_text
    ]

    top_cluster_recall = (
        len(matched_expected_top) / len(case.expected_titles)
        if case.expected_titles
        else 1.0
    )
    overall_recall = (
        len(matched_expected_any) / len(case.expected_titles)
        if case.expected_titles
        else 1.0
    )
    forbidden_clean_score = (
        1.0 - (len(matched_forbidden_any) / len(case.forbidden_titles))
        if case.forbidden_titles
        else 1.0
    )
    cluster_requirement_score = min(
        len(cluster_snapshots) / case.min_clusters,
        1.0,
    )
    required_term_recall = (
        len(matched_required_any) / len(case.required_terms)
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
    strict_pass = (
        overall_recall == 1.0
        and not matched_forbidden_any
        and len(cluster_snapshots) >= case.min_clusters
        and required_term_recall == 1.0
    )

    return CaseReport(
        case_id=case.case_id,
        query=case.query,
        kind=case.kind,
        expected_titles=case.expected_titles,
        forbidden_titles=case.forbidden_titles,
        required_terms=case.required_terms,
        cluster_count=len(cluster_snapshots),
        cluster_sizes=cluster_sizes,
        top_cluster_titles=top_titles,
        matched_expected_titles_top=matched_expected_top,
        matched_expected_titles_any=matched_expected_any,
        matched_forbidden_titles_top=matched_forbidden_top,
        matched_forbidden_titles_any=matched_forbidden_any,
        matched_required_terms_any=matched_required_any,
        top_cluster_recall=round(top_cluster_recall, 4),
        overall_recall=round(overall_recall, 4),
        forbidden_clean_score=round(forbidden_clean_score, 4),
        cluster_requirement_score=round(cluster_requirement_score, 4),
        required_term_recall=round(required_term_recall, 4),
        aggregate_score=round(aggregate_score, 4),
        strict_pass=strict_pass,
        draft_answer=synthesize_answer(cluster_snapshots),
        clusters=cluster_snapshots,
    )


def snapshot_cluster(cluster: object) -> ClusterSnapshot:
    titles = Counter(parse_title(hit.text) for hit in cluster.hits)
    top_hits = [
        {
            "memory_id": hit.memory_id,
            "title": parse_title(hit.text),
            "query_similarity": hit.query_similarity,
            "preview": hit.preview(180),
            "excerpt": extract_excerpt(hit.text, 220),
        }
        for hit in cluster.hits[:6]
    ]
    return ClusterSnapshot(
        cluster_id=cluster.cluster_id,
        score=cluster.score,
        size=len(cluster.memory_ids),
        titles=dict(titles),
        top_hits=top_hits,
    )


def synthesize_answer(clusters: list[ClusterSnapshot], max_clusters: int = 2) -> str:
    if not clusters:
        return "No relevant memory clusters were retrieved."
    parts = []
    for cluster in clusters[:max_clusters]:
        excerpts = []
        seen_titles: set[str] = set()
        for hit in cluster.top_hits:
            title = hit["title"]
            if title in seen_titles:
                continue
            seen_titles.add(title)
            excerpts.append(f"{title}: {hit['excerpt']}")
            if len(excerpts) >= 2:
                break
        cluster_label = ", ".join(seen_titles) if seen_titles else cluster.cluster_id
        parts.append(f"{cluster.cluster_id} [{cluster_label}] " + " ".join(excerpts))
    return " ".join(parts)


def score_case(
    *,
    kind: str,
    top_cluster_recall: float,
    overall_recall: float,
    forbidden_clean_score: float,
    cluster_requirement_score: float,
    required_term_recall: float,
) -> float:
    if kind == "synthesis":
        return (
            0.38 * overall_recall
            + 0.15 * top_cluster_recall
            + 0.2 * cluster_requirement_score
            + 0.15 * forbidden_clean_score
            + 0.12 * required_term_recall
        )
    return (
        0.42 * top_cluster_recall
        + 0.2 * overall_recall
        + 0.15 * forbidden_clean_score
        + 0.10 * cluster_requirement_score
        + 0.13 * required_term_recall
    )


def summarize_benchmark(case_reports: list[CaseReport]) -> dict[str, object]:
    if not case_reports:
        return {
            "benchmark_score": 0.0,
            "average_top_cluster_recall": 0.0,
            "average_overall_recall": 0.0,
            "average_forbidden_clean_score": 0.0,
            "average_required_term_recall": 0.0,
            "strict_pass_count": 0,
            "strict_pass_rate": 0.0,
            "cluster_requirement_pass_count": 0,
            "case_scores": {},
        }

    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 4)

    benchmark_score = avg([report.aggregate_score for report in case_reports])
    strict_pass_count = sum(1 for report in case_reports if report.strict_pass)
    cluster_requirement_pass_count = sum(
        1
        for report in case_reports
        if report.cluster_count
        >= next(
            case.min_clusters for case in BENCHMARK_CASES if case.case_id == report.case_id
        )
    )
    return {
        "benchmark_score": benchmark_score,
        "average_top_cluster_recall": avg(
            [report.top_cluster_recall for report in case_reports]
        ),
        "average_overall_recall": avg(
            [report.overall_recall for report in case_reports]
        ),
        "average_forbidden_clean_score": avg(
            [report.forbidden_clean_score for report in case_reports]
        ),
        "average_required_term_recall": avg(
            [report.required_term_recall for report in case_reports]
        ),
        "strict_pass_count": strict_pass_count,
        "strict_pass_rate": round(strict_pass_count / len(case_reports), 4),
        "cluster_requirement_pass_count": cluster_requirement_pass_count,
        "cluster_requirement_pass_rate": round(
            sum(
                1
                for report in case_reports
                if report.cluster_count
                >= next(
                    case.min_clusters
                    for case in BENCHMARK_CASES
                    if case.case_id == report.case_id
                )
            )
            / len(case_reports),
            4,
        ),
        "case_scores": {
            report.case_id: report.aggregate_score for report in case_reports
        },
    }


def print_benchmark_report(payload: dict[str, object]) -> None:
    print("Agent Memory benchmark")
    print(f"Embedding backend: {payload['embedding_backend']}")
    print(f"Ingested titles: {len(payload['titles'])}")
    print(f"Saved paragraphs: {payload['paragraph_count']}")
    stats = payload["stats"]
    print(
        "Graph stats: "
        f"memories={stats['memory_count']} "
        f"similarity_edges={stats['similarity_edge_count']} "
        f"next_edges={stats['next_edge_count']}"
    )
    summary = payload["summary"]
    print(
        "Benchmark summary: "
        f"score={summary['benchmark_score']} "
        f"top_recall={summary['average_top_cluster_recall']} "
        f"overall_recall={summary['average_overall_recall']} "
        f"forbidden_clean={summary['average_forbidden_clean_score']} "
        f"required_terms={summary['average_required_term_recall']} "
        f"strict_pass={summary['strict_pass_count']}/{len(payload['queries'])}"
    )

    for query_report in payload["queries"]:
        print("\n---")
        print(f"Case: {query_report['case_id']}")
        print(f"Query: {query_report['query']}")
        print(
            "Scores: "
            f"aggregate={query_report['aggregate_score']} "
            f"top_recall={query_report['top_cluster_recall']} "
            f"overall_recall={query_report['overall_recall']} "
            f"forbidden_clean={query_report['forbidden_clean_score']}"
        )
        print(f"Expected(any): {query_report['matched_expected_titles_any']}")
        print(f"Forbidden(any): {query_report['matched_forbidden_titles_any']}")
        print(f"Required(any): {query_report['matched_required_terms_any']}")
        print(f"Cluster sizes: {query_report['cluster_sizes']}")
        print(f"Draft answer: {query_report['draft_answer']}")
        for cluster in query_report["clusters"][:3]:
            print(
                f"  {cluster['cluster_id']} score={cluster['score']} size={cluster['size']} titles={cluster['titles']}"
            )
            for hit in cluster["top_hits"][:3]:
                print(
                    f"    [{hit['query_similarity']}] {hit['title']} :: {hit['preview']}"
                )
    if payload.get("workspace_removed"):
        print("\nWorkspace was removed after the run.")
    else:
        print(f"\nWorkspace: {payload['workspace']}")
        print(f"Report JSON: {payload['report_path']}")
