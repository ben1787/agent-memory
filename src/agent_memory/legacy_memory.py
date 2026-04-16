from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from agent_memory.models import MemoryMetadata


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
BULLET_RE = re.compile(r"^(\s*)[-*]\s+(.*\S.*)$")
DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2},\s*")


@dataclass(slots=True)
class LegacyMemoryEntry:
    line_number: int
    section_path: tuple[str, ...]
    text: str


def _normalize_heading(value: str) -> str:
    return " ".join(value.casefold().split())


def _capture_section(section_path: tuple[str, ...]) -> bool:
    if not section_path:
        return False
    normalized = [_normalize_heading(item) for item in section_path]
    if any("rules" in item or "legacy note" in item for item in normalized):
        return False
    return any(
        item.endswith("notes")
        or "durable" in item
        or "point-in-time" in item
        or "point in time" in item
        for item in normalized[1:]
    )


def parse_legacy_memory_markdown(path: Path) -> list[LegacyMemoryEntry]:
    lines = path.read_text(encoding="utf-8").splitlines()
    headings: list[str] = []
    entries: list[LegacyMemoryEntry] = []
    current_item_lines: list[str] = []
    current_item_line_number: int | None = None
    current_section_path: tuple[str, ...] = ()

    def flush_item() -> None:
        nonlocal current_item_lines, current_item_line_number, current_section_path
        if not current_item_lines or current_item_line_number is None:
            current_item_lines = []
            current_item_line_number = None
            current_section_path = ()
            return
        text = " ".join(part.strip() for part in current_item_lines if part.strip()).strip()
        if text:
            entries.append(
                LegacyMemoryEntry(
                    line_number=current_item_line_number,
                    section_path=current_section_path,
                    text=text,
                )
            )
        current_item_lines = []
        current_item_line_number = None
        current_section_path = ()

    for line_number, raw_line in enumerate(lines, start=1):
        heading_match = HEADING_RE.match(raw_line)
        if heading_match:
            flush_item()
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            headings = headings[: level - 1]
            headings.append(heading)
            continue

        bullet_match = BULLET_RE.match(raw_line)
        if bullet_match:
            flush_item()
            section_path = tuple(headings)
            if not _capture_section(section_path):
                continue
            current_section_path = section_path
            current_item_line_number = line_number
            current_item_lines = [bullet_match.group(2).strip()]
            continue

        if current_item_line_number is None:
            continue

        stripped = raw_line.strip()
        if not stripped:
            continue

        current_item_lines.append(stripped)

    flush_item()
    return entries


def _truncate_title(value: str, *, limit: int = 72) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) <= limit:
        return cleaned
    clipped = cleaned[:limit].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip(" ,:;.-") + "..."


def generate_legacy_title(text: str) -> str:
    candidate = DATE_PREFIX_RE.sub("", text.strip())
    for delimiter in (": ", ". ", "; "):
        if delimiter in candidate:
            candidate = candidate.split(delimiter, 1)[0]
            break
    return _truncate_title(candidate or text.strip())


def detect_legacy_environment(text: str) -> str:
    lowered = f" {text.casefold()} "
    if " localhost " in lowered or " local " in lowered:
        return "local"
    if " production " in lowered or " prod " in lowered:
        return "prod"
    if " development " in lowered or " dev " in lowered:
        return "dev"
    if " staging " in lowered or " qa " in lowered:
        return "qa"
    if " test mode " in lowered or " sandbox " in lowered or " test " in lowered:
        return "test"
    return "legacy"


def entry_to_metadata(
    entry: LegacyMemoryEntry,
    *,
    default_subsystem: str,
    workstream_override: str | None = None,
    environment_override: str | None = None,
    kind_override: str | None = None,
) -> MemoryMetadata:
    section_name = entry.section_path[-1] if entry.section_path else "legacy memory"
    normalized_section = _normalize_heading(section_name)
    default_kind = "historical" if "point-in-time" in normalized_section or "point in time" in normalized_section else "operational"
    workstream = workstream_override or normalized_section.replace("point-in-time", "point in time")
    return MemoryMetadata(
        title=generate_legacy_title(entry.text),
        kind=kind_override or default_kind,
        subsystem=default_subsystem,
        workstream=workstream,
        environment=environment_override or detect_legacy_environment(entry.text),
    )
