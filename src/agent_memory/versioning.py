from __future__ import annotations

import re


__version__ = "0.2.1"

_PEP440_VERSION_RE = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"(?:(?P<pre>a|alpha|b|beta|rc)(?P<pre_num>\d+))?$",
    re.IGNORECASE,
)
_DISPLAY_VERSION_RE = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"(?:-(?P<pre>alpha|beta|rc)\.(?P<pre_num>\d+))?$",
    re.IGNORECASE,
)
_PRECEDENCE = {
    "a": 0,
    "alpha": 0,
    "b": 1,
    "beta": 1,
    "rc": 2,
    None: 3,
}


def version_to_display(version: str) -> str:
    match = _PEP440_VERSION_RE.fullmatch(version)
    if not match:
        return version

    major = match.group("major")
    minor = match.group("minor")
    patch = match.group("patch")
    pre = match.group("pre")
    pre_num = match.group("pre_num")

    rendered = f"{major}.{minor}.{patch}"
    if pre and pre_num:
        pre_label = {"a": "alpha", "b": "beta", "alpha": "alpha", "beta": "beta", "rc": "rc"}[pre.lower()]
        rendered += f"-{pre_label}.{pre_num}"
    return rendered


def version_to_release_tag(version: str) -> str:
    return f"v{version_to_display(version)}"


def version_key(value: str) -> tuple[int, int, int, int, int]:
    cleaned = value.strip().lstrip("v")

    match = _DISPLAY_VERSION_RE.fullmatch(cleaned)
    if match:
        pre = match.group("pre")
        pre_num = int(match.group("pre_num") or 0)
        return (
            int(match.group("major")),
            int(match.group("minor")),
            int(match.group("patch")),
            _PRECEDENCE[pre.lower() if pre else None],
            pre_num,
        )

    match = _PEP440_VERSION_RE.fullmatch(cleaned)
    if match:
        pre = match.group("pre")
        pre_num = int(match.group("pre_num") or 0)
        return (
            int(match.group("major")),
            int(match.group("minor")),
            int(match.group("patch")),
            _PRECEDENCE[pre.lower() if pre else None],
            pre_num,
        )

    parts: list[int] = []
    for chunk in cleaned.split("-", 1)[0].split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2], _PRECEDENCE[None], 0)


__display_version__ = version_to_display(__version__)
__release_tag__ = version_to_release_tag(__version__)
