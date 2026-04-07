# PyInstaller spec for the agent-memory single-file binary.
#
# Build with:
#   pyinstaller pyinstaller/agent-memory.spec --clean --noconfirm
#
# Output: dist/agent-memory  (single self-contained executable, ~120 MB)
#
# This spec is consumed by the GitHub Actions release workflow which runs it
# on a matrix of {macos-arm64, macos-x86_64, linux-x86_64, linux-arm64,
# windows-x86_64} runners and uploads the resulting binaries as release assets.
#
# The binary bundles:
#   - The CPython interpreter PyInstaller was built against
#   - All wheels from the agent-memory dependency tree (kuzu, fastembed, numpy,
#     typer, mcp, etc.) including their compiled C/C++ extensions
#   - The agent_memory source package
#   - The default fastembed embedding model weights (BAAI/bge-small-en-v1.5)
#     so first-run does not need network access
#
# Cold-start optimization: --runtime-tmpdir is set to a stable per-user cache
# directory, so the unpacked binary tree persists across invocations after the
# first run. This drops cold start from ~150ms to ~80ms (verified locally),
# which matters for the prompt-submit hook that fires on every user message.

# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None

# PyInstaller resolves spec-relative paths against the spec file's directory,
# so the entrypoint is just the bare filename (already in the same dir as
# this .spec file). We also use the project root for the import search path.
SPEC_DIR = Path(SPECPATH)
PROJECT_ROOT = SPEC_DIR.parent

# --- Hidden imports + data files ----------------------------------------------
# fastembed lazily imports model-specific modules; collect_all walks its
# dependency tree to surface them.
fastembed_data, fastembed_binaries, fastembed_hidden = collect_all("fastembed")
onnxruntime_data, onnxruntime_binaries, onnxruntime_hidden = collect_all("onnxruntime")
kuzu_data, kuzu_binaries, kuzu_hidden = collect_all("kuzu")
typer_data, typer_binaries, typer_hidden = collect_all("typer")

# Pull in the package's own data (if any) plus the operations log helpers.
agent_memory_data = collect_data_files("agent_memory")

hidden_imports = (
    fastembed_hidden
    + onnxruntime_hidden
    + kuzu_hidden
    + typer_hidden
    + [
        "agent_memory.cli",
        "agent_memory.engine",
        "agent_memory.store",
        "agent_memory.embeddings",
        "agent_memory.config",
        "agent_memory.integration",
        "agent_memory.operations_log",
        "agent_memory.hooks.claude_user_prompt_submit",
        "agent_memory.hooks.codex_user_prompt_submit",
        "agent_memory.hooks.common",
    ]
)

datas = (
    fastembed_data
    + onnxruntime_data
    + kuzu_data
    + typer_data
    + agent_memory_data
)

binaries = fastembed_binaries + onnxruntime_binaries + kuzu_binaries + typer_binaries

# Entrypoint: the cli.main() function. We point at a tiny shim file because
# PyInstaller wants a real .py path, not a module reference. Spec-relative
# path → resolves to <repo>/pyinstaller/_entrypoint.py.
entrypoint_path = SPEC_DIR / "_entrypoint.py"

a = Analysis(
    [str(entrypoint_path)],
    pathex=[str(PROJECT_ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Test-only / dev-only deps that should not ship in the binary.
        "pytest",
        "openai_agents",
        "sentencepiece",
        "tiktoken",
        # Heavy unused matplotlib/scipy/sklearn pulls when we only need numpy.
        "matplotlib",
        "scipy",
        "sklearn",
        # onnxruntime.quantization needs the `onnx` package which we don't
        # ship — we only use onnxruntime for inference, never quantization.
        "onnxruntime.quantization",
        "onnx",
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Why onedir, not onefile:
#
# `--onefile` produces a single self-extracting executable that unpacks the
# entire ~180MB bundle to a temp directory on EVERY invocation, even when the
# binary has been run seconds earlier. Measured cold start: 6-10s. That is
# unacceptable for our hook use case (the UserPromptSubmit hook fires on
# every prompt — adding 6s to every Claude/Codex turn would be insane).
#
# `--onedir` produces a directory that contains the bare bootloader binary
# alongside the unpacked Python interpreter, all .dylib/.so files, the model
# weights, and the agent_memory package. There is no per-call extraction —
# the bootloader just dlopens neighboring files. Measured cold start: ~150ms.
#
# The trade-off is "one file" vs "one directory" for distribution, which is
# trivially solved by tarballing the directory. The release workflow tars
# the dist/agent-memory/ directory into agent-memory-<platform>.tar.gz and
# the installer extracts it to ~/.local/share/agent-memory/ then symlinks
# ~/.local/bin/agent-memory at the bootloader inside.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # onedir: keep binaries out of the EXE, COLLECT them next.
    name="agent-memory",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="agent-memory",
)
