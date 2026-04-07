from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import signal
import shutil
import subprocess
import struct
import tempfile
import termios
import time

from agent_memory.config import MemoryConfig, init_project
from agent_memory.engine import open_memory_with_retry
from agent_memory.hooks.common import hook_log_entries
from agent_memory.integration import codex_project_trust_state
from agent_memory.integration import (
    ensure_local_git_excludes,
    install_claude_hooks,
    install_codex_feature_flag,
    install_codex_hooks,
    install_codex_mcp_server,
    install_codex_project_trust,
    install_mcp_server,
    remove_local_git_excludes,
    uninstall_claude_hooks,
    uninstall_codex_feature_flag,
    uninstall_codex_hooks,
    uninstall_codex_mcp_server,
    uninstall_codex_project_trust,
    uninstall_mcp_server,
)
from agent_memory.store import GraphStore


FIRST_PROMPT = """This repo uses the installed `agent-memory` package.

Before doing anything else:
1. Tell me whether Agent Memory instructions were injected into your context.
2. Call `save_memory` and store exactly these two memories:
   - This repo is being used to test whether agent-memory hooks are active.
   - The billing webhook handler lives in services/billing/webhooks.py.
3. Then reply only with: saved
"""

SECOND_PROMPT = (
    "Use agent-memory if useful. Do not search the repo unless recall fails. "
    "Reply only with the exact path to the billing webhook handler."
)
EXPECTED_MEMORY = "The billing webhook handler lives in services/billing/webhooks.py."
EXPECTED_FINAL_ANSWER = "services/billing/webhooks.py"
DEFAULT_ROWS = 24
DEFAULT_COLUMNS = 80


class SmokeTestError(RuntimeError):
    pass


@dataclass(slots=True)
class TranscriptSummary:
    user_messages: list[str]
    final_answers: list[str]
    exec_commands: list[str]
    tool_calls: list[str]


@dataclass(slots=True)
class SmokeTestResult:
    repo_root: str
    first_session_file: str
    second_session_file: str
    used_temp_repo: bool
    uninstall_verified: bool
    baseline_memory_count: int
    post_save_memory_count: int
    hook_event_count: int
    first_pre_submit_verified: bool
    second_pre_submit_verified: bool
    save_path_verified: bool
    read_path_verified: bool
    recall_top_hit: str | None
    first_final_answer: str | None
    second_final_answer: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class PtyProcess:
    pid: int
    returncode: int | None = None

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        try:
            child_pid, status = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            self.returncode = 0
            return self.returncode
        if child_pid == 0:
            return None
        if os.WIFEXITED(status):
            self.returncode = os.WEXITSTATUS(status)
        elif os.WIFSIGNALED(status):
            self.returncode = -os.WTERMSIG(status)
        else:
            self.returncode = status
        return self.returncode

    def terminate(self) -> None:
        if self.poll() is not None:
            return
        try:
            os.kill(self.pid, signal.SIGTERM)
        except ProcessLookupError:
            self.returncode = 0

    def kill(self) -> None:
        if self.poll() is not None:
            return
        try:
            os.kill(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is not None:
            return self.returncode
        deadline = None if timeout is None else time.time() + timeout
        while True:
            polled = self.poll()
            if polled is not None:
                return polled
            if deadline is not None and time.time() >= deadline:
                raise subprocess.TimeoutExpired(cmd=f"pid {self.pid}", timeout=timeout)
            time.sleep(0.1)


def _codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".codex").resolve()


def _parse_transcript(text: str) -> TranscriptSummary:
    user_messages: list[str] = []
    final_answers: list[str] = []
    exec_commands: list[str] = []
    tool_calls: list[str] = []

    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        payload = event.get("payload") or {}
        event_type = event.get("type")
        if event_type == "response_item":
            item_type = payload.get("type")
            if item_type == "message" and payload.get("role") == "user":
                for content in payload.get("content") or []:
                    if content.get("type") == "input_text":
                        user_messages.append(content.get("text", ""))
            elif item_type == "function_call":
                tool_name = payload.get("name")
                if isinstance(tool_name, str):
                    tool_calls.append(tool_name)
                if tool_name == "exec_command":
                    arguments = payload.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            parsed = json.loads(arguments)
                        except json.JSONDecodeError:
                            parsed = {}
                        cmd = parsed.get("cmd")
                        if isinstance(cmd, str):
                            exec_commands.append(cmd)
        elif event_type == "event_msg":
            if payload.get("type") == "agent_message" and payload.get("phase") == "final_answer":
                message = payload.get("message")
                if isinstance(message, str):
                    final_answers.append(message)

    return TranscriptSummary(
        user_messages=user_messages,
        final_answers=final_answers,
        exec_commands=exec_commands,
        tool_calls=tool_calls,
    )


def _session_file_for_session_id(session_id: str, *, timeout_seconds: float) -> Path:
    sessions_root = _codex_home() / "sessions"
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        candidates = sorted(
            sessions_root.rglob(f"*{session_id}.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            return candidate
        time.sleep(0.5)

    raise SmokeTestError(
        f"Timed out waiting for a Codex session transcript for session {session_id} under {sessions_root}."
    )


def _read_transcript_summary(session_file: Path) -> TranscriptSummary:
    return _parse_transcript(session_file.read_text(errors="ignore"))


def _run_checked(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout_seconds: int = 120,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        raise SmokeTestError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n{details}"
        )
    return result


def _hook_entries(repo_root: Path) -> list[dict[str, object]]:
    return hook_log_entries(repo_root)


def _latest_hook_session_id(repo_root: Path) -> str | None:
    for entry in reversed(_hook_entries(repo_root)):
        payload = entry.get("payload")
        if not isinstance(payload, dict):
            continue
        session_id = payload.get("session_id")
        if isinstance(session_id, str) and session_id:
            return session_id
    return None


def _has_hook_action(entries: list[dict[str, object]], session_id: str, hook: str, action: str) -> bool:
    for entry in entries:
        if entry.get("hook") != hook or entry.get("action") != action:
            continue
        payload = entry.get("payload")
        if isinstance(payload, dict) and payload.get("session_id") == session_id:
            return True
    return False


def _remove_if_empty(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        return


def _ensure_clean_uninstall(repo_root: Path) -> bool:
    local_exclude_entries = [
        ".agent-memory/",
        ".claude/settings.local.json",
        ".codex/config.toml",
        ".codex/hooks.json",
        ".mcp.json",
    ]
    uninstall_mcp_server(repo_root)
    uninstall_codex_hooks(repo_root)
    uninstall_codex_feature_flag(repo_root)
    uninstall_codex_mcp_server(repo_root)
    uninstall_claude_hooks(repo_root)
    remove_local_git_excludes(repo_root, entries=local_exclude_entries)
    uninstall_codex_project_trust(repo_root)
    if (repo_root / ".agent-memory").exists():
        shutil.rmtree(repo_root / ".agent-memory")
    _remove_if_empty(repo_root / ".codex")
    _remove_if_empty(repo_root / ".claude")

    missing_paths = [
        repo_root / ".agent-memory",
        repo_root / ".mcp.json",
        repo_root / ".codex" / "config.toml",
        repo_root / ".codex" / "hooks.json",
        repo_root / ".claude" / "settings.local.json",
    ]
    trust_state, trust_error = codex_project_trust_state(repo_root)
    if trust_error:
        raise SmokeTestError(trust_error)
    return all(not path.exists() for path in missing_paths) and trust_state is False


def _init_repo(repo_root: Path) -> None:
    config = MemoryConfig(embedding_backend="hash")
    project = init_project(repo_root, config=config, force=True)
    store = GraphStore(project.db_path, config.embedding_dimensions)
    store.close()

    ensure_local_git_excludes(project.root)
    install_mcp_server(project.root)
    install_codex_feature_flag(project.root)
    install_codex_mcp_server(project.root)
    install_codex_hooks(project.root)
    install_codex_project_trust(project.root)
    install_claude_hooks(project.root)


def _stats(repo_root: Path) -> dict[str, object]:
    memory = open_memory_with_retry(repo_root, exact=True, read_only=True)
    try:
        return memory.stats().to_dict()
    finally:
        memory.close()


def _recall(repo_root: Path, query: str) -> dict[str, object]:
    memory = open_memory_with_retry(repo_root, exact=True, read_only=True)
    try:
        return memory.recall(query).to_dict()
    finally:
        memory.close()


def _drain_pty(master_fd: int) -> str:
    chunks: list[bytes] = []
    while True:
        ready, _, _ = select.select([master_fd], [], [], 0.05)
        if not ready:
            break
        try:
            chunk = os.read(master_fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="ignore")


def _answer_terminal_queries(output: str, master_fd: int) -> None:
    if "\x1b[6n" in output:
        os.write(master_fd, b"\x1b[1;1R")
    if "\x1b[c" in output:
        os.write(master_fd, b"\x1b[?1;2c")
    if "\x1b]10;?\x1b\\" in output:
        os.write(master_fd, b"\x1b]10;rgb:ffff/ffff/ffff\x1b\\")
    if "\x1b]11;?\x1b\\" in output:
        os.write(master_fd, b"\x1b]11;rgb:0000/0000/0000\x1b\\")
    if "\x1b[?u" in output:
        os.write(master_fd, b"\x1b[?0u")


def _start_codex_session(codex_bin: str, repo_root: Path, prompt: str) -> tuple[PtyProcess, int]:
    env = os.environ.copy()
    env["TERM"] = "dumb"
    pid, master_fd = pty.fork()
    if pid == 0:
        try:
            os.execvpe(codex_bin, [codex_bin, "-C", str(repo_root), prompt], env)
        except OSError as exc:
            os.write(2, f"Failed to exec Codex: {exc}\n".encode("utf-8", errors="ignore"))
            os._exit(1)
    fcntl.ioctl(master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", DEFAULT_ROWS, DEFAULT_COLUMNS, 0, 0))
    return PtyProcess(pid=pid), master_fd


def _wait_for(
    check,
    *,
    timeout_seconds: int,
    failure_message: str,
    process: PtyProcess | None = None,
    master_fd: int | None = None,
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if master_fd is not None:
            output = _drain_pty(master_fd)
            if output:
                _answer_terminal_queries(output, master_fd)
                if "Continue anyway?" in output:
                    os.write(master_fd, b"y\n")
        if process is not None and process.poll() is not None:
            transcript = _drain_pty(master_fd) if master_fd is not None else ""
            raise SmokeTestError(f"Codex exited while waiting.\n{transcript}")
        if check():
            return
        time.sleep(0.5)
    raise SmokeTestError(failure_message)


def _close_live_session(process: PtyProcess | None, master_fd: int | None) -> None:
    if master_fd is not None:
        try:
            os.write(master_fd, b"/exit\r")
        except OSError:
            pass
        try:
            _drain_pty(master_fd)
        except OSError:
            pass
        try:
            os.close(master_fd)
        except OSError:
            pass
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _run_codex_prompt_turn(
    codex_bin: str,
    repo_root: Path,
    prompt: str,
    *,
    timeout_seconds: int,
    expected_final_answer: str,
    require_save: bool = False,
    require_recall: bool = False,
) -> tuple[str, Path, TranscriptSummary]:
    hook_count_before = len(_hook_entries(repo_root))
    process: PtyProcess | None = None
    master_fd: int | None = None
    try:
        process, master_fd = _start_codex_session(codex_bin, repo_root, prompt)
        _wait_for(
            lambda: len(_hook_entries(repo_root)) >= hook_count_before + 2,
            timeout_seconds=timeout_seconds,
            failure_message="Timed out waiting for UserPromptSubmit hook entries.",
            process=process,
            master_fd=master_fd,
        )
        session_id = _latest_hook_session_id(repo_root)
        if not session_id:
            raise SmokeTestError("Could not find a Codex session_id after prompt submission.")
        session_file = _session_file_for_session_id(session_id, timeout_seconds=timeout_seconds)
        if require_save:
            _wait_for(
                lambda: _contains_save_activity(_read_transcript_summary(session_file)),
                timeout_seconds=timeout_seconds,
                failure_message="Timed out waiting for save activity in the Codex transcript.",
                process=process,
                master_fd=master_fd,
            )
        if require_recall:
            _wait_for(
                lambda: _contains_recall_activity(_read_transcript_summary(session_file)),
                timeout_seconds=timeout_seconds,
                failure_message="Timed out waiting for recall activity in the Codex transcript.",
                process=process,
                master_fd=master_fd,
            )
        _wait_for(
            lambda: any(
                expected_final_answer in answer
                for answer in _read_transcript_summary(session_file).final_answers
            ),
            timeout_seconds=timeout_seconds,
            failure_message="Timed out waiting for the Codex final answer.",
            process=process,
            master_fd=master_fd,
        )
        _wait_for(
            lambda: len(_hook_entries(repo_root)) >= hook_count_before + 4,
            timeout_seconds=timeout_seconds,
            failure_message="Timed out waiting for Stop hook entries.",
            process=process,
            master_fd=master_fd,
        )
        return session_id, session_file, _read_transcript_summary(session_file)
    finally:
        _close_live_session(process, master_fd)


def _contains_save_activity(summary: TranscriptSummary) -> bool:
    if "save_memory" in summary.tool_calls:
        return True
    return any("agent-memory save" in command for command in summary.exec_commands)


def _contains_recall_activity(summary: TranscriptSummary) -> bool:
    if "recall_memories" in summary.tool_calls:
        return True
    return any("agent-memory recall" in command for command in summary.exec_commands)


def _install_from_source(reinstall_from: Path) -> None:
    uv_bin = shutil.which("uv")
    if not uv_bin:
        raise SmokeTestError("`uv` is required for `--reinstall-from`, but it was not found on PATH.")
    _run_checked([uv_bin, "tool", "install", "--reinstall", str(reinstall_from.resolve())])


def run_codex_smoke_test(
    *,
    project_root: Path | None = None,
    destructive: bool = False,
    reinstall_from: Path | None = None,
    keep_repo: bool = False,
    timeout_seconds: int = 120,
) -> SmokeTestResult:
    if project_root is not None and not destructive:
        raise SmokeTestError("Refusing to modify an explicit project without `--destructive`.")

    agent_memory_bin = shutil.which("agent-memory")
    if not agent_memory_bin:
        raise SmokeTestError("`agent-memory` was not found on PATH.")

    codex_bin = shutil.which("codex")
    if not codex_bin:
        raise SmokeTestError("`codex` was not found on PATH.")

    if reinstall_from is not None:
        _install_from_source(reinstall_from)
        agent_memory_bin = shutil.which("agent-memory")
        if not agent_memory_bin:
            raise SmokeTestError("`agent-memory` was not found on PATH after reinstall.")

    temp_dir: Path | None = None
    used_temp_repo = project_root is None
    if project_root is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="agent-memory-smoke."))
        repo_root = temp_dir / "repo"
        repo_root.mkdir(parents=True, exist_ok=True)
        _run_checked(["git", "init", str(repo_root)])
        _init_repo(repo_root)
    else:
        repo_root = project_root.resolve()
        repo_root.mkdir(parents=True, exist_ok=True)
        if not (repo_root / ".git").exists():
            _run_checked(["git", "init", str(repo_root)])

    try:
        uninstall_verified = _ensure_clean_uninstall(repo_root)
        _init_repo(repo_root)

        baseline_stats = _stats(repo_root)
        baseline_memory_count = int(baseline_stats.get("memory_count", -1))
        if baseline_memory_count != 0:
            raise SmokeTestError(f"Expected empty store after reinstall, got memory_count={baseline_memory_count}.")

        hook_log_before = _hook_entries(repo_root)
        if hook_log_before:
            raise SmokeTestError("Expected empty hook log before starting the live session.")

        first_session_id, first_session_file, first_summary = _run_codex_prompt_turn(
            codex_bin,
            repo_root,
            FIRST_PROMPT,
            timeout_seconds=timeout_seconds,
            expected_final_answer="saved",
            require_save=True,
        )

        after_save_stats = _stats(repo_root)
        post_save_memory_count = int(after_save_stats.get("memory_count", -1))
        if post_save_memory_count < 2:
            raise SmokeTestError(
                f"Expected at least two memories after first turn, got memory_count={post_save_memory_count}."
            )

        recall_payload = _recall(repo_root, "billing webhook handler")
        hits = recall_payload.get("hits") or []
        recall_top_hit = hits[0]["text"] if hits else None
        if recall_top_hit != EXPECTED_MEMORY:
            raise SmokeTestError(f"Unexpected recall top hit: {recall_top_hit!r}")

        second_session_id, second_session_file, second_summary = _run_codex_prompt_turn(
            codex_bin,
            repo_root,
            SECOND_PROMPT,
            timeout_seconds=timeout_seconds,
            expected_final_answer=EXPECTED_FINAL_ANSWER,
            require_recall=True,
        )
        hook_log = _hook_entries(repo_root)
        if first_session_id == second_session_id:
            raise SmokeTestError("Expected the read verification to run in a fresh second Codex session.")

        return SmokeTestResult(
            repo_root=str(repo_root),
            first_session_file=str(first_session_file),
            second_session_file=str(second_session_file),
            used_temp_repo=used_temp_repo,
            uninstall_verified=uninstall_verified,
            baseline_memory_count=baseline_memory_count,
            post_save_memory_count=post_save_memory_count,
            hook_event_count=len(hook_log),
            first_pre_submit_verified=_has_hook_action(
                hook_log, first_session_id, "codex_user_prompt_submit", "start"
            )
            and _has_hook_action(hook_log, first_session_id, "codex_user_prompt_submit", "inject_context"),
            second_pre_submit_verified=_has_hook_action(
                hook_log, second_session_id, "codex_user_prompt_submit", "start"
            )
            and _has_hook_action(hook_log, second_session_id, "codex_user_prompt_submit", "inject_context"),
            save_path_verified=_contains_save_activity(first_summary),
            read_path_verified=_contains_recall_activity(second_summary),
            recall_top_hit=recall_top_hit,
            first_final_answer=first_summary.final_answers[-1] if first_summary.final_answers else None,
            second_final_answer=second_summary.final_answers[-1] if second_summary.final_answers else None,
        )
    finally:
        if used_temp_repo and temp_dir is not None and not keep_repo:
            try:
                _ensure_clean_uninstall(temp_dir / "repo")
            except SmokeTestError:
                pass
            shutil.rmtree(temp_dir, ignore_errors=True)
