from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_memory.cli import _codex_feature_state, _doctor_payload, app
from agent_memory.config import MemoryConfig, init_project
from agent_memory.engine import open_memory_with_retry
from agent_memory.integration import IntegrationResult


def test_codex_feature_state_parses_false(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("agent_memory.cli.shutil.which", lambda name: "/usr/local/bin/codex" if name == "codex" else None)

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="codex_hooks                      under development  false\n",
            stderr="",
        )

    monkeypatch.setattr("agent_memory.cli.subprocess.run", fake_run)

    state, error = _codex_feature_state(tmp_path)

    assert state is False
    assert error is None


def test_doctor_payload_reports_codex_exec_warning(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".agent-memory").mkdir()
    (tmp_path / ".agent-memory" / "config.json").write_text("{}\n", encoding='utf-8')
    (tmp_path / ".agent-memory" / "instructions.md").write_text("instructions\n", encoding='utf-8')
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".codex" / "config.toml").write_text(
        '[features]\n'
        'codex_hooks = true\n\n'
        '[mcp_servers."agent-memory"]\n'
        'command = "/usr/bin/python3"\n'
        'args = ["-m", "agent_memory.cli", "serve-mcp", "--cwd", "/tmp/repo"]\n'
    , encoding='utf-8')
    (tmp_path / ".codex" / "hooks.json").write_text('{"hooks": {}}\n', encoding='utf-8')
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.local.json").write_text('{"hooks": {}}\n', encoding='utf-8')
    (tmp_path / ".mcp.json").write_text('{"mcpServers": {}}\n', encoding='utf-8')

    def fake_which(name: str) -> str | None:
        if name == "codex":
            return "/Applications/Codex.app/Contents/Resources/codex"
        if name == "agent-memory":
            return "/Users/test/.local/bin/agent-memory"
        return None

    monkeypatch.setattr("agent_memory.cli.shutil.which", fake_which)
    monkeypatch.setattr("agent_memory.cli._codex_feature_state", lambda root: (False, None))
    monkeypatch.setattr("agent_memory.cli.codex_project_trust_state", lambda root: (False, None))

    payload = _doctor_payload(tmp_path)

    assert payload["project_root"] == str(tmp_path.resolve())
    assert payload["codex_hooks_effective"] is False
    assert payload["codex_mcp_server"] is True
    assert payload["codex_project_trusted"] is False
    warnings = payload["warnings"]
    assert isinstance(warnings, list)
    assert any("fresh interactive Codex session" in warning for warning in warnings)
    assert any("codex exec" in warning for warning in warnings)
    assert any("codex_hooks" in warning for warning in warnings)
    assert any("trusted in `~/.codex/config.toml`" in warning for warning in warnings)


def test_doctor_payload_warns_when_codex_not_on_path(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".agent-memory").mkdir()
    (tmp_path / ".agent-memory" / "config.json").write_text("{}\n", encoding='utf-8')
    (tmp_path / ".agent-memory" / "instructions.md").write_text("instructions\n", encoding='utf-8')
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".codex" / "config.toml").write_text('[features]\ncodex_hooks = true\n', encoding='utf-8')
    (tmp_path / ".codex" / "hooks.json").write_text('{"hooks": {}}\n', encoding='utf-8')

    monkeypatch.setattr("agent_memory.cli.shutil.which", lambda name: "/Users/test/.local/bin/agent-memory" if name == "agent-memory" else None)
    monkeypatch.setattr("agent_memory.cli.codex_project_trust_state", lambda root: (False, None))

    payload = _doctor_payload(tmp_path)

    assert payload["codex_hooks_effective"] is None
    warnings = payload["warnings"]
    assert isinstance(warnings, list)
    assert any("Codex CLI not found on PATH" in warning for warning in warnings)


def test_init_reports_codex_trust_install(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    seen: dict[str, Path] = {}

    def fake_install_codex_project_trust(project_root: Path) -> IntegrationResult:
        seen["project_root"] = project_root
        return IntegrationResult(
            path=Path("/tmp/fake-codex-config.toml"),
            status="created",
            details="Installed trust entry.",
        )

    monkeypatch.setattr("agent_memory.cli.install_codex_project_trust", fake_install_codex_project_trust)

    result = runner.invoke(
        app,
        [
            "init",
            str(tmp_path),
            "--embedding-backend",
            "hash",
            "--without-mcp",
            "--no-install-local-excludes",
            "--no-install-claude-hooks",
        ],
    )

    assert result.exit_code == 0
    assert seen["project_root"] == tmp_path.resolve()
    assert "Codex trust: Installed trust entry." in result.stdout


def test_uninstall_keeps_store_by_default(tmp_path: Path) -> None:
    runner = CliRunner()
    (tmp_path / ".git").mkdir()

    init_result = runner.invoke(
        app,
        [
            "init",
            str(tmp_path),
            "--embedding-backend",
            "hash",
            "--no-install-codex-trust",
        ],
    )
    assert init_result.exit_code == 0

    uninstall_result = runner.invoke(
        app,
        [
            "uninstall",
            str(tmp_path),
            "--keep-codex-trust",
        ],
    )

    assert uninstall_result.exit_code == 0
    assert (tmp_path / ".agent-memory").exists()
    assert not (tmp_path / ".mcp.json").exists()
    assert not (tmp_path / ".codex" / "hooks.json").exists()
    assert not (tmp_path / ".codex" / "config.toml").exists()
    assert not (tmp_path / ".claude" / "settings.local.json").exists()
    exclude_text = (tmp_path / ".git" / "info" / "exclude").read_text(encoding='utf-8')
    assert ".agent-memory/" in exclude_text
    assert ".mcp.json" not in exclude_text


def test_uninstall_remove_store_deletes_project_memory_data(tmp_path: Path) -> None:
    runner = CliRunner()
    (tmp_path / ".git").mkdir()

    init_result = runner.invoke(
        app,
        [
            "init",
            str(tmp_path),
            "--embedding-backend",
            "hash",
            "--no-install-codex-trust",
        ],
    )
    assert init_result.exit_code == 0

    uninstall_result = runner.invoke(
        app,
        [
            "uninstall",
            str(tmp_path),
            "--remove-store",
            "--keep-codex-trust",
            "--json",
        ],
    )

    assert uninstall_result.exit_code == 0
    payload = json.loads(uninstall_result.stdout)
    assert payload["remove_store"] is True
    assert payload["store"]["status"] == "removed"
    assert not (tmp_path / ".agent-memory").exists()
    assert not (tmp_path / ".mcp.json").exists()
    assert not (tmp_path / ".codex").exists()
    assert not (tmp_path / ".claude").exists()
    exclude_text = (tmp_path / ".git" / "info" / "exclude").read_text(encoding='utf-8')
    assert ".agent-memory/" not in exclude_text


def test_uninstall_all_removes_project_and_machine_artifacts(monkeypatch, tmp_path: Path) -> None:
    if sys.platform.startswith("win"):
        pytest.skip("clean-room uninstall test exercises POSIX shell rc cleanup")

    runner = CliRunner()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
    monkeypatch.setattr("agent_memory.cli.shutil.which", lambda name: None)
    monkeypatch.setattr(
        "agent_memory.upgrade._resolve_running_binary_path",
        lambda: home / ".local" / "share" / "agent-memory" / "v0.2.8" / "agent-memory",
    )

    project_root = tmp_path / "repo"
    (project_root / ".git").mkdir(parents=True)
    init_result = runner.invoke(
        app,
        [
            "init",
            str(project_root),
            "--embedding-backend",
            "hash",
            "--no-install-codex-trust",
        ],
    )
    assert init_result.exit_code == 0

    standalone_binary = home / ".local" / "bin" / "agent-memory"
    standalone_binary.parent.mkdir(parents=True)
    standalone_binary.write_text("binary\n", encoding="utf-8")
    libexec_binary = home / ".local" / "share" / "agent-memory" / "v0.2.8" / "agent-memory"
    libexec_binary.parent.mkdir(parents=True)
    libexec_binary.write_text("bundle\n", encoding="utf-8")
    cache_path = home / ".cache" / "agent-memory" / "update-check.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{}", encoding="utf-8")

    zshrc = home / ".zshrc"
    zshrc.write_text(
        "# before\n"
        "# added by agent-memory installer\n"
        f'export PATH="{home / ".local/bin"}:$PATH"\n'
        "# after\n",
        encoding="utf-8",
    )

    claude_settings = home / ".claude" / "settings.json"
    claude_settings.parent.mkdir(parents=True)
    claude_settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "# AGENT_MEMORY_INSTALLER_PATH_HOOK_v1\nprintf 'export PATH=\"%s:$PATH\"\\n' \"/tmp/bin\" >> \"$CLAUDE_ENV_FILE\"",
                                },
                                {
                                    "type": "command",
                                    "command": "echo keep-me",
                                },
                            ]
                        }
                    ]
                },
                "enabledPlugins": {
                    "agent-memory@agent-memory-plugins": True,
                    "other@market": True,
                },
                "extraKnownMarketplaces": {
                    "agent-memory-plugins": {"source": {"source": "github", "repo": "ben1787/agent-memory"}},
                    "other": {"source": {"source": "github", "repo": "foo/bar"}},
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    known_marketplaces = home / ".claude" / "plugins" / "known_marketplaces.json"
    known_marketplaces.parent.mkdir(parents=True)
    known_marketplaces.write_text(
        json.dumps(
            {
                "agent-memory-plugins": {
                    "source": {"source": "github", "repo": "ben1787/agent-memory"},
                    "installLocation": str(home / ".claude" / "plugins" / "marketplaces" / "agent-memory-plugins"),
                },
                "other": {"source": {"source": "github", "repo": "foo/bar"}},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    installed_plugins = home / ".claude" / "plugins" / "installed_plugins.json"
    installed_plugins.write_text(
        json.dumps(
            {
                "version": 2,
                "plugins": {
                    "agent-memory@agent-memory-plugins": [{"scope": "user"}],
                    "other@market": [{"scope": "user"}],
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    plugin_cache = home / ".claude" / "plugins" / "cache" / "agent-memory-plugins"
    plugin_cache.mkdir(parents=True)
    (plugin_cache / "cached.txt").write_text("cache\n", encoding="utf-8")
    plugin_data = home / ".claude" / "plugins" / "data" / "agent-memory-agent-memory-plugins"
    plugin_data.mkdir(parents=True)
    (plugin_data / "managed.txt").write_text("data\n", encoding="utf-8")
    marketplace_clone = home / ".claude" / "plugins" / "marketplaces" / "agent-memory-plugins"
    marketplace_clone.mkdir(parents=True)
    (marketplace_clone / "marketplace.json").write_text("{}\n", encoding="utf-8")

    result = runner.invoke(app, ["uninstall-all", str(project_root), "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["project"]["project_root"] == str(project_root.resolve())
    assert payload["project"]["store"]["status"] == "removed"
    assert not (project_root / ".agent-memory").exists()
    assert not standalone_binary.exists()
    assert not (home / ".local" / "share" / "agent-memory").exists()
    assert not cache_path.parent.exists()
    assert not plugin_cache.exists()
    assert not plugin_data.exists()
    assert not marketplace_clone.exists()

    settings_payload = json.loads(claude_settings.read_text(encoding="utf-8"))
    session_start = settings_payload["hooks"]["SessionStart"][0]["hooks"]
    assert len(session_start) == 1
    assert session_start[0]["command"] == "echo keep-me"
    assert "agent-memory@agent-memory-plugins" not in settings_payload["enabledPlugins"]
    assert "agent-memory-plugins" not in settings_payload["extraKnownMarketplaces"]

    known_marketplaces_payload = json.loads(known_marketplaces.read_text(encoding="utf-8"))
    assert "agent-memory-plugins" not in known_marketplaces_payload
    installed_plugins_payload = json.loads(installed_plugins.read_text(encoding="utf-8"))
    assert "agent-memory@agent-memory-plugins" not in installed_plugins_payload["plugins"]

    zshrc_text = zshrc.read_text(encoding="utf-8")
    assert "# added by agent-memory installer" not in zshrc_text
    assert f'export PATH="{home / ".local/bin"}:$PATH"' not in zshrc_text


def test_save_command_persists_memory(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    runner = CliRunner()

    result = runner.invoke(app, ["save", "--cwd", str(tmp_path), "--json", "CLI save works"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["total_memories"] == 1
    memory = open_memory_with_retry(tmp_path, exact=True, read_only=True)
    try:
        recall = memory.recall("CLI save works", limit=5).to_dict()
    finally:
        memory.close()
    assert recall["nodes"][0]["text"] == "CLI save works"


def test_save_command_reads_stdin(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    runner = CliRunner()
    body = 'multi-line memory\nwith "quotes" and `backticks`\nand a $literal'

    result = runner.invoke(
        app,
        ["save", "--cwd", str(tmp_path), "--json", "--stdin"],
        input=body,
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["total_memories"] == 1
    memory = open_memory_with_retry(tmp_path, exact=True, read_only=True)
    try:
        recall = memory.recall("multi-line memory quotes backticks", limit=5).to_dict()
    finally:
        memory.close()
    assert any(body == node["text"] for node in recall["nodes"])


def test_save_command_stdin_rejects_combined_args(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["save", "--cwd", str(tmp_path), "--stdin", "extra"],
        input="body",
    )

    assert result.exit_code != 0
    assert "stdin" in result.stdout.lower() or "stdin" in (result.stderr or "").lower()


def test_save_command_requires_text_or_stdin(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    runner = CliRunner()

    result = runner.invoke(app, ["save", "--cwd", str(tmp_path)])

    assert result.exit_code != 0


def test_import_repo_command_bootstraps_project_corpus(tmp_path: Path) -> None:
    init_project(
        tmp_path,
        config=MemoryConfig(
            embedding_backend="hash",
            embedding_dimensions=64,
            max_memory_words=80,
        ),
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text(
        "# Guide\n\nThis importer creates memories from repo files.\n",
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "def main():\n    return 'ok'\n",
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "import-repo",
            "--cwd",
            str(tmp_path),
            "--json",
            "--max-memories",
            "5",
            "--max-file-kb",
            "64",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["imported_memories"] >= 2
    assert payload["total_memories"] == payload["imported_memories"]


def test_list_command_returns_recent_memories_newest_first(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    memory = open_memory_with_retry(tmp_path, exact=True)
    try:
        memory.save("first memory")
        memory.save("second memory")
        memory.save("third memory")
    finally:
        memory.close()

    runner = CliRunner()
    result = runner.invoke(app, ["list", "--cwd", str(tmp_path), "--recent", "2", "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["total_memories"] == 3
    assert payload["shown"] == 2
    texts = [m["text"] for m in payload["memories"]]
    assert texts[0] == "third memory"
    assert texts[1] == "second memory"


def test_show_command_returns_full_memory(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    memory = open_memory_with_retry(tmp_path, exact=True)
    try:
        result = memory.save("a memory worth showing")
        mem_id = result.saved[0].memory_id
    finally:
        memory.close()

    runner = CliRunner()
    show = runner.invoke(app, ["show", mem_id, "--cwd", str(tmp_path), "--json"])

    assert show.exit_code == 0
    payload = json.loads(show.stdout)
    assert payload["memory_id"] == mem_id
    assert payload["text"] == "a memory worth showing"


def test_show_command_missing_id_exits_nonzero(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    runner = CliRunner()
    result = runner.invoke(app, ["show", "mem_does_not_exist", "--cwd", str(tmp_path)])
    assert result.exit_code != 0


def test_edit_command_one_shot_updates_text(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    memory = open_memory_with_retry(tmp_path, exact=True)
    try:
        result = memory.save("original text")
        mem_id = result.saved[0].memory_id
    finally:
        memory.close()

    runner = CliRunner()
    edit = runner.invoke(
        app,
        ["edit", mem_id, "corrected text", "--cwd", str(tmp_path), "--json"],
    )

    assert edit.exit_code == 0, edit.stdout
    payload = json.loads(edit.stdout)
    assert payload["text"] == "corrected text"


def test_edit_command_stdin_handles_special_chars(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    memory = open_memory_with_retry(tmp_path, exact=True)
    try:
        result = memory.save("placeholder")
        mem_id = result.saved[0].memory_id
    finally:
        memory.close()

    body = 'multi-line edit\nwith "quotes" and `backticks`\nand a $literal'
    runner = CliRunner()
    edit = runner.invoke(
        app,
        ["edit", mem_id, "--stdin", "--cwd", str(tmp_path), "--json"],
        input=body,
    )

    assert edit.exit_code == 0, edit.stdout
    payload = json.loads(edit.stdout)
    assert payload["text"] == body


def test_edit_command_stdin_rejects_combined_text(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    memory = open_memory_with_retry(tmp_path, exact=True)
    try:
        result = memory.save("placeholder")
        mem_id = result.saved[0].memory_id
    finally:
        memory.close()

    runner = CliRunner()
    edit = runner.invoke(
        app,
        ["edit", mem_id, "extra", "--stdin", "--cwd", str(tmp_path)],
        input="body",
    )
    assert edit.exit_code != 0


def test_delete_command_removes_memory(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    memory = open_memory_with_retry(tmp_path, exact=True)
    try:
        result = memory.save("doomed memory")
        mem_id = result.saved[0].memory_id
    finally:
        memory.close()

    runner = CliRunner()
    delete = runner.invoke(
        app,
        ["delete", mem_id, "--yes", "--cwd", str(tmp_path), "--json"],
    )

    assert delete.exit_code == 0, delete.stdout
    payload = json.loads(delete.stdout)
    assert payload["deleted"]["memory_id"] == mem_id
    assert payload["total_memories"] == 0


def test_delete_command_requires_yes_for_nonzero_confirmation(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    memory = open_memory_with_retry(tmp_path, exact=True)
    try:
        result = memory.save("careful memory")
        mem_id = result.saved[0].memory_id
    finally:
        memory.close()

    runner = CliRunner()
    # Without --yes and answering "n" to the prompt, should abort.
    delete = runner.invoke(
        app,
        ["delete", mem_id, "--cwd", str(tmp_path)],
        input="n\n",
    )
    assert delete.exit_code != 0
    # Memory should still be there.
    memory = open_memory_with_retry(tmp_path, exact=True, read_only=True)
    try:
        assert memory.get(mem_id) is not None
    finally:
        memory.close()


def test_consolidation_status_and_complete_command(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    runner = CliRunner()

    initial = runner.invoke(app, ["consolidation-status", "--cwd", str(tmp_path), "--json"])
    assert initial.exit_code == 0, initial.stdout
    initial_payload = json.loads(initial.stdout)
    assert initial_payload["last_consolidation_date"] is None
    assert initial_payload["is_due_today"] is True

    completed = runner.invoke(app, ["consolidation-complete", "--cwd", str(tmp_path), "--json"])
    assert completed.exit_code == 0, completed.stdout
    completed_payload = json.loads(completed.stdout)
    assert completed_payload["status"] == "completed"
    assert completed_payload["is_completed_today"] is True
    assert completed_payload["is_due_today"] is False
    assert completed_payload["last_consolidation_date"] == completed_payload["today"]


def test_undo_command_reverts_save(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    memory = open_memory_with_retry(tmp_path, exact=True)
    try:
        result = memory.save("an undoable save")
        mem_id = result.saved[0].memory_id
    finally:
        memory.close()

    runner = CliRunner()
    undo = runner.invoke(app, ["undo", "--cwd", str(tmp_path), "--json"])

    assert undo.exit_code == 0, undo.stdout
    payload = json.loads(undo.stdout)
    assert payload["reverted"] == "save"
    assert payload["memory_id"] == mem_id
    assert payload["total_memories"] == 0


def test_undo_command_reverts_delete(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    memory = open_memory_with_retry(tmp_path, exact=True)
    try:
        result = memory.save("about to be killed")
        mem_id = result.saved[0].memory_id
        memory.delete(mem_id)
        assert memory.get(mem_id) is None
    finally:
        memory.close()

    runner = CliRunner()
    undo = runner.invoke(app, ["undo", "--cwd", str(tmp_path), "--json"])

    assert undo.exit_code == 0, undo.stdout
    payload = json.loads(undo.stdout)
    assert payload["reverted"] == "delete"
    assert payload["memory_id"] == mem_id

    memory = open_memory_with_retry(tmp_path, exact=True, read_only=True)
    try:
        restored = memory.get(mem_id)
        assert restored is not None
        assert restored.text == "about to be killed"
    finally:
        memory.close()


def test_undo_command_with_empty_log_exits_nonzero(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    runner = CliRunner()
    undo = runner.invoke(app, ["undo", "--cwd", str(tmp_path)])
    assert undo.exit_code != 0


def test_reembed_command_rebuilds_store_with_current_config(tmp_path: Path) -> None:
    init_project(
        tmp_path,
        config=MemoryConfig(
            version=4,
            embedding_backend="hash",
            embedding_model="hash-legacy",
            embedding_dimensions=2,
            max_memory_words=1000,
            stored_embedding_backend="hash",
            stored_embedding_model="hash-legacy",
            stored_embedding_dimensions=2,
        ),
    )
    memory = open_memory_with_retry(tmp_path, exact=True)
    try:
        memory.save("alpha memory")
    finally:
        memory.close()

    updated_config = MemoryConfig(
        embedding_backend="hash",
        embedding_model="hash-v2",
        embedding_dimensions=8,
        stored_embedding_backend="hash",
        stored_embedding_model="hash-legacy",
        stored_embedding_dimensions=2,
        max_memory_words=250,
    )
    (tmp_path / ".agent-memory" / "config.json").write_text(
        json.dumps(updated_config.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(app, ["reembed", "--cwd", str(tmp_path), "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["reembedded"] is True
    assert payload["memory_count"] == 1
    assert payload["current_store"]["embedding_dimensions"] == 8

    reopened = open_memory_with_retry(tmp_path, exact=True)
    try:
        records = reopened.list_all()
        assert len(records) == 1
        assert len(records[0].embedding) == 8
        assert reopened.project.config.stored_embedding_dimensions == 8
    finally:
        reopened.close()


def test_prune_model_cache_command_removes_stale_fastembed_models(monkeypatch, tmp_path: Path) -> None:
    init_project(
        tmp_path,
        config=MemoryConfig(
            embedding_backend="fastembed",
            embedding_model="snowflake/snowflake-arctic-embed-m",
            embedding_dimensions=768,
            stored_embedding_backend="fastembed",
            stored_embedding_model="snowflake/snowflake-arctic-embed-m",
            stored_embedding_dimensions=768,
            max_memory_words=250,
        ),
    )

    cache_root = tmp_path / "fastembed-cache"
    keep_dir = cache_root / "models--Snowflake--snowflake-arctic-embed-m"
    stale_dir = cache_root / "models--qdrant--bge-small-en-v1.5-onnx-q"
    keep_dir.mkdir(parents=True)
    stale_dir.mkdir(parents=True)
    (keep_dir / "weights.bin").write_bytes(b"keep")
    (stale_dir / "weights.bin").write_bytes(b"stale-cache")
    stale_lock = cache_root / ".locks" / stale_dir.name
    stale_lock.mkdir(parents=True)

    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(cache_root))

    runner = CliRunner()
    result = runner.invoke(app, ["prune-model-cache", "--cwd", str(tmp_path), "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["kept_models"] == ["snowflake/snowflake-arctic-embed-m"]
    assert len(payload["pruned"]) == 1
    assert payload["pruned"][0]["model_name"] == "BAAI/bge-small-en-v1.5"
    assert payload["freed_bytes"] > 0
    assert keep_dir.exists()
    assert not stale_dir.exists()
    assert not stale_lock.exists()


def test_recall_command_accepts_unquoted_multi_word_query(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    memory = open_memory_with_retry(tmp_path, exact=True)
    try:
        memory.save("The billing webhook handler lives in services/billing/webhooks.py.")
    finally:
        memory.close()

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["recall", "--cwd", str(tmp_path), "--json", "billing", "webhook", "handler"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["query"] == "billing webhook handler"
    assert payload["nodes"][0]["source"] == "QUERY"
    assert payload["nodes"][0]["text"] == "The billing webhook handler lives in services/billing/webhooks.py."
