from __future__ import annotations

import json
from pathlib import Path

from agent_memory.config import init_project, load_project, write_linked_project_roots
from agent_memory.integration import (
    INSTRUCTIONS_BEGIN_MARKER,
    INSTRUCTIONS_END_MARKER,
    ensure_local_git_excludes,
    install_claude_hooks,
    install_codex_feature_flag,
    install_codex_hooks,
    install_codex_mcp_server,
    install_codex_project_trust,
    install_mcp_server,
    install_memory_instructions,
    refresh_project_integration,
    remove_local_git_excludes,
    suggest_project_root,
    uninstall_codex_feature_flag,
    uninstall_codex_hooks,
    uninstall_codex_mcp_server,
    uninstall_codex_project_trust,
    uninstall_mcp_server,
    uninstall_memory_instructions,
)


def test_suggest_project_root_prefers_git_ancestor(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    nested = repo / "src" / "feature"
    (repo / ".git").mkdir(parents=True)
    nested.mkdir(parents=True)

    assert suggest_project_root(nested) == repo.resolve()


def test_install_mcp_server_creates_entry(tmp_path: Path) -> None:
    result = install_mcp_server(tmp_path)

    assert result.status == "created"
    payload = json.loads((tmp_path / ".mcp.json").read_text(encoding='utf-8'))
    server = payload["mcpServers"]["agent-memory"]
    assert server["type"] == "stdio"
    assert server["args"][-2:] == ["--cwd", str(tmp_path.resolve())]
    assert server["env"]["AGENT_MEMORY_PROJECT_ROOT"] == str(tmp_path.resolve())


def test_install_mcp_server_preserves_existing_servers(tmp_path: Path) -> None:
    mcp_path = tmp_path / ".mcp.json"
    mcp_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "context7": {
                        "type": "stdio",
                        "command": "npx",
                        "args": ["-y", "@upstash/context7-mcp"],
                    }
                }
            },
            indent=2,
        )
        + "\n"
    , encoding='utf-8')

    install_mcp_server(tmp_path)

    payload = json.loads(mcp_path.read_text(encoding='utf-8'))
    assert "context7" in payload["mcpServers"]
    assert "agent-memory" in payload["mcpServers"]


def test_uninstall_mcp_server_removes_only_agent_memory_entry(tmp_path: Path) -> None:
    mcp_path = tmp_path / ".mcp.json"
    mcp_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "context7": {"type": "stdio", "command": "npx"},
                    "agent-memory": {"type": "stdio", "command": "python3"},
                }
            },
            indent=2,
        )
        + "\n"
    , encoding='utf-8')

    result = uninstall_mcp_server(tmp_path)

    assert result.status == "updated"
    payload = json.loads(mcp_path.read_text(encoding='utf-8'))
    assert "context7" in payload["mcpServers"]
    assert "agent-memory" not in payload["mcpServers"]


def test_ensure_local_git_excludes_creates_and_dedupes(tmp_path: Path) -> None:
    (tmp_path / ".git" / "info").mkdir(parents=True)
    first = ensure_local_git_excludes(tmp_path)
    second = ensure_local_git_excludes(tmp_path)

    assert first.status in {"created", "updated"}
    assert second.status == "unchanged"
    content = (tmp_path / ".git" / "info" / "exclude").read_text(encoding='utf-8')
    assert ".agent-memory/" in content
    assert ".claude/settings.local.json" in content
    assert ".codex/config.toml" in content
    assert ".codex/hooks.json" in content
    assert ".mcp.json" in content


def test_remove_local_git_excludes_removes_only_requested_entries(tmp_path: Path) -> None:
    exclude_path = tmp_path / ".git" / "info" / "exclude"
    exclude_path.parent.mkdir(parents=True)
    exclude_path.write_text(".agent-memory/\n.mcp.json\nother-local-file\n", encoding='utf-8')

    result = remove_local_git_excludes(tmp_path, entries=[".agent-memory/", ".mcp.json"])

    assert result.status == "updated"
    assert exclude_path.read_text(encoding='utf-8') == "other-local-file\n"


def test_install_codex_feature_flag_creates_repo_config(tmp_path: Path) -> None:
    result = install_codex_feature_flag(tmp_path)

    assert result.status == "created"
    config = (tmp_path / ".codex" / "config.toml").read_text(encoding='utf-8')
    assert "[features]" in config
    assert "codex_hooks = true" in config


def test_install_codex_feature_flag_preserves_existing_config(tmp_path: Path) -> None:
    config_path = tmp_path / ".codex" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('[sandbox]\nmode = "workspace-write"\n', encoding='utf-8')

    install_codex_feature_flag(tmp_path)

    content = config_path.read_text(encoding='utf-8')
    assert '[sandbox]' in content
    assert 'mode = "workspace-write"' in content
    assert '[features]' in content
    assert 'codex_hooks = true' in content


def test_install_codex_mcp_server_creates_repo_config(tmp_path: Path) -> None:
    result = install_codex_mcp_server(tmp_path)

    assert result.status == "created"
    config = (tmp_path / ".codex" / "config.toml").read_text(encoding='utf-8')
    assert '[mcp_servers."agent-memory"]' in config
    assert '--cwd' in config
    assert str(tmp_path.resolve()) in config


def test_install_codex_mcp_server_preserves_existing_config(tmp_path: Path) -> None:
    config_path = tmp_path / ".codex" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('[sandbox]\nmode = "workspace-write"\n', encoding='utf-8')

    install_codex_mcp_server(tmp_path)

    content = config_path.read_text(encoding='utf-8')
    assert '[sandbox]' in content
    assert 'mode = "workspace-write"' in content
    assert '[mcp_servers."agent-memory"]' in content
    assert 'AGENT_MEMORY_PROJECT_ROOT' in content


def test_uninstall_codex_mcp_server_preserves_other_config(tmp_path: Path) -> None:
    config_path = tmp_path / ".codex" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        '[features]\n'
        'codex_hooks = true\n\n'
        '[mcp_servers."agent-memory"]\n'
        'command = "/usr/bin/python3"\n'
        'args = ["-m", "agent_memory.cli"]\n\n'
        '[mcp_servers."agent-memory".env]\n'
        'AGENT_MEMORY_PROJECT_ROOT = "/tmp/repo"\n\n'
        '[mcp_servers."context7"]\n'
        'command = "npx"\n'
    , encoding='utf-8')

    result = uninstall_codex_mcp_server(tmp_path)

    assert result.status == "updated"
    content = config_path.read_text(encoding='utf-8')
    assert '[features]' in content
    assert '[mcp_servers."context7"]' in content
    assert '[mcp_servers."agent-memory"]' not in content


def test_install_codex_project_trust_creates_global_config(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    codex_home = tmp_path / "codex-home"

    result = install_codex_project_trust(project_root, codex_home=codex_home)

    assert result.status == "created"
    config = (codex_home / "config.toml").read_text(encoding='utf-8')
    # Reason: paths render as TOML literal strings (single-quoted) so Windows
    # backslashes don't break tomllib parsing.
    assert f"[projects.'{project_root.resolve()}']" in config
    assert 'trust_level = "trusted"' in config


def test_install_codex_project_trust_preserves_existing_project_settings(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    codex_home = tmp_path / "codex-home"
    config_path = codex_home / "config.toml"
    codex_home.mkdir()
    # Reason: literal-string keys so fixture parses on Windows (see above).
    config_path.write_text(
        '[sandbox]\n'
        'mode = "workspace-write"\n\n'
        f"[projects.'{project_root.resolve()}']\n"
        'approval_policy = "never"\n'
        'trust_level = "untrusted"\n'
    , encoding='utf-8')

    install_codex_project_trust(project_root, codex_home=codex_home)

    content = config_path.read_text(encoding='utf-8')
    assert '[sandbox]' in content
    assert 'mode = "workspace-write"' in content
    assert 'approval_policy = "never"' in content
    assert 'trust_level = "trusted"' in content


def test_uninstall_codex_project_trust_preserves_other_project_settings(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    codex_home = tmp_path / "codex-home"
    config_path = codex_home / "config.toml"
    codex_home.mkdir()
    # Reason: literal-string keys so fixture parses on Windows (see above).
    config_path.write_text(
        f"[projects.'{project_root.resolve()}']\n"
        'approval_policy = "never"\n'
        'trust_level = "trusted"\n'
    , encoding='utf-8')

    result = uninstall_codex_project_trust(project_root, codex_home=codex_home)

    assert result.status == "updated"
    content = config_path.read_text(encoding='utf-8')
    assert 'approval_policy = "never"' in content
    assert 'trust_level = "trusted"' not in content


def test_install_codex_hooks_creates_hooks_json(tmp_path: Path) -> None:
    result = install_codex_hooks(tmp_path)

    assert result.status == "created"
    payload = json.loads((tmp_path / ".codex" / "hooks.json").read_text(encoding='utf-8'))
    assert "UserPromptSubmit" in payload["hooks"]
    user_prompt_hook = payload["hooks"]["UserPromptSubmit"][0]["hooks"][0]
    # Portable form: dispatches via the on-PATH `agent-memory` binary, no
    # absolute python interpreter path. Also includes a PATH=$HOME/.local/bin
    # prefix so /bin/sh -c hook subprocesses can find the binary even though
    # /bin/sh has a stripped-down PATH that does not include ~/.local/bin.
    assert "agent-memory _hook codex-user-prompt-submit" in user_prompt_hook["command"]
    assert "AGENT_MEMORY_PROJECT_ROOT=" in user_prompt_hook["command"]
    assert "PATH=$HOME/.local/bin:$PATH" in user_prompt_hook["command"]
    assert "/python3" not in user_prompt_hook["command"]


def test_install_codex_hooks_preserves_existing_hooks(tmp_path: Path) -> None:
    hooks_path = tmp_path / ".codex" / "hooks.json"
    hooks_path.parent.mkdir(parents=True, exist_ok=True)
    hooks_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [{"type": "command", "command": "echo pre"}],
                        }
                    ]
                }
            },
            indent=2,
        )
        + "\n"
    , encoding='utf-8')

    install_codex_hooks(tmp_path)

    payload = json.loads(hooks_path.read_text(encoding='utf-8'))
    assert "PreToolUse" in payload["hooks"]
    assert "UserPromptSubmit" in payload["hooks"]


def test_uninstall_codex_hooks_preserves_other_hooks(tmp_path: Path) -> None:
    hooks_path = tmp_path / ".codex" / "hooks.json"
    hooks_path.parent.mkdir(parents=True, exist_ok=True)
    hooks_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo pre"}]}
                    ],
                    "UserPromptSubmit": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "python -m agent_memory.hooks.codex_user_prompt_submit",
                                }
                            ]
                        }
                    ],
                }
            },
            indent=2,
        )
        + "\n"
    , encoding='utf-8')

    result = uninstall_codex_hooks(tmp_path)

    assert result.status == "updated"
    payload = json.loads(hooks_path.read_text(encoding='utf-8'))
    assert "PreToolUse" in payload["hooks"]
    assert "UserPromptSubmit" not in payload["hooks"]


def test_uninstall_codex_feature_flag_leaves_other_hooked_projects_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / ".codex" / "config.toml"
    hooks_path = tmp_path / ".codex" / "hooks.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('[features]\ncodex_hooks = true\n', encoding='utf-8')
    hooks_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo pre"}]}
                    ]
                }
            },
            indent=2,
        )
        + "\n"
    , encoding='utf-8')

    result = uninstall_codex_feature_flag(tmp_path)

    assert result.status == "unchanged"
    assert 'codex_hooks = true' in config_path.read_text(encoding='utf-8')


def test_install_claude_hooks_creates_settings(tmp_path: Path) -> None:
    result = install_claude_hooks(tmp_path)

    assert result.status == "created"
    payload = json.loads((tmp_path / ".claude" / "settings.local.json").read_text(encoding='utf-8'))
    # Default install is CLI-only — we must NOT touch enabledMcpjsonServers.
    assert "enabledMcpjsonServers" not in payload
    assert "UserPromptSubmit" in payload["hooks"]
    user_prompt_hook = payload["hooks"]["UserPromptSubmit"][0]["hooks"][0]
    # Portable form: dispatches via the on-PATH `agent-memory` binary, no
    # absolute python interpreter path. Also includes a PATH=$HOME/.local/bin
    # prefix so /bin/sh -c hook subprocesses can find the binary.
    assert "agent-memory _hook claude-user-prompt-submit" in user_prompt_hook["command"]
    assert "AGENT_MEMORY_PROJECT_ROOT=" in user_prompt_hook["command"]
    assert "PATH=$HOME/.local/bin:$PATH" in user_prompt_hook["command"]
    assert "/python3" not in user_prompt_hook["command"]


def test_install_claude_hooks_with_mcp_registers_server(tmp_path: Path) -> None:
    install_claude_hooks(tmp_path, register_mcp_server=True)

    payload = json.loads((tmp_path / ".claude" / "settings.local.json").read_text(encoding='utf-8'))
    assert "agent-memory" in payload["enabledMcpjsonServers"]
    assert "UserPromptSubmit" in payload["hooks"]


def test_install_claude_hooks_preserves_existing_settings(tmp_path: Path) -> None:
    settings_path = tmp_path / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(
            {
                "enabledMcpjsonServers": ["context7"],
                "permissions": {"allow": ["Bash"]},
            },
            indent=2,
        )
        + "\n"
    , encoding='utf-8')

    install_claude_hooks(tmp_path)

    payload = json.loads(settings_path.read_text(encoding='utf-8'))
    assert payload["permissions"]["allow"] == ["Bash"]
    # Default install does not append agent-memory; the user's existing server list is untouched.
    assert payload["enabledMcpjsonServers"] == ["context7"]
    assert "agent-memory" not in payload["enabledMcpjsonServers"]


def test_install_claude_hooks_with_mcp_preserves_existing_settings(tmp_path: Path) -> None:
    settings_path = tmp_path / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(
            {
                "enabledMcpjsonServers": ["context7"],
                "permissions": {"allow": ["Bash"]},
            },
            indent=2,
        )
        + "\n"
    , encoding='utf-8')

    install_claude_hooks(tmp_path, register_mcp_server=True)

    payload = json.loads(settings_path.read_text(encoding='utf-8'))
    assert payload["permissions"]["allow"] == ["Bash"]
    assert "context7" in payload["enabledMcpjsonServers"]
    assert "agent-memory" in payload["enabledMcpjsonServers"]


def test_install_memory_instructions_skips_when_files_missing(tmp_path: Path) -> None:
    results = install_memory_instructions(tmp_path)

    assert {r.path.name for r in results} == {"CLAUDE.md", "AGENTS.md"}
    assert all(r.status == "skipped" for r in results)
    assert not (tmp_path / "CLAUDE.md").exists()
    assert not (tmp_path / "AGENTS.md").exists()


def test_install_memory_instructions_injects_after_h1(tmp_path: Path) -> None:
    claude_md = tmp_path / "CLAUDE.md"
    agents_md = tmp_path / "AGENTS.md"
    claude_md.write_text("# Project\n\n## Existing\n\nstuff\n", encoding='utf-8')
    agents_md.write_text("# Project\n\n## Existing\n\nstuff\n", encoding='utf-8')

    results = install_memory_instructions(tmp_path)

    statuses = {r.path.name: r.status for r in results}
    assert statuses == {"CLAUDE.md": "created", "AGENTS.md": "created"}

    for path in (claude_md, agents_md):
        text = path.read_text(encoding='utf-8')
        assert INSTRUCTIONS_BEGIN_MARKER in text
        assert INSTRUCTIONS_END_MARKER in text
        assert "Agent Memory" in text
        assert "agent-memory recall" in text
        assert "agent-memory save" in text
        # CLI-only — must NOT push the agent toward MCP tool calls.
        assert "save_memory" not in text
        assert "recall_memories" not in text
        # Block lives between the H1 and the existing "## Existing" section.
        h1_pos = text.find("# Project")
        marker_pos = text.find(INSTRUCTIONS_BEGIN_MARKER)
        existing_pos = text.find("## Existing")
        assert h1_pos < marker_pos < existing_pos
        # Original content preserved.
        assert "stuff" in text


def test_install_memory_instructions_is_idempotent(tmp_path: Path) -> None:
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("# Project\n\nfoo\n", encoding='utf-8')

    install_memory_instructions(tmp_path)
    text_after_first = claude_md.read_text(encoding='utf-8')
    second = install_memory_instructions(tmp_path)
    text_after_second = claude_md.read_text(encoding='utf-8')

    assert text_after_first == text_after_second
    statuses = {r.path.name: r.status for r in second}
    assert statuses["CLAUDE.md"] == "unchanged"
    # Exactly one block, not two.
    assert text_after_second.count(INSTRUCTIONS_BEGIN_MARKER) == 1
    assert text_after_second.count(INSTRUCTIONS_END_MARKER) == 1


def test_install_memory_instructions_replaces_stale_block(tmp_path: Path) -> None:
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(
        f"# Project\n\n{INSTRUCTIONS_BEGIN_MARKER}\nold stale text\n{INSTRUCTIONS_END_MARKER}\n\n## After\nbody\n"
    , encoding='utf-8')

    results = install_memory_instructions(tmp_path)

    text = claude_md.read_text(encoding='utf-8')
    assert "old stale text" not in text
    assert "agent-memory recall" in text
    assert "## After" in text
    assert "body" in text
    statuses = {r.path.name: r.status for r in results}
    assert statuses["CLAUDE.md"] == "updated"


def test_uninstall_memory_instructions_removes_only_marker_block(tmp_path: Path) -> None:
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("# Project\n\nfoo\n\n## After\nbody\n", encoding='utf-8')
    install_memory_instructions(tmp_path)
    assert INSTRUCTIONS_BEGIN_MARKER in claude_md.read_text(encoding='utf-8')

    uninstall_memory_instructions(tmp_path)

    text = claude_md.read_text(encoding='utf-8')
    assert INSTRUCTIONS_BEGIN_MARKER not in text
    assert INSTRUCTIONS_END_MARKER not in text
    assert "# Project" in text
    assert "## After" in text
    assert "body" in text
    assert "foo" in text


def test_uninstall_memory_instructions_no_op_when_block_absent(tmp_path: Path) -> None:
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("# Project\n\nfoo\n", encoding='utf-8')

    results = uninstall_memory_instructions(tmp_path)

    statuses = {r.path.name: r.status for r in results}
    assert statuses["CLAUDE.md"] == "unchanged"
    assert claude_md.read_text(encoding='utf-8') == "# Project\n\nfoo\n"


def test_install_codex_hooks_can_target_shared_parent_store(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)

    result = install_codex_hooks(child, memory_project_root=parent)

    assert result.status == "created"
    payload = json.loads((child / ".codex" / "hooks.json").read_text(encoding="utf-8"))
    command = payload["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
    assert str(parent.resolve()) in command
    assert str(child.resolve()) not in command


def test_install_memory_instructions_describes_shared_store(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    agents_md = child / "AGENTS.md"
    agents_md.write_text("# Child Repo\n", encoding="utf-8")

    results = install_memory_instructions(child, memory_project_root=parent)

    assert {result.path.name: result.status for result in results}["AGENTS.md"] == "created"
    text = agents_md.read_text(encoding="utf-8")
    assert str((parent / ".agent-memory").resolve()) in text
    assert "agent-memory link-root" in text


def test_refresh_project_integration_refreshes_linked_roots(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    init_project(parent)
    project = load_project(parent, exact=True)
    project.config.integration_version = "v0.0.1"
    project.config_path.write_text(json.dumps(project.config.to_dict(), indent=2) + "\n", encoding="utf-8")
    write_linked_project_roots(parent, [str(child.resolve())])

    install_codex_hooks(child, memory_project_root=child)
    (child / "AGENTS.md").write_text("# Child Repo\n", encoding="utf-8")
    install_memory_instructions(child, memory_project_root=child)

    payload = refresh_project_integration(project, current_version="v0.2.10", force=True)

    command = json.loads((child / ".codex" / "hooks.json").read_text(encoding="utf-8"))["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
    assert str(parent.resolve()) in command
    assert str(child.resolve()) not in command
    assert str((parent / ".agent-memory").resolve()) in (child / "AGENTS.md").read_text(encoding="utf-8")
    assert payload["refreshed_roots"] == [str(child.resolve())]
