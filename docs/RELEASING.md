# Releasing agent-memory

This file documents how to cut a release. The `release` workflow at
`.github/workflows/release.yml` does everything automatically once you push
a `v*` tag — no manual steps.

## Cutting a release (the normal flow)

```bash
# Bump the version in pyproject.toml and src/agent_memory/__init__.py first.
# Both must agree.
git add pyproject.toml src/agent_memory/__init__.py
git commit -m "Bump to vX.Y.Z"

# Tag and push.
git tag vX.Y.Z
git push origin main vX.Y.Z
```

GitHub Actions takes over from there:

1. **Build matrix** runs on `macos-14` (arm64), `macos-13` (x86_64),
   `ubuntu-22.04` (x86_64), `ubuntu-22.04-arm` (arm64), and `windows-2022`
   (x86_64). Each runner builds a PyInstaller onedir bundle using
   `pyinstaller pyinstaller/agent-memory.spec`, tars it (or zips on Windows),
   and produces a sha256 file.

2. **Release publish** uploads all five archives plus their `.sha256` files
   to a new GitHub Release at `vX.Y.Z`. This is the canonical source of
   truth — `install.sh` and `agent-memory upgrade` both download from these
   release assets.

You can watch the workflow at:
https://github.com/ben1787/agent-memory/actions

## Verifying the install path after a release

```bash
# Curl install.sh from the freshly-tagged release.
curl -LsSf https://raw.githubusercontent.com/ben1787/agent-memory/vX.Y.Z/install.sh | sh

# OR force a specific version against the install.sh on main.
AGENT_MEMORY_VERSION=vX.Y.Z curl -LsSf https://raw.githubusercontent.com/ben1787/agent-memory/main/install.sh | sh

# Verify it's on PATH.
agent-memory --version
```

## Pre-release sanity checks

Before tagging:

- [ ] `uv run pytest tests/ -q` is green locally
- [ ] `pyinstaller pyinstaller/agent-memory.spec --clean --noconfirm` builds
      successfully and `./dist/agent-memory/agent-memory --version` prints
      the new version
- [ ] `pyproject.toml` and `src/agent_memory/__init__.py` agree on the version
- [ ] You've smoke-tested `agent-memory init` + `save` + `recall` against a
      throwaway directory using the freshly-built binary

If any of those fail, fix them on `main` first — never tag a known-broken
revision, since the GitHub Release is hard to walk back gracefully.

## Walking back a bad release

If a tag was pushed but the binaries are broken:

```bash
# Delete the GitHub Release (does NOT delete the tag).
gh release delete vX.Y.Z -R ben1787/agent-memory --yes

# Delete the tag locally and on the remote.
git tag -d vX.Y.Z
git push origin :refs/tags/vX.Y.Z

# Fix the bug, commit, re-tag (you can reuse the same version number now
# that the previous tag is gone), push.
git tag vX.Y.Z
git push origin main vX.Y.Z
```

Users who already ran `agent-memory upgrade` against the bad release can
re-run it once you re-tag — `upgrade` always pulls the latest, not a
specific cached version.
