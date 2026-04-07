# Releasing agent-memory

This file documents how to cut a release and how to wire up the optional
Homebrew tap auto-bump. The `release` workflow at
`.github/workflows/release.yml` does almost everything automatically — the
only manual step that ever needs touching is the one-time PAT setup for the
Homebrew tap.

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
   (x86_64). Each runner builds a single self-contained binary using
   `pyinstaller pyinstaller/agent-memory.spec` and produces a sha256 file.

2. **Release publish** uploads all five binaries plus their `.sha256` files
   to a new GitHub Release at `vX.Y.Z`. This is the canonical source of
   truth — `install.sh`, the Homebrew formula, and `agent-memory upgrade`
   all download from these release assets.

3. **Homebrew tap bump** (optional, see below) regenerates
   `Formula/agent-memory.rb` in the tap repo with the new version + sha256
   values and pushes the commit. Runs only on real release tags (not
   `vX.Y.Z-rc.N` prereleases) and only if `HOMEBREW_TAP_TOKEN` is set.

You can watch the workflow at:
https://github.com/ben1787/agent-memory/actions

## Verifying the install paths after a release

```bash
# Curl install.sh from the freshly-tagged release.
curl -LsSf https://raw.githubusercontent.com/ben1787/agent-memory/vX.Y.Z/install.sh | sh

# OR force a specific version.
AGENT_MEMORY_VERSION=vX.Y.Z curl -LsSf https://raw.githubusercontent.com/ben1787/agent-memory/main/install.sh | sh

# Verify it's on PATH.
agent-memory --version

# Verify Homebrew (after tap bump has landed).
brew install ben1787/tap/agent-memory
agent-memory --version
```

## One-time setup: Homebrew tap auto-bump

The `bump-homebrew-tap` job in `release.yml` pushes a commit to a separate
repo (`ben1787/homebrew-tap`). The default `GITHUB_TOKEN` that Actions
provides is scoped only to the agent-memory repo, so we need a separate
token with push access to the tap repo.

This is **optional**. If you skip this setup, the release still publishes
binaries — only the Homebrew formula stops auto-updating, and users on the
tap will need to wait for a manual bump (or you can run the bump locally).

### Steps

1. Create a fine-grained personal access token at
   https://github.com/settings/tokens?type=beta with:
   - **Resource owner**: ben1787
   - **Repository access**: only `ben1787/homebrew-tap`
   - **Permissions** → Repository:
     - Contents: Read and write
     - Metadata: Read-only (auto-selected)
   - Expiration: whatever feels right (1 year is sane)

2. Copy the generated token (it will only be shown once).

3. Add it as a secret on the agent-memory repo:

   ```bash
   gh secret set HOMEBREW_TAP_TOKEN -R ben1787/agent-memory
   # paste the token when prompted
   ```

   Or via the web UI:
   https://github.com/ben1787/agent-memory/settings/secrets/actions/new

4. Done. The next non-prerelease tag push will auto-bump the tap formula.

### Alternative: skip auto-bump and bump manually after each release

If you'd rather not maintain a token, leave `HOMEBREW_TAP_TOKEN` unset and
update the tap by hand once per release:

```bash
TAG=vX.Y.Z
TAP=$(mktemp -d)
git clone git@github.com:ben1787/homebrew-tap.git "$TAP"

# Pull the four release artifact shas.
for asset in macos-arm64 macos-x86_64 linux-x86_64 linux-arm64; do
  curl -LsSO "https://github.com/ben1787/agent-memory/releases/download/${TAG}/agent-memory-${asset}.sha256"
done

# Patch the formula by hand using the printed shas, commit, push.
$EDITOR "$TAP/Formula/agent-memory.rb"
cd "$TAP" && git commit -am "agent-memory ${TAG}" && git push
```

## Pre-release sanity checks

Before tagging:

- [ ] `uv run pytest tests/ -q` is green locally
- [ ] `pyinstaller pyinstaller/agent-memory.spec --clean --noconfirm` builds
      successfully and `./dist/agent-memory --version` prints the new version
- [ ] `pyproject.toml` and `src/agent_memory/__init__.py` agree on the version
- [ ] You've smoke-tested `agent-memory init` + `save` + `recall` against a
      throwaway directory using the freshly-built binary

If any of those fail, fix them on `main` first — never tag a known-broken
revision, since the GitHub Release is hard to walk back gracefully.
