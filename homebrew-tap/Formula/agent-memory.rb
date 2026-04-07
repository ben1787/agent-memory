# typed: false
# frozen_string_literal: true
#
# Homebrew formula for agent-memory.
#
# This is the seed file for a separate homebrew tap repo at:
#   github.com/<OWNER>/homebrew-tap
#
# The release.yml workflow auto-bumps version + sha256 on every tag, so this
# file only needs to exist on day one to prove the formula compiles. After
# that, every `git tag vX.Y.Z` updates the live formula in the tap repo
# automatically.
#
# Users install with:
#   brew install <OWNER>/tap/agent-memory
#
# OR explicitly tap first:
#   brew tap <OWNER>/tap
#   brew install agent-memory
class AgentMemory < Formula
  desc "Local-first associative memory store for AI coding agents"
  homepage "https://github.com/ben1787/agent-memory"
  version "0.1.0"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/ben1787/agent-memory/releases/download/v0.1.0/agent-memory-macos-arm64.tar.gz"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"
    else
      url "https://github.com/ben1787/agent-memory/releases/download/v0.1.0/agent-memory-macos-x86_64.tar.gz"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/ben1787/agent-memory/releases/download/v0.1.0/agent-memory-linux-arm64.tar.gz"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"
    else
      url "https://github.com/ben1787/agent-memory/releases/download/v0.1.0/agent-memory-linux-x86_64.tar.gz"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"
    end
  end

  def install
    # The tarball extracts to ./agent-memory/ — a PyInstaller onedir bundle
    # with the binary alongside its native libs and bundled data files.
    # Drop the whole bundle into libexec, then symlink the bootloader into bin.
    libexec.install Dir["agent-memory/*"]
    bin.install_symlink libexec/"agent-memory"
  end

  test do
    # Smoke-test: --version should print and exit clean.
    system "#{bin}/agent-memory", "--version"
  end
end
