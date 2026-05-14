#!/bin/bash
# Filesystem sandbox for the autonomous claude agent.
# Exposes: CWD (rw), system dirs (ro), GPUs, OAuth credentials.
# Hides: everything else under $HOME and /fsx, plus past Claude transcripts.

set -euo pipefail

WORK="$(pwd)"
REAL_HOME="${HOME:-/admin/home/craffel}"
SANDBOX_HOME="$WORK/.sandbox_home"

mkdir -p "$SANDBOX_HOME"

exec bwrap \
  --ro-bind /usr /usr \
  --ro-bind /etc /etc \
  --ro-bind /opt /opt \
  --ro-bind /sys /sys \
  --symlink usr/bin   /bin \
  --symlink usr/sbin  /sbin \
  --symlink usr/lib   /lib \
  --symlink usr/lib64 /lib64 \
  --proc /proc \
  --dev-bind /dev /dev \
  --tmpfs /tmp \
  --tmpfs /run \
  --ro-bind /run/systemd/resolve /run/systemd/resolve \
  --bind "$SANDBOX_HOME" "$REAL_HOME" \
  --ro-bind "$REAL_HOME/.local" "$REAL_HOME/.local" \
  --bind "$REAL_HOME/.claude" "$REAL_HOME/.claude" \
  --bind "$REAL_HOME/.claude.json" "$REAL_HOME/.claude.json" \
  --tmpfs "$REAL_HOME/.claude/projects" \
  --tmpfs "$REAL_HOME/.claude/sessions" \
  --tmpfs "$REAL_HOME/.claude/backups" \
  --tmpfs "$REAL_HOME/.claude/shell-snapshots" \
  --tmpfs "$REAL_HOME/.claude/session-env" \
  --tmpfs "$REAL_HOME/.claude/cache" \
  --tmpfs "$REAL_HOME/.claude/downloads" \
  --tmpfs "$REAL_HOME/.claude/plugins" \
  --bind /dev/null "$REAL_HOME/.claude/history.jsonl" \
  --bind "$WORK" "$WORK" \
  --chdir "$WORK" \
  --setenv HOME "$REAL_HOME" \
  --die-with-parent \
  -- \
  "$@"
