#!/bin/bash
# Filesystem sandbox for the autonomous claude agent.
# Exposes: CWD (rw), system dirs (ro), GPUs, OAuth credentials.
# Hides: everything else under $HOME and /fsx, plus past Claude transcripts.

set -euo pipefail

WORK="$(pwd)"
REAL_HOME="${HOME:-/admin/home/craffel}"
SANDBOX_HOME="$WORK/.sandbox_home"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BWRAP="$SCRIPT_DIR/bwrap"
SLURM_BIN_DIR="${SLURM_BIN_DIR:-/opt/slurm/bin}"  # host path to real sbatch/scancel/squeue
WRAPPER_BIN_DIR="$SCRIPT_DIR/bin"                  # tag-enforcing wrappers

# AGENT_ID is required: it scopes slurm submission/cancel for this agent.
: "${AGENT_ID:?AGENT_ID must be set by the caller before invoking sandbox_run.sh}"

mkdir -p "$SANDBOX_HOME"

exec "$BWRAP" \
  --unshare-all \
  --share-net \
  --as-pid-1 \
  --new-session \
  --die-with-parent \
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
  --ro-bind /run/munge /run/munge \
  --ro-bind /var/lib/sss/pipes /var/lib/sss/pipes \
  --bind "$SANDBOX_HOME" "$REAL_HOME" \
  --ro-bind "$REAL_HOME/.local" "$REAL_HOME/.local" \
  --bind "$REAL_HOME/.gemini" "$REAL_HOME/.gemini" \
  --tmpfs "$REAL_HOME/.gemini/history" \
  --tmpfs "$REAL_HOME/.gemini/tmp" \
  --bind "$WORK" "$WORK" \
  --ro-bind /fsx/craffel/miniconda3 /fsx/craffel/miniconda3 \
  --ro-bind /admin/slurm /admin/slurm \
  --ro-bind "$SLURM_BIN_DIR" /run/slurm-real/bin \
  --ro-bind "$WRAPPER_BIN_DIR/sbatch"  "$SLURM_BIN_DIR/sbatch" \
  --ro-bind "$WRAPPER_BIN_DIR/scancel" "$SLURM_BIN_DIR/scancel" \
  --ro-bind "$WRAPPER_BIN_DIR/squeue"  "$SLURM_BIN_DIR/squeue" \
  --chdir "$WORK" \
  --setenv HOME "$REAL_HOME" \
  --setenv AGENT_ID "$AGENT_ID" \
  --setenv REAL_SLURM_BIN /run/slurm-real/bin \
  -- \
  "$@"
