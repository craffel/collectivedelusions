#!/bin/bash
#SBATCH --job-name=review-agent
#SBATCH --partition=hopper-cpu
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=1:00:00

echo "Starting Review Agent Job: $SLURM_JOB_ID"
echo "Allocated Node: $(hostname)"

# SKELETON_DIR holds runtime files outside the agent's writable CWD.
: "${SKELETON_DIR:?SKELETON_DIR must be set by the orchestrator (sbatch --export)}"

# Sandbox requires an agent ID; reviewer should not submit GPU jobs, so cap at 0 nodes.
export AGENT_ID="$SLURM_JOB_ID"
export AGENT_NODE_CAP=0

echo "[$(date)] Invoking claude..."

# Run the agent autonomously inside a bwrap sandbox (only CWD is visible).
CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 stdbuf -oL -eL "$SKELETON_DIR/sandbox_run.sh" /admin/home/craffel/.local/bin/claude --output-format stream-json --verbose --include-partial-messages --dangerously-skip-permissions --model "claude-opus-4-7" -p "Follow the instructions in reviewing_plan.md" | stdbuf -oL jq --unbuffered -rj -f "$SKELETON_DIR/format_log.jq"

echo "[$(date)] claude-cli invocation finished."
