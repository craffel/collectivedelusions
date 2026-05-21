#!/bin/bash
#SBATCH --job-name=autoresearch-agent
#SBATCH --partition=hopper-cpu
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=6:00:00

echo "Starting Autoresearch Agent Job: $SLURM_JOB_ID"
echo "Allocated Node: $(hostname)"

# SKELETON_DIR holds the sandbox, wrappers, and bwrap. It is intentionally
# NOT inside the agent's CWD, so the agent cannot edit these files from
# inside the sandbox.
: "${SKELETON_DIR:?SKELETON_DIR must be set by the orchestrator (sbatch --export)}"

# Scope wrapped slurm commands inside the sandbox to this agent's tag.
export AGENT_ID="$SLURM_JOB_ID"

# Repeatedly invoke the agent with a 10-minute (600 seconds) delay
while true; do
    echo "[$(date)] Invoking claude..."

    # Run the agent autonomously inside a bwrap sandbox (only CWD is visible).
    CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 stdbuf -oL -eL "$SKELETON_DIR/sandbox_run.sh" /admin/home/craffel/.local/bin/claude --output-format stream-json --verbose --include-partial-messages --dangerously-skip-permissions --model "claude-opus-4-7" -p "Follow the instructions in research_plan.md" | stdbuf -oL jq --unbuffered -rj -f "$SKELETON_DIR/format_log.jq"

    echo "[$(date)] claude-cli invocation finished. Sleeping for 10 minutes."
    sleep 600
done
