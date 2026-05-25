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

source ~/.bashrc
source "$CONDA_ROOT/etc/profile.d/conda.sh"
source activate gemini


echo "Starting Autoresearch Agent Job: $SLURM_JOB_ID"
echo "Allocated Node: $(hostname)"

# SKELETON_DIR holds the sandbox, wrappers, and bwrap. It is intentionally
# NOT inside the agent's CWD, so the agent cannot edit these files from
# inside the sandbox.
: "${SKELETON_DIR:?SKELETON_DIR must be set by the orchestrator (sbatch --export)}"

# Scope wrapped slurm commands inside the sandbox to this agent's tag.
export AGENT_ID="$SLURM_JOB_ID"

# Stagger the first claude invocation so co-located agents don't all clobber
# ~/.claude.json at the same instant. AGENT_STARTUP_OFFSET is set per-submission
# by the orchestrator; defaults to 0 if invoked outside the orchestrator.
offset="${AGENT_STARTUP_OFFSET:-0}"
if (( offset > 0 )); then
    echo "[$(date)] sleeping ${offset}s before first invocation (startup stagger)"
    sleep "$offset"
fi

# Repeatedly invoke the agent with a 10-minute (600 seconds) delay
while true; do
    echo "[$(date)] Invoking gemini..."

    # Run the agent autonomously inside a bwrap sandbox (only CWD is visible).
    stdbuf -oL -eL "$SKELETON_DIR/sandbox_run.sh" /fsx/craffel/miniconda3/envs/gemini/bin/gemini --yolo --debug --model "gemini-3.5-flash" --prompt "Follow the instructions in research_plan.md"

    echo "[$(date)] gemini invocation finished. Sleeping for 10 minutes."
    sleep 600
done
