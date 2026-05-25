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

source ~/.bashrc
source "$CONDA_ROOT/etc/profile.d/conda.sh"
source activate gemini

echo "Starting Review Agent Job: $SLURM_JOB_ID"
echo "Allocated Node: $(hostname)"

# SKELETON_DIR holds runtime files outside the agent's writable CWD.
: "${SKELETON_DIR:?SKELETON_DIR must be set by the orchestrator (sbatch --export)}"

# Sandbox requires an agent ID; reviewer should not submit GPU jobs, so cap at 0 nodes.
export AGENT_ID="$SLURM_JOB_ID"
export AGENT_NODE_CAP=0

# Stagger first invocation: see comment in run_research_agent.slurm.
offset="${AGENT_STARTUP_OFFSET:-0}"
if (( offset > 0 )); then
    echo "[$(date)] sleeping ${offset}s before invocation (startup stagger)"
    sleep "$offset"
fi

echo "[$(date)] Invoking gemini..."

# Run the agent autonomously inside a bwrap sandbox (only CWD is visible).
stdbuf -oL -eL "$SKELETON_DIR/sandbox_run.sh" /fsx/craffel/miniconda3/envs/gemini/bin/gemini --yolo --debug --model "gemini-3.5-flash" --prompt "Follow the instructions in reviewing_plan.md"

echo "[$(date)] gemini invocation finished."
