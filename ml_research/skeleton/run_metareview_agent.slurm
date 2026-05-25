#!/bin/bash
#SBATCH --job-name=metareview-agent
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

echo "Starting Metareview Agent Job: $SLURM_JOB_ID"
echo "Allocated Node: $(hostname)"

# SKELETON_DIR holds runtime files outside the agent's writable CWD.
: "${SKELETON_DIR:?SKELETON_DIR must be set by the orchestrator (sbatch --export)}"

# Sandbox requires an agent ID; metareviewer should not submit GPU jobs, so cap at 0 nodes.
export AGENT_ID="$SLURM_JOB_ID"
export AGENT_NODE_CAP=0

echo "[$(date)] Invoking gemini..."

# Run the agent autonomously inside a bwrap sandbox (only CWD is visible).
stdbuf -oL -eL "$SKELETON_DIR/sandbox_run.sh" /fsx/craffel/miniconda3/envs/gemini/bin/gemini --yolo --debug --model "gemini-3.5-flash" --prompt "Follow the instructions in metareview_plan.md"

echo "[$(date)] gemini invocation finished."
