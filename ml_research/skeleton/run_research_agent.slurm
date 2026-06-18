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

    # Determine the current phase from progress.json (default to 1 if not found)
    PHASE=$(python3 -c "
import json, os
try:
    if os.path.exists('progress.json'):
        with open('progress.json') as f:
            data = json.load(f)
            print(data.get('phase', '1'))
    else:
        print('1')
except Exception:
    print('1')
")

    if [ "$PHASE" == "1" ]; then
        PLAN="ideator_plan.md"
    elif [ "$PHASE" == "2" ]; then
        PLAN="experimenter_plan.md"
    elif [ "$PHASE" == "3" ]; then
        PLAN="writer_plan.md"
    else
        # If phase is 4 or higher, could re-invoke writer for refinement or ideator for restart, 
        # but let's assume phase 4 just starts over or refines.
        # We will default to writer_plan.md if it's phase 4 since refinement usually involves writing or doing more experiments.
        # To strictly follow the plan:
        PLAN="writer_plan.md"
    fi

    echo "[$(date)] Current phase is $PHASE, using plan $PLAN"

    if [ "$PHASE" == "completed" ]; then
        echo "[$(date)] Phase is 'completed', exiting agent loop."
        break
    fi

    # Run the agent autonomously inside a bwrap sandbox (only CWD is visible).
    stdbuf -oL -eL "$SKELETON_DIR/sandbox_run.sh" /fsx/craffel/miniconda3/envs/gemini/bin/gemini --yolo --debug --model "gemini-3.5-flash" --prompt "Follow the instructions in $PLAN"

    echo "[$(date)] gemini invocation finished. Sleeping for 1 minute."
    sleep 60
done
