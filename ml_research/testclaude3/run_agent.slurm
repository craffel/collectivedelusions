#!/bin/bash
#SBATCH --job-name=autoresearch-agent
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=88
#SBATCH --gpus-per-node=8
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=6:00:00

echo "Starting Autoresearch Agent Job: $SLURM_JOB_ID"
echo "Allocated Node: $(hostname)"

# Repeatedly invoke the agent with a 10-minute (600 seconds) delay
while true; do
    echo "[$(date)] Invoking claude..."

    # Run the agent autonomously inside a bwrap sandbox (only CWD is visible).
    CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 stdbuf -oL -eL ./sandbox_run.sh /admin/home/craffel/.local/bin/claude --output-format stream-json --verbose --include-partial-messages --dangerously-skip-permissions --model "claude-opus-4-7" -p "Follow the instructions in plan.md" | stdbuf -oL jq --unbuffered -rj -f format_log.jq
    
    echo "[$(date)] claude-cli invocation finished. Sleeping for 10 minutes."
    sleep 600
done
