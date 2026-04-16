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
#SBATCH --time=24:00:00

# We don't use 'set -e' here so that if an agent invocation fails,
# the script will still sleep and retry rather than terminating the whole job.
source ~/.bashrc
source "$CONDA_ROOT/etc/profile.d/conda.sh"
source activate gemini

echo "Starting Autoresearch Agent Job: $SLURM_JOB_ID"
echo "Allocated Node: $(hostname)"

# Repeatedly invoke the agent with a 10-minute (600 seconds) delay
while true; do
    echo "[$(date)] Invoking gemini-cli..."
    
    # Run the agent autonomously
    gemini --yolo --debug --prompt "Read the instructions in plan.md"
    
    echo "[$(date)] gemini-cli invocation finished. Sleeping for 10 minutes."
    sleep 600
done
